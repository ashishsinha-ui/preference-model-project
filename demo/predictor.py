"""Shared inference core for the Streamlit demo (demo/app.py) and the API (demo/api.py).

For one (prompt, response A, response B) it
  1. scores both orders: p_fwd = P(A | A,B) and p_swp = P(A | B,A) (the swap test),
  2. symmetrizes: p_sym = (p_fwd + 1 - p_swp) / 2 (removes position bias, +~1 point accuracy),
  3. calibrates: temperature scaling fitted on the model's validation set (results/<run>/calibration.json),
and raises the review flags used in the paper: low confidence, order sensitivity, truncation.
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tokenizers are cached; avoids the server's flaky network
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from src.data.dataset import PairDataset  # noqa: E402
from src.models.cross_encoder import load_checkpoint  # noqa: E402
from src.utils.tokenization import MAX_RESP  # noqa: E402

hf_logging.set_verbosity_error()

MODELS = {  # label shown in the UI -> run name (checkpoints/<run>, results/<run>)
    "DeBERTa-v3 cross-encoder - HH helpfulness (best accuracy)": "t3_deberta_hard_hh_seed42",
    "DeBERTa-v3 cross-encoder - SHP Reddit, soft labels (best calibrated)": "t3_deberta_soft_shp_seed42",
    "BERT bi-encoder - HH helpfulness (position-invariant by design)": "t5_bert-bi_hard_hh_seed42",
}
DEFAULT_MODEL = next(iter(MODELS))
LOW_CONFIDENCE = 0.60    # calibrated confidence below this -> "needs human review"
ORDER_GAP = 0.10         # |p_fwd - (1 - p_swp)| above this -> "order-sensitive"


def _device():
    want = os.environ.get("DEMO_DEVICE", "auto")
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=3)
def get_model(run):
    model, tok, cfg = load_checkpoint(ROOT / "checkpoints" / run, device=_device())
    cal = json.load(open(ROOT / "results" / run / "calibration.json"))
    return model, tok, cfg, float(cal["temperature_sym"])


@torch.no_grad()
def _p_a(model, tok, cfg, prompt, a, b):
    df = pd.DataFrame([{"prompt": prompt, "response_a": a, "response_b": b, "label": 1, "p_a": 1.0}])
    batch = model.make_collator(tok)([PairDataset(df, tok, max_prompt=cfg["max_prompt"], max_resp=MAX_RESP)[0]])
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg["bf16"] and device.type == "cuda"):
        return float(torch.sigmoid(model(batch).float()).item())


def predict(prompt, response_a, response_b, run=None):
    run = run or MODELS[DEFAULT_MODEL]
    model, tok, cfg, temperature = get_model(run)
    p_fwd = _p_a(model, tok, cfg, prompt, response_a, response_b)
    p_swp = _p_a(model, tok, cfg, prompt, response_b, response_a)
    p_sym = (p_fwd + 1 - p_swp) / 2
    z = np.log(np.clip(p_sym, 1e-6, 1 - 1e-6) / np.clip(1 - p_sym, 1e-6, 1))
    p_cal = float(1 / (1 + np.exp(-z / temperature)))
    confidence = max(p_cal, 1 - p_cal)
    order_gap = abs(p_fwd - (1 - p_swp))

    n_tok = {k: len(tok.encode(t, add_special_tokens=False))
             for k, t in [("prompt", prompt), ("a", response_a), ("b", response_b)]}
    words_a, words_b = len(response_a.split()), len(response_b.split())
    winner = "A" if p_cal > 0.5 else "B"
    return {
        "winner": winner,
        "p_a": p_cal,
        "confidence": confidence,
        "needs_review": confidence < LOW_CONFIDENCE or order_gap > ORDER_GAP,
        "bias_checks": {
            "p_a_order_ab": p_fwd,
            "p_a_order_ba": 1 - p_swp,
            "order_gap": order_gap,
            "order_sensitive": order_gap > ORDER_GAP,
            "decision_flips_with_order": (p_fwd > 0.5) != (p_swp < 0.5),
            "words_a": words_a,
            "words_b": words_b,
            "winner_is_longer": (words_a > words_b) == (winner == "A") and words_a != words_b,
            "length_ratio": max(words_a, words_b) / max(1, min(words_a, words_b)),
            "truncated": {"prompt": bool(cfg["max_prompt"]) and n_tok["prompt"] > cfg["max_prompt"],
                          "a": n_tok["a"] > MAX_RESP, "b": n_tok["b"] > MAX_RESP},
        },
        "model": run,
        "temperature": temperature,
        "token_budget": {"prompt": cfg["max_prompt"], "response": MAX_RESP},
    }
