"""Main training entry point for the transformer cross-encoders (T2 BERT, T3/T4 DeBERTa).

  python -m src.training.train --config configs/bert_base.yaml
  python -m src.training.train --config configs/deberta_base.yaml dataset=shp label_type=soft
Anything after the config is a key=value override of the YAML.
Saves checkpoints/<run_name>/{config.yaml, model.pt} and results/<run_name>/.
"""
import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.data.dataset import PairDataset, load_split
from src.evaluation.evaluate import evaluate_model
from src.models.cross_encoder import build_model
from src.training.trainer import make_predict_fn, train
from src.utils import tracking
from src.utils.seed import set_seed
from src.utils.tokenization import MAX_RESP

FLOAT_KEYS = ["lr", "weight_decay", "warmup_ratio", "swap_prob", "dropout"]


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise KeyError(f"unknown config key: {key}")
        cfg[key] = yaml.safe_load(value)
    for key in FLOAT_KEYS:
        cfg[key] = float(cfg[key])
    return cfg


def run_name(cfg):
    tags = (("-swapaug" if cfg["swap_prob"] > 0 else "")
            + ("-noprompt" if cfg["max_prompt"] == 0 else "")
            + (f"-dev{cfg['subset_size']}" if cfg["subset_size"] else ""))
    return f"{cfg['tier']}_{cfg['short_name']}{tags}_{cfg['label_type']}_{cfg['dataset']}_seed{cfg['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no_track", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value overrides of the YAML config")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    hf_logging.set_verbosity_error()
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg["seed"])
    name = run_name(cfg)
    ckpt_dir = Path("checkpoints") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    tracking.start_run(name, cfg, enabled=not args.no_track)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds_kw = dict(max_prompt=cfg["max_prompt"], max_resp=MAX_RESP)
    train_ds = PairDataset(load_split(cfg["dataset"], "train", cfg["subset_size"], cfg["seed"]), tok,
                           cfg["label_type"], swap_prob=cfg["swap_prob"], **ds_kw)
    val_ds = PairDataset(load_split(cfg["dataset"], "val"), tok, cfg["label_type"], **ds_kw)
    model = build_model(cfg)  # cfg["arch"]: cross (default) or bi
    collate = model.make_collator(tok)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params / 1e6:.0f}M params | train={len(train_ds)} val={len(val_ds)}")
    tracking.log({"n_params_m": n_params / 1e6})

    model = train(model, train_ds, val_ds, collate, ckpt_dir / "model.pt",
                  epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                  weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
                  grad_accum=cfg["grad_accum"], eval_every=cfg["eval_every"], patience=cfg["patience"],
                  bf16=cfg["bf16"], param_groups=model.param_groups(cfg["weight_decay"]))
    evaluate_model(make_predict_fn(model, tok, collate, batch_size=cfg["eval_batch_size"],
                                   bf16=cfg["bf16"], **ds_kw), name)
    tracking.finish()


if __name__ == "__main__":
    main()
