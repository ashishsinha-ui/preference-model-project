"""Shared evaluation runner: every model tier calls evaluate_model() with its predict function.

predict_fn(df) -> array of P(A preferred). For each eval set we predict in both A/B orders and save
  results/<run_name>/preds_<set>.parquet   uid, p_a, p_a_swapped   (reused by calibration/ensembles)
  results/<run_name>/metrics.json
and log the headline numbers to MLflow.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_split
from src.evaluation import metrics as M
from src.utils import tracking

EVAL_SETS = {  # name -> (dataset, split)
    "hh_val": ("hh", "val"),
    "hh_test": ("hh", "test"),
    "hh_harmless": ("hh", "harmless_shift"),
    "shp_val": ("shp", "val_full"),
    "shp_test": ("shp", "test"),
}


def swap(df):
    return df.rename(columns={"response_a": "response_b", "response_b": "response_a",
                              "len_a": "len_b", "len_b": "len_a"})


def evaluate_set(df, p, p_swp):
    y = df.label.to_numpy()
    out = M.classification_metrics(p, y)
    out["accuracy_ci95"] = M.bootstrap_ci(((p > 0.5) == y)[y >= 0])
    out.update(M.swap_metrics(p, p_swp, y))
    out.update(M.length_bias(p, df.len_a.to_numpy(), df.len_b.to_numpy(), y))
    if "high_disagreement" in df:  # SHP: compare against the human vote share
        h, hd = df.p_a.to_numpy(), df.high_disagreement.to_numpy()
        out.update(M.soft_metrics(p, h))
        out["ece_high_disagreement"] = M.ece(p[hd], y[hd])
        out["kl_high_disagreement"] = M.binary_kl(h[hd], p[hd])
        out["mean_abs_dev_high_disagreement"] = float(np.abs(p[hd] - 0.5).mean())
    group = df["subset"] if "subset" in df else df["domain"]
    correct = pd.Series((p > 0.5) == y, index=df.index)
    out["per_domain_accuracy"] = {k: float(v) for k, v in correct.groupby(group).mean().items()}
    out["n"] = int(len(df))
    return out


def evaluate_model(predict_fn, run_name, sets=tuple(EVAL_SETS), out_dir="results", max_n=None):
    """max_n: evaluate on a fixed random subsample of each set (used by the slow LLM judge)."""
    out = Path(out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in sets:
        df = load_split(*EVAL_SETS[name], subset_size=max_n)
        p = np.asarray(predict_fn(df), float)
        p_swp = np.asarray(predict_fn(swap(df)), float)
        pd.DataFrame({"uid": df.uid, "p_a": p, "p_a_swapped": p_swp}).to_parquet(
            out / f"preds_{name}.parquet", index=False)
        res = evaluate_set(df, p, p_swp)
        results[name] = res
        tracking.log({f"{name}/{k}": v for k, v in res.items()
                      if isinstance(v, (int, float)) and np.isfinite(v)})
        print(f"{name:12s} n={res['n']:6d} acc={res['accuracy']:.4f} auc={res['auc']:.4f} "
              f"brier={res['brier']:.4f} ece={res['ece']:.4f} swap_cons={res['swap_consistency']:.3f} "
              f"pos_bias={res['position_bias']:+.3f} len_r={res['length_r_model']:+.3f} "
              f"(human {res['length_r_human']:+.3f})")
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    tracking.log_file(out / "metrics.json")
    return results
