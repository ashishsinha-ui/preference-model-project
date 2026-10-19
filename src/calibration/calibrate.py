"""Post-hoc calibration and positional debiasing from saved predictions (CPU only, no retraining).

For a run in results/<run_name>/, each calibrator is fit on the validation predictions of the
dataset the model was trained on, then applied to every eval set and all metrics are recomputed.
  raw       the model's own probabilities
  sym       positional debiasing: p = (p_AB + 1 - p_BA) / 2   (no fitting)
  temp      temperature scaling
  sym+temp  both
  isotonic  isotonic regression
Note: the *_val row of the training dataset is in-sample for the fitted methods.

  python -m src.calibration.calibrate t2_bert_hard_hh_seed42 t1_bilstm-cross_hard_hh_seed42
  python -m src.calibration.calibrate --all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.temperature_scaling import TemperatureScaler
from src.data.dataset import load_split
from src.evaluation.evaluate import EVAL_SETS, evaluate_set
from src.utils import tracking

KEYS = ["accuracy", "nll", "brier", "ece", "swap_consistency", "length_r_model",
        "kl_to_human", "ece_high_disagreement"]


def load_preds(run_dir, set_name):
    """Eval split joined with saved predictions: p_a = human label/vote share, p_a_model = model."""
    df = load_split(*EVAL_SETS[set_name])
    preds = pd.read_parquet(run_dir / f"preds_{set_name}.parquet")
    return df.merge(preds, on="uid", how="inner", suffixes=("", "_model"))


def calibrate_run(run_name, results_dir="results", track=True):
    run_dir = Path(results_dir) / run_name
    dataset = run_name.split("_")[-2]  # ..._<label>_<dataset>_seed42
    val = load_preds(run_dir, f"{dataset}_val")
    yv, pv, sv = val.label.to_numpy(), val.p_a_model.to_numpy(), val.p_a_swapped.to_numpy()
    temp = TemperatureScaler().fit(pv, yv)
    temp_sym = TemperatureScaler().fit((pv + 1 - sv) / 2, yv)
    iso = IsotonicCalibrator().fit(pv, yv)

    methods = {  # (p_AB, p_BA) -> calibrated (p_AB, p_BA)
        "raw": lambda p, s: (p, s),
        "sym": lambda p, s: ((p + 1 - s) / 2, (s + 1 - p) / 2),
        "temp": lambda p, s: (temp.transform(p), temp.transform(s)),
        "sym+temp": lambda p, s: (temp_sym.transform((p + 1 - s) / 2), temp_sym.transform((s + 1 - p) / 2)),
        "isotonic": lambda p, s: (iso.transform(p), iso.transform(s)),
    }
    out, rows = {"temperature": temp.T, "temperature_sym": temp_sym.T, "sets": {}}, []
    tracking.start_run(f"{run_name}+calib", {"base_run": run_name, "fit_on": f"{dataset}_val"}, enabled=track)
    tracking.log({"temperature": temp.T, "temperature_sym": temp_sym.T})
    for set_name in EVAL_SETS:
        if not (run_dir / f"preds_{set_name}.parquet").exists():
            continue
        df = load_preds(run_dir, set_name)
        p, s = df.p_a_model.to_numpy(), df.p_a_swapped.to_numpy()
        out["sets"][set_name] = {}
        for method, fn in methods.items():
            res = evaluate_set(df, *fn(p, s))
            res.pop("per_domain_accuracy")
            out["sets"][set_name][method] = res
            rows.append({"set": set_name, "method": method, **{k: res.get(k, np.nan) for k in KEYS}})
            tracking.log({f"{set_name}/{method}/{k}": res[k] for k in KEYS
                          if k in res and np.isfinite(res[k])})
    tracking.finish()

    with open(run_dir / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n### {run_name}   T={temp.T:.3f}   T_sym={temp_sym.T:.3f}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every run in results/ that has predictions")
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()
    runs = args.runs or []
    if args.all:
        runs = sorted(p.parent.name for p in Path("results").glob("*/preds_hh_test.parquet"))
    for run in runs:
        calibrate_run(run, track=not args.no_track)


if __name__ == "__main__":
    main()
