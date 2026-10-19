"""Deep ensemble (Lakshminarayanan et al., 2017) from saved predictions (CPU only).

Averages the logits of independently trained members (different seeds); the spread of member
probabilities is the uncertainty estimate, analysed like MC dropout (uncertainty.json).

  python -m src.calibration.ensemble t2_bert_hard_hh_seed42 t2_bert_hard_hh_seed1 t2_bert_hard_hh_seed2
Writes results/<tier>_<model>-ens<N>_<label>_<dataset>_seed42/ in the standard run layout, so
calibrate.py works on it too.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.dataset import load_split
from src.evaluation.evaluate import EVAL_SETS, evaluate_set
from src.utils import tracking


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ensemble_name(member):
    parts = member.split("_")
    parts[1] = parts[1].split("-")[0] + "-ens"  # e.g. bert -> bert-ens
    parts[-1] = "seed42"
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    name = ensemble_name(args.members[0]).replace("-ens", f"-ens{len(args.members)}")
    out_dir = Path("results") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tracking.start_run(name, {"members": ",".join(args.members)}, enabled=not args.no_track)

    metrics, unc = {}, {}
    for set_name in EVAL_SETS:
        files = [Path("results") / m / f"preds_{set_name}.parquet" for m in args.members]
        if not all(f.exists() for f in files):
            continue
        df = load_split(*EVAL_SETS[set_name])
        preds = [df[["uid"]].merge(pd.read_parquet(f), on="uid", how="left") for f in files]
        P = np.stack([x.p_a.to_numpy() for x in preds])
        S = np.stack([x.p_a_swapped.to_numpy() for x in preds])
        p = 1 / (1 + np.exp(-_logit(P).mean(0)))
        p_swp = 1 / (1 + np.exp(-_logit(S).mean(0)))
        std = P.std(0)
        pd.DataFrame({"uid": df.uid, "p_a": p, "p_a_swapped": p_swp, "std": std}).to_parquet(
            out_dir / f"preds_{set_name}.parquet", index=False)

        res = evaluate_set(df, p, p_swp)
        y = df.label.to_numpy()
        u = {"mean_std": float(std.mean()), "auroc_error": float(roc_auc_score((p > 0.5) != y, std))}
        if "high_disagreement" in df:
            hd = df.high_disagreement.to_numpy()
            u["auroc_hd_std"] = float(roc_auc_score(hd, std))
            u["auroc_hd_conf"] = float(roc_auc_score(hd, -np.abs(p - 0.5)))
            u["frac_hd_in_flagged_0.4_0.6"] = float(hd[(p > 0.4) & (p < 0.6)].mean())
            u["frac_hd_overall"] = float(hd.mean())
        metrics[set_name], unc[set_name] = res, u
        tracking.log({f"{set_name}/{k}": v for k, v in {**res, **u}.items()
                      if isinstance(v, (int, float)) and np.isfinite(v)})
        print(f"{set_name:12s} acc={res['accuracy']:.4f} ece={res['ece']:.4f} nll={res['nll']:.4f} "
              f"swap={res['swap_consistency']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in u.items()))

    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    json.dump(unc, open(out_dir / "uncertainty.json", "w"), indent=2)
    tracking.finish()
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
