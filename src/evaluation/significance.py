"""Paired bootstrap significance tests between two models (or one model before/after a post-hoc
method) on the same eval pairs.

A side is "run" or "run@method", method in {raw, sym, temp}:
  sym  = average of both A/B orders (positional debiasing)
  temp = temperature scaling fit on the run's own validation set
Pairs shared by both sides are resampled jointly (2000 times). Reported per metric: both values,
the difference A - B, its 95% CI and a two-sided p-value. Comparisons whose predictions are missing
are skipped.

  python -m src.evaluation.significance        # -> results/significance.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.calibrate import load_preds
from src.calibration.temperature_scaling import TemperatureScaler
from src.evaluation import metrics as M

N_BOOT = 2000
HIGHER_IS_BETTER = {"accuracy"}
SOFT = ["accuracy", "ece", "kl_to_human", "ece_high_disagreement", "kl_high_disagreement"]

COMPARISONS = [  # (question, eval set, side A, side B, metrics)
    ("RQ1 small fine-tuned model vs LLM judge", "hh_test",
     "t3_deberta-ens3_hard_hh_seed42", "t6_llama3-8b-3shot_judge_hh_seed42", ["accuracy", "ece"]),
    ("RQ1 same, both symmetrized", "hh_test",
     "t3_deberta-ens3_hard_hh_seed42@sym", "t6_llama3-8b-3shot_judge_hh_seed42@sym", ["accuracy"]),
    ("RQ1 DeBERTa vs BERT", "hh_test", "t3_deberta_hard_hh_seed42", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("RQ2 symmetrization (BERT)", "hh_test", "t2_bert_hard_hh_seed42@sym", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("RQ2 symmetrization (DeBERTa)", "hh_test", "t3_deberta_hard_hh_seed42@sym", "t3_deberta_hard_hh_seed42", ["accuracy"]),
    ("RQ2 swap augmentation (BERT)", "hh_test", "t2_bert-swapaug_hard_hh_seed42", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("Bi-encoder vs cross-encoder (BERT)", "hh_test", "t5_bert-bi_hard_hh_seed42", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("Ablation: no prompt vs prompt (BERT)", "hh_test", "t2_bert-noprompt_hard_hh_seed42", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("RQ3 soft vs hard (BiLSTM)", "shp_test", "t1_bilstm-cross_soft_shp_seed42", "t1_bilstm-cross_hard_shp_seed42", SOFT),
    ("RQ3 soft vs hard (BERT)", "shp_test", "t2_bert_soft_shp_seed42", "t2_bert_hard_shp_seed42", SOFT),
    ("RQ3 soft vs hard (DeBERTa)", "shp_test", "t3_deberta_soft_shp_seed42", "t3_deberta_hard_shp_seed42", SOFT),
    ("RQ3 soft vs temperature-scaled hard (BERT)", "shp_test",
     "t2_bert_soft_shp_seed42", "t2_bert_hard_shp_seed42@temp", SOFT),
    ("RQ3 soft vs temperature-scaled hard (DeBERTa)", "shp_test",
     "t3_deberta_soft_shp_seed42", "t3_deberta_hard_shp_seed42@temp", SOFT),
    ("Reverse domain shift: harmless-trained vs helpful-trained on harmless", "hhh_test",
     "t2_bert_hard_hhh_seed42", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("Reverse domain shift: harmless-trained vs helpful-trained on helpful", "hh_test",
     "t2_bert_hard_hhh_seed42", "t2_bert_hard_hh_seed42", ["accuracy"]),
    ("AI labels vs human labels, unseen human data (SHP)", "shp_test",
     "t2_bert_hard_uf_seed42", "t2_bert_hard_hh_seed42", ["accuracy", "ece"]),
    ("AI labels vs human labels, on HH human labels", "hh_test",
     "t2_bert_hard_uf_seed42", "t2_bert_hard_hh_seed42", ["accuracy", "ece"]),
]


def side(spec, set_name):
    """-> (eval rows with uid/labels, P(A) under the requested method)."""
    run, _, method = spec.partition("@")
    df = load_preds(Path("results") / run, set_name)
    p, s = df.p_a_model.to_numpy(), df.p_a_swapped.to_numpy()
    if method == "sym":
        p = (p + 1 - s) / 2
    elif method == "temp":
        val = load_preds(Path("results") / run, f"{run.split('_')[-2]}_val")
        p = TemperatureScaler().fit(val.p_a_model, val.label).transform(p)
    return df.drop(columns=["p_a_model", "p_a_swapped"]), p


def metric(name, p, a):
    y = a["y"]
    if name == "accuracy":
        return float(((p > 0.5) == y).mean())
    if name == "ece":
        return M.ece(p, y)
    hd = a["hd"]
    return {"kl_to_human": lambda: M.binary_kl(a["h"], p),
            "ece_high_disagreement": lambda: M.ece(p[hd], y[hd]),
            "kl_high_disagreement": lambda: M.binary_kl(a["h"][hd], p[hd])}[name]()


def compare(set_name, spec_a, spec_b, metrics, rng):
    da, pa = side(spec_a, set_name)
    db, pb = side(spec_b, set_name)
    both = da[["uid"]].reset_index().merge(db[["uid"]].reset_index(), on="uid", suffixes=("_a", "_b"))
    ia, ib = both.index_a.to_numpy(), both.index_b.to_numpy()
    pa, pb, rows = pa[ia], pb[ib], da.iloc[ia]
    arrays = {"y": rows.label.to_numpy(), "h": rows.p_a.to_numpy(),
              "hd": rows.high_disagreement.to_numpy() if "high_disagreement" in rows else None}
    idx = rng.integers(0, len(ia), size=(N_BOOT, len(ia)))
    out = []
    for m in metrics:
        va, vb = metric(m, pa, arrays), metric(m, pb, arrays)
        diffs = np.array([metric(m, pa[i], {k: (v[i] if v is not None else None) for k, v in arrays.items()})
                          - metric(m, pb[i], {k: (v[i] if v is not None else None) for k, v in arrays.items()})
                          for i in idx])
        p_val = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        a_better = (va - vb > 0) == (m in HIGHER_IS_BETTER)
        out.append(dict(metric=m, A=va, B=vb, diff=va - vb, ci_low=lo, ci_high=hi, p=p_val, n=len(ia),
                        winner=("A" if a_better else "B") if p_val < 0.05 else "n.s."))
    return out


def main():
    rng = np.random.default_rng(42)
    rows = []
    for question, set_name, a, b, metrics in COMPARISONS:
        try:
            res = compare(set_name, a, b, metrics, rng)
        except FileNotFoundError as e:
            print(f"skip: {question} ({Path(e.filename).parent.name} has no {Path(e.filename).name})")
            continue
        for r in res:
            rows.append(dict(question=question, set=set_name, side_A=a, side_B=b, **r))
            print(f"{question[:52]:52s} {r['metric']:22s} A={r['A']:.4f} B={r['B']:.4f} "
                  f"diff={r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={r['p']:.4f} "
                  f"n={r['n']} -> {r['winner']}")
    pd.DataFrame(rows).to_csv("results/significance.csv", index=False)
    print("\n-> results/significance.csv")


if __name__ == "__main__":
    main()
