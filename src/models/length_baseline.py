"""T0-len: length-only reference model.  P(A) = sigmoid(w * log((len_a + 1) / (len_b + 1))).

No intercept, so it is position-invariant. Every other model should be judged by how far it
gets beyond this.
  python -m src.models.length_baseline --dataset hh
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.dataset import load_split
from src.evaluation.evaluate import evaluate_model
from src.utils import tracking


def features(df):
    return np.log((df.len_a.to_numpy() + 1) / (df.len_b.to_numpy() + 1))[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hh", "shp"], default="hh")
    ap.add_argument("--label_type", choices=["hard", "soft"], default="hard")
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    run_name = f"t0_length_{args.label_type}_{args.dataset}_seed42"
    tracking.start_run(run_name, vars(args), enabled=not args.no_track)
    train = load_split(args.dataset, "train")
    x, y, w = features(train), train.label.to_numpy(), None
    if args.label_type == "soft":
        n = len(train)
        x, y, w = np.r_[x, x], np.r_[np.ones(n), np.zeros(n)], np.r_[train.p_a, 1 - train.p_a]
    clf = LogisticRegression(fit_intercept=False).fit(x, y, sample_weight=w)
    print(f"{run_name}: weight on log length ratio = {clf.coef_[0, 0]:+.3f}\n")
    tracking.log({"length_weight": clf.coef_[0, 0]})
    evaluate_model(lambda df: clf.predict_proba(features(df))[:, 1], run_name)
    tracking.finish()


if __name__ == "__main__":
    main()
