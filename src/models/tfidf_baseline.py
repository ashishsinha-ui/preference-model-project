"""T0 baseline: TF-IDF + Logistic Regression.

features=diff   : x = tfidf(A) - tfidf(B), no intercept -> swapping A/B exactly negates the logit
                  (position-invariant by construction)
features=concat : x = [tfidf(A), tfidf(B)] + intercept -> free to learn a position preference
Soft labels: each pair is used twice, as (x, 1) with weight p_a and (x, 0) with weight 1 - p_a,
which is exactly logistic regression with a soft cross-entropy target.

  python -m src.models.tfidf_baseline --dataset hh --label_type hard --features diff
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.dataset import load_split
from src.evaluation.evaluate import evaluate_model
from src.evaluation.metrics import classification_metrics
from src.utils import tracking
from src.utils.seed import set_seed


class TfidfPreference:
    def __init__(self, features="diff", max_features=10000):
        self.features = features
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3,
                                   sublinear_tf=True)
        self.clf = None

    def _x(self, df):
        a, b = self.vec.transform(df.response_a), self.vec.transform(df.response_b)
        return (a - b).tocsr() if self.features == "diff" else sp.hstack([a, b]).tocsr()

    def fit(self, df, label_type="hard", C=1.0):
        if not hasattr(self.vec, "vocabulary_"):
            self.vec.fit(pd.concat([df.response_a, df.response_b]))
        x, y, w = self._x(df), df.label.to_numpy(), None
        if label_type == "soft":
            n = x.shape[0]
            x = sp.vstack([x, x]).tocsr()
            y = np.r_[np.ones(n), np.zeros(n)]
            w = np.r_[df.p_a.to_numpy(), 1 - df.p_a.to_numpy()]
        self.clf = LogisticRegression(C=C, max_iter=2000, fit_intercept=self.features == "concat")
        self.clf.fit(x, y, sample_weight=w)
        return self

    def predict(self, df):
        return self.clf.predict_proba(self._x(df))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hh", "shp"], default="hh")
    ap.add_argument("--label_type", choices=["hard", "soft"], default="hard")
    ap.add_argument("--features", choices=["diff", "concat"], default="diff")
    ap.add_argument("--max_features", type=int, default=10000)
    ap.add_argument("--subset_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    run_name = f"t0_tfidf-{args.features}_{args.label_type}_{args.dataset}_seed{args.seed}"
    tracking.start_run(run_name, vars(args), enabled=not args.no_track)
    train = load_split(args.dataset, "train", args.subset_size, args.seed)
    val = load_split(args.dataset, "val")

    model, best = TfidfPreference(args.features, args.max_features), None
    t0 = time.time()
    for C in [0.1, 1.0, 10.0]:  # pick C by validation NLL
        model.fit(train, args.label_type, C)
        m = classification_metrics(model.predict(val), val.label)
        print(f"C={C:<5} val_acc={m['accuracy']:.4f} val_nll={m['nll']:.4f}")
        tracking.log({f"tune/val_nll_C{C}": m["nll"], f"tune/val_acc_C{C}": m["accuracy"]})
        if best is None or m["nll"] < best[1]:
            best = (C, m["nll"])
    model.fit(train, args.label_type, best[0])
    tracking.log({"best_C": best[0], "train_time_s": time.time() - t0})
    print(f"best C={best[0]}  ({time.time() - t0:.0f}s)\n")

    evaluate_model(model.predict, run_name)
    tracking.finish()


if __name__ == "__main__":
    main()
