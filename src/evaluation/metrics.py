"""Preference-model metrics (plan section 5): accuracy, AUC, Brier, NLL, ECE, KL/JS, bias scores.

Convention everywhere: p_a = predicted P(response A is preferred), label 1 = A, 0 = B, -1 = tie.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def _corr(fn, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(fn(x, y).statistic)


# ----------------------------------------------------------------------------- calibration
def reliability_bins(p_a, label, n_bins=15):
    """Top-label reliability: confidence of the predicted winner vs. how often it is right.
    Confidence lives in [0.5, 1], so the 15 bins span that range."""
    p, y = _clip(p_a), np.asarray(label)
    conf = np.maximum(p, 1 - p)
    correct = ((p > 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return rows  # list of (mean confidence, accuracy, count)


def ece(p_a, label, n_bins=15):
    rows = reliability_bins(p_a, label, n_bins)
    n = sum(c for _, _, c in rows)
    return float(sum(c / n * abs(conf - acc) for conf, acc, c in rows)) if n else float("nan")


def binary_kl(p, q):
    """Mean KL(p || q) between Bernoulli distributions. p = human vote share, q = model."""
    p, q = _clip(p), _clip(q)
    return float(np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))


def binary_js(p, q):
    m = (_clip(p) + _clip(q)) / 2
    return 0.5 * binary_kl(p, m) + 0.5 * binary_kl(q, m)


# ----------------------------------------------------------------------------- metric groups
def classification_metrics(p_a, label, n_bins=15):
    p, y = _clip(p_a), np.asarray(label)
    keep = y >= 0
    p, y = p[keep], y[keep]
    return {"accuracy": float(((p > 0.5) == y).mean()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ece": ece(p, y, n_bins)}


def soft_metrics(p_a, p_human):
    p, h = _clip(p_a), np.asarray(p_human, float)
    return {"kl_to_human": binary_kl(h, p), "js_to_human": binary_js(h, p),
            "brier_soft": float(np.mean((p - h) ** 2))}


def swap_metrics(p_fwd, p_swp, label):
    """p_swp = model's P(A) after swapping A and B. A position-invariant model has p_swp = 1 - p_fwd."""
    p_fwd, p_swp, y = np.asarray(p_fwd, float), np.asarray(p_swp, float), np.asarray(label)
    p_sym = (p_fwd + 1 - p_swp) / 2  # symmetrized prediction (simple debiasing)
    keep = y >= 0
    return {"swap_consistency": float(((p_fwd > 0.5) != (p_swp > 0.5)).mean()),
            "position_bias": float(np.mean(p_fwd + p_swp - 1)),  # > 0 favours slot A, < 0 slot B
            "accuracy_symmetrized": float(((p_sym > 0.5) == y)[keep].mean())}


def length_bias(p_a, len_a, len_b, label=None):
    """Correlation between length difference (A - B) and the model's P(A).
    length_bias_excess = model correlation minus the humans' own correlation."""
    diff = np.asarray(len_a, float) - np.asarray(len_b, float)
    out = {"length_r_model": _corr(pearsonr, diff, p_a),
           "length_rho_model": _corr(spearmanr, diff, p_a)}
    if label is not None:
        y = np.asarray(label)
        keep = y >= 0
        out["length_r_human"] = _corr(pearsonr, diff[keep], y[keep])
        out["length_bias_excess"] = out["length_r_model"] - out["length_r_human"]
    return out


def bootstrap_ci(correct, n_boot=1000, seed=42):
    """95% bootstrap confidence interval for a mean (e.g. per-pair correctness)."""
    c = np.asarray(correct, float)
    rng = np.random.default_rng(seed)
    means = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
