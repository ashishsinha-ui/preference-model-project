# %% [markdown]
# # 01 — Exploratory Data Analysis
# Label balance, domains, length bias in the human labels, SHP agreement (IAA proxy), token budget.
# Run as a script (`python notebooks/01_eda.py`) or cell-by-cell in VS Code ("Run Cell").

# %%
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data.dataset import load_split
from src.utils import tracking

OUT = Path("results/eda")
OUT.mkdir(parents=True, exist_ok=True)
summary = {}

sets = {
    "hh_train": load_split("hh", "train"),
    "hh_test": load_split("hh", "test"),
    "hh_harmless": load_split("hh", "harmless_shift"),
    "shp_train": load_split("shp", "train"),
    "shp_test": load_split("shp", "test"),
}
for name, df in sets.items():
    print(f"{name:12s} n={len(df):7d}  frac_A={df.label.mean():.3f}")

# %% [markdown]
# ## 1. Domains

# %%
print(sets["hh_train"].subset.value_counts(), "\n")
dom = pd.DataFrame({"train_gated": sets["shp_train"].domain.value_counts(),
                    "test": sets["shp_test"].domain.value_counts()}).fillna(0).astype(int)
print(dom)
summary["hh_subsets_train"] = sets["hh_train"].subset.value_counts().to_dict()
summary["shp_domains"] = dom.to_dict()

# %% [markdown]
# ## 2. Length bias in the HUMAN labels
# If humans already prefer longer answers, a model's length correlation is partly legitimate.
# "longer_wins_accuracy" is the accuracy of the trivial rule "pick the longer response".

# %%
def length_stats(df):
    a_wins = (df.label == 1).to_numpy()
    chosen = np.where(a_wins, df.len_a, df.len_b)
    rejected = np.where(a_wins, df.len_b, df.len_a)
    differ = chosen != rejected
    longer_wins = np.where(df.len_a == df.len_b, 0.5, (df.len_a > df.len_b).to_numpy() == a_wins)
    return {"median_words_chosen": float(np.median(chosen)),
            "median_words_rejected": float(np.median(rejected)),
            "p_chosen_longer": float((chosen > rejected)[differ].mean()),
            "longer_wins_accuracy": float(longer_wins.mean())}


summary["length"] = {name: length_stats(df) for name, df in sets.items()}
print(pd.DataFrame(summary["length"]).T.round(3))

fig, ax = plt.subplots(figsize=(7, 4))
for name in ["hh_train", "hh_harmless", "shp_train"]:
    df = sets[name]
    r = np.log((df.len_a + 1) / (df.len_b + 1))
    bins = pd.qcut(r, 10, duplicates="drop")
    ax.plot(r.groupby(bins, observed=True).mean(), df.label.groupby(bins, observed=True).mean(),
            marker="o", label=name)
ax.axhline(0.5, ls="--", c="gray")
ax.set_xlabel("log length ratio (A / B)")
ax.set_ylabel("P(human prefers A)")
ax.set_title("Human preference vs. relative length")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "length_vs_preference.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name in zip(axes, ["hh_train", "shp_train"]):
    df = sets[name]
    a_wins = (df.label == 1).to_numpy()
    edges = np.logspace(0, 3.5, 50)
    ax.hist(np.where(a_wins, df.len_a, df.len_b), edges, alpha=0.5, label="chosen")
    ax.hist(np.where(a_wins, df.len_b, df.len_a), edges, alpha=0.5, label="rejected")
    ax.set_xscale("log")
    ax.set_xlabel("response length (words)")
    ax.set_title(name)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "length_hist.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. SHP agreement (IAA proxy) and the quality gate
# kappa_proxy = 2|p_A - 0.5|. Test split is ungated; train is after the gate.

# %%
shp_test, shp_train = sets["shp_test"], sets["shp_train"]
summary["shp_iaa"] = {
    "test_frac_high_disagreement": float(shp_test.high_disagreement.mean()),
    "test_mean_kappa_proxy": float(shp_test.kappa_proxy.mean()),
    "test_frac_passes_gate": float(shp_test.passes_gate.mean()),
    "train_mean_kappa_proxy": float(shp_train.kappa_proxy.mean()),
}
print(summary["shp_iaa"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(shp_test.p_a, bins=40)
axes[0].axvspan(0.4, 0.6, color="red", alpha=0.2, label="high disagreement")
axes[0].set_title("SHP test (ungated): p_A")
axes[0].legend()
axes[1].hist(shp_train.p_a, bins=40)
axes[1].set_title("SHP train (after quality gate): p_A")
fig.tight_layout()
fig.savefig(OUT / "shp_pa_hist.png", dpi=150)
plt.show()

per_dom = shp_test.groupby("domain").agg(n=("uid", "size"),
                                         high_disagreement=("high_disagreement", "mean"),
                                         passes_gate=("passes_gate", "mean"),
                                         median_kappa=("kappa_proxy", "median")).round(3)
print(per_dom)
summary["shp_per_domain"] = per_dom.to_dict()
ax = per_dom[["high_disagreement", "passes_gate"]].sort_values("passes_gate").plot.barh(figsize=(7, 6))
ax.set_title("SHP test: disagreement and gate pass rate by domain")
plt.tight_layout()
plt.savefig(OUT / "shp_domains.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Token budget check (DeBERTa-v3 tokenizer, 20k-pair samples)
# Budget: prompt 128 (left-truncated), each response 190 (right-truncated).

# %%
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name in zip(axes, ["hh_train", "shp_train"]):
    df = sets[name].sample(20000, random_state=42)
    n = {c: np.array([len(x) for x in tok(list(df[c]), add_special_tokens=False)["input_ids"]])
         for c in ["prompt", "response_a", "response_b"]}
    resp = np.r_[n["response_a"], n["response_b"]]
    summary[f"tokens_{name}"] = {
        "prompt_p50": float(np.median(n["prompt"])), "prompt_p95": float(np.percentile(n["prompt"], 95)),
        "resp_p50": float(np.median(resp)), "resp_p95": float(np.percentile(resp, 95)),
        "prompt_truncated": float((n["prompt"] > 128).mean()),
        "resp_truncated": float((resp > 190).mean()),
        "both_resp_fit": float(((n["response_a"] <= 190) & (n["response_b"] <= 190)).mean())}
    ax.hist(np.clip(n["prompt"], 0, 1000), bins=50, alpha=0.5, label="prompt")
    ax.hist(np.clip(resp, 0, 1000), bins=50, alpha=0.5, label="response")
    ax.axvline(128, c="C0", ls="--")
    ax.axvline(190, c="C1", ls="--")
    ax.set_xlabel("tokens (clipped at 1000)")
    ax.set_title(name)
    ax.legend()
    print(name, summary[f"tokens_{name}"])
fig.tight_layout()
fig.savefig(OUT / "token_lengths.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Save summary + log to MLflow

# %%
with open(OUT / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

tracking.start_run("eda_v1", {"stage": "eda"})
for group in ["length", "shp_iaa", "tokens_hh_train", "tokens_shp_train"]:
    d = summary[group]
    items = d.items() if group != "length" else ((f"{s}/{k}", v) for s, m in d.items() for k, v in m.items())
    tracking.log({f"{group}/{k}": v for k, v in items})
for path in list(OUT.glob("*.png")) + [OUT / "eda_summary.json"]:
    tracking.log_file(path)
tracking.finish()
print(f"Saved to {OUT}/")
