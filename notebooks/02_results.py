# %% [markdown]
# # 02 — Results: master tables and paper figures
# Reads results/*/metrics.json, calibration.json, length_perturbation.json, uncertainty.json and
# writes results/tables/*.{csv,md} and results/figures/*.{png,pdf}.
# Run as a script (`python notebooks/02_results.py`) or cell by cell in VS Code.
# Colours: validated reference palette (slots 1-3 + one-hue blue ramp), light surface, recessive chrome.

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

from src.calibration.calibrate import load_preds
from src.calibration.temperature_scaling import TemperatureScaler
from src.evaluation.metrics import reliability_bins

FIG, TAB = Path("results/figures"), Path("results/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",              # categorical slots 1-3
     "b250": "#86b6ef", "b550": "#1c5cab",                            # one-hue ramp (dumbbells)
     "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7"}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "font.family": "sans-serif", "font.size": 10, "text.color": C["ink"],
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": C["muted"], "ytick.color": C["muted"], "xtick.labelcolor": C["ink2"],
    "ytick.labelcolor": C["ink2"], "legend.frameon": False, "legend.labelcolor": C["ink2"],
    "lines.linewidth": 2, "lines.markersize": 8})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def ref_line(ax, x, text, n):
    """Dashed reference line with its label tucked under the last row (never over data)."""
    ax.axvline(x, color=C["muted"], linestyle="--", linewidth=1)
    ax.set_ylim(-0.9, n - 0.5)
    ax.text(x, -0.75, f" {text}", color=C["muted"], fontsize=8, va="center")


def legend_below(ax, ncol=2):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol, fontsize=8)


def load(run, kind="metrics"):
    try:
        return json.load(open(f"results/{run}/{kind}.json"))
    except FileNotFoundError:
        return None


# Models shown in the HH tables/figures, in tier order: (label, run)
HH_MODELS = [
    ("T0 length only", "t0_length_hard_hh_seed42"),
    ("T0 TF-IDF + LR", "t0_tfidf-diff_hard_hh_seed42"),
    ("T1 BiLSTM cross", "t1_bilstm-cross_hard_hh_seed42"),
    ("T1 BiLSTM siamese", "t1_bilstm-siamese_hard_hh_seed42"),
    ("T2 BERT cross", "t2_bert_hard_hh_seed42"),
    ("T2 BERT cross, 3-seed ens.", "t2_bert-ens3_hard_hh_seed42"),
    ("T5 BERT bi-encoder", "t5_bert-bi_hard_hh_seed42"),
    ("T3 DeBERTa cross", "t3_deberta_hard_hh_seed42"),
    ("T3 DeBERTa cross, 3-seed ens.", "t3_deberta-ens3_hard_hh_seed42"),
    ("T6 LLaMA-3-8B judge (3-shot)", "t6_llama3-8b-3shot_judge_hh_seed42"),
]
SHP_PAIRS = [("BiLSTM", "t1_bilstm-cross"), ("BERT", "t2_bert"), ("DeBERTa", "t3_deberta")]

# %% [markdown]
# ## Table 1 — HH (helpful) test: accuracy, calibration, position and length bias, domain shift

# %%
rows = []
for label, run in HH_MODELS:
    m, cal = load(run), load(run, "calibration")
    if m is None or "hh_test" not in m:
        continue
    h = m["hh_test"]
    sym = cal["sets"]["hh_test"]["sym"]["accuracy"] if cal and "hh_test" in cal["sets"] else np.nan
    rows.append({"model": label, "run": run, "n": h["n"], "acc": h["accuracy"],
                 "acc_ci_low": h["accuracy_ci95"][0], "acc_ci_high": h["accuracy_ci95"][1],
                 "acc_symmetrized": sym, "auc": h["auc"], "ece": h["ece"], "brier": h["brier"],
                 "swap_consistency": h["swap_consistency"], "position_bias": h["position_bias"],
                 "length_r_model": h["length_r_model"], "length_r_human": h["length_r_human"],
                 "harmless_acc": m.get("hh_harmless", {}).get("accuracy", np.nan),
                 "harmless_n": m.get("hh_harmless", {}).get("n", np.nan)})
t1 = pd.DataFrame(rows)
t1.to_csv(TAB / "table1_hh.csv", index=False)
fmt = t1.assign(acc=lambda d: d.apply(lambda r: f"{r.acc:.3f} [{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}]", axis=1))
(TAB / "table1_hh.md").write_text(fmt[["model", "n", "acc", "acc_symmetrized", "auc", "ece", "swap_consistency",
                                       "position_bias", "length_r_model", "harmless_acc"]].round(3).to_markdown(index=False))
print(fmt[["model", "n", "acc", "acc_symmetrized", "ece", "swap_consistency", "harmless_acc"]].round(3).to_string(index=False))

# %% [markdown]
# ## Table 2 — SHP test: hard vs soft labels (RQ3)

# %%
rows = []
for tier, prefix in SHP_PAIRS:
    for lt in ("hard", "soft"):
        m = load(f"{prefix}_{lt}_shp_seed42")
        if m is None:
            continue
        s = m["shp_test"]
        rows.append({"model": tier, "labels": lt, "acc": s["accuracy"], "ece": s["ece"], "nll": s["nll"],
                     "kl_to_human": s["kl_to_human"], "ece_high_disagreement": s["ece_high_disagreement"],
                     "kl_high_disagreement": s["kl_high_disagreement"], "swap_consistency": s["swap_consistency"]})
    cal = load(f"{prefix}_hard_shp_seed42", "calibration")
    if cal:
        s = cal["sets"]["shp_test"]["temp"]
        rows.append({"model": tier, "labels": "hard + temp. scaling", "acc": s["accuracy"], "ece": s["ece"],
                     "nll": s["nll"], "kl_to_human": s["kl_to_human"],
                     "ece_high_disagreement": s["ece_high_disagreement"], "kl_high_disagreement": np.nan,
                     "swap_consistency": s["swap_consistency"]})
t2 = pd.DataFrame(rows)
t2.to_csv(TAB / "table2_shp_soft_vs_hard.csv", index=False)
(TAB / "table2_shp_soft_vs_hard.md").write_text(t2.round(3).to_markdown(index=False))
print(t2.round(3).to_string(index=False))

# %% [markdown]
# ## Figure 1 — Accuracy ladder (HH test, 95% bootstrap CI) and the effect of symmetrization

# %%
d = t1.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.2))
y = np.arange(len(d))
ax.hlines(y, d.acc_ci_low, d.acc_ci_high, color=C["b250"], linewidth=2, zorder=2)
ax.plot(d.acc, y, "o", color=C["s1"], label="raw prediction", zorder=3)
has_sym = d.acc_symmetrized.notna()
ax.plot(d.acc_symmetrized[has_sym], y[has_sym], "D", color=C["s2"], markersize=7,
        markeredgecolor=C["surface"], markeredgewidth=1.5, label="symmetrized (avg. of both A/B orders)", zorder=4)
ref_line(ax, 0.5, "chance", len(d))
ax.set_yticks(y, [f"{m}  (n={int(n):,})" if n != 6194 else m for m, n in zip(d.model, d.n)])
ax.set_xlabel("HH helpful test accuracy (bars: 95% bootstrap CI)")
ax.set_xlim(0.48, 0.72)
ax.grid(axis="y", visible=False)
legend_below(ax)
ax.set_title("Accuracy rises with model tier; averaging both orders adds ~1 point")
save(fig, "fig1_accuracy_ladder")

# %% [markdown]
# ## Figure 2 — Position bias: swap consistency (share of pairs whose decision flips when A/B swap)

# %%
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.2))
ax.barh(y, d.swap_consistency, height=0.55, color=C["s1"], zorder=2)
for yi, v in zip(y, d.swap_consistency):
    ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=8, color=C["ink2"])
ref_line(ax, 0.95, "plan target 0.95", len(d))
ax.set_yticks(y, d.model)
ax.set_xlim(0, 1.08)
ax.set_xlabel("swap consistency on HH test (1.0 = never depends on order)")
ax.grid(axis="y", visible=False)
ax.set_title("Cross-encoders and the LLM judge depend on response order")
save(fig, "fig2_swap_consistency")

# %% [markdown]
# ## Figure 3 — Reliability diagrams on SHP test: hard vs hard + temperature scaling vs soft labels

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, (tier, prefix) in zip(axes, SHP_PAIRS[1:]):
    hard_dir, soft_dir = Path(f"results/{prefix}_hard_shp_seed42"), Path(f"results/{prefix}_soft_shp_seed42")
    hard, soft = load_preds(hard_dir, "shp_test"), load_preds(soft_dir, "shp_test")
    val = load_preds(hard_dir, "shp_val")
    temp = TemperatureScaler().fit(val.p_a_model, val.label)
    series = [("hard labels", hard.p_a_model, hard.label, C["s1"]),
              (f"hard + temp. scaling (T={temp.T:.2f})", temp.transform(hard.p_a_model), hard.label, C["s3"]),
              ("soft labels", soft.p_a_model, soft.label, C["s2"])]
    ax.plot([0.5, 1], [0.5, 1], color=C["muted"], linestyle="--", linewidth=1)
    for name, p, yv, col in series:
        b = reliability_bins(np.asarray(p), np.asarray(yv))
        conf, acc, cnt = map(np.array, zip(*[r for r in b if r[2] >= 30]))  # bins with >= 30 pairs
        ece = sum(c / sum(x[2] for x in b) * abs(cf - a) for cf, a, c in b)
        ax.plot(conf, acc, "-o", color=col, markersize=6, label=f"{name}  ECE {ece:.3f}")
    ax.set_title(f"{tier} (SHP test)")
    ax.set_xlabel("model confidence in its predicted winner")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.4, 1)
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("observed accuracy")
fig.suptitle("Hard labels are overconfident; temperature scaling and soft labels both repair it",
             fontsize=11, fontweight="bold", color=C["ink"])
save(fig, "fig3_reliability_shp")

# %% [markdown]
# ## Figure 4 — RQ3: distance to the human vote distribution (KL), hard → soft labels

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (metric, title) in zip(axes, [("kl_to_human", "KL to human vote share (all pairs)"),
                                      ("kl_high_disagreement", "KL on high-disagreement pairs")]):
    sub = t2[t2.labels.isin(["hard", "soft"])].pivot(index="model", columns="labels", values=metric)
    sub = sub.reindex([t for t, _ in SHP_PAIRS])
    yy = np.arange(len(sub))[::-1]
    ax.hlines(yy, sub.soft, sub.hard, color=C["axis"], linewidth=2, zorder=1)
    ax.plot(sub.hard, yy, "o", color=C["b250"], label="hard labels", zorder=2)
    ax.plot(sub.soft, yy, "o", color=C["b550"], label="soft labels", zorder=3)
    for yi, (h, s) in zip(yy, zip(sub.hard, sub.soft)):
        ax.text(s, yi - 0.2, f"{s:.2f}", ha="center", va="top", fontsize=8, color=C["ink2"])
        ax.text(h, yi - 0.2, f"{h:.2f}", ha="center", va="top", fontsize=8, color=C["ink2"])
    ax.set_yticks(yy, sub.index)
    ax.set_ylim(-0.6, len(sub) - 0.5)
    ax.set_xlim(0, sub.values.max() * 1.12)
    ax.set_xlabel("KL divergence (lower = closer to humans)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8,
           bbox_to_anchor=(0.5, -0.04))
save(fig, "fig4_rq3_kl_hard_vs_soft")

# %% [markdown]
# ## Figure 5 — Length bias (Test B): preference for the longer of two same-content responses

# %%
rows = []
for label, run in [("BERT cross", "t2_bert_hard_hh_seed42"), ("BERT cross + swap aug.", "t2_bert-swapaug_hard_hh_seed42"),
                   ("BERT no prompt", "t2_bert-noprompt_hard_hh_seed42"), ("BERT bi-encoder", "t5_bert-bi_hard_hh_seed42"),
                   ("DeBERTa cross", "t3_deberta_hard_hh_seed42"), ("DeBERTa + swap aug.", "t3_deberta-swapaug_hard_hh_seed42")]:
    r = load(run, "length_perturbation")
    if r:
        rows.append({"model": label, "p_prefers_longer_same_content": r["equal_compress"]["p_prefers_longer"],
                     "p_prefers_padded": r["equal_pad"]["p_prefers_longer"], "acc_orig": r["orig"]["accuracy"],
                     "acc_better_answer_compressed": r["compress_chosen"]["accuracy"], "compress_drop": r["compress_drop"]})
t5 = pd.DataFrame(rows)
t5.to_csv(TAB / "table5_test_b_length.csv", index=False)
d5 = t5.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
yy = np.arange(len(d5))
ax.barh(yy, d5.p_prefers_longer_same_content, height=0.55, color=C["s1"], zorder=2)
for yi, v in zip(yy, d5.p_prefers_longer_same_content):
    ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=C["ink2"])
ref_line(ax, 0.5, "no length preference", len(d5))
ax.set_yticks(yy, d5.model)
ax.set_xlim(0, 1)
ax.set_xlabel("share of same-content pairs where the model picks the longer version")
ax.grid(axis="y", visible=False)
ax.set_title("Every architecture prefers the longer version of the same answer")
save(fig, "fig5_length_bias_test_b")

# %% [markdown]
# ## Figure 6 — Domain shift: helpful-trained models on helpful vs harmless prompts

# %%
d6 = t1.dropna(subset=["harmless_acc"]).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d6) + 1.2))
yy = np.arange(len(d6))
ax.hlines(yy, d6.harmless_acc, d6.acc, color=C["axis"], linewidth=2, zorder=1)
ax.plot(d6.acc, yy, "o", color=C["b550"], label="helpful test (in domain)", zorder=3)
ax.plot(d6.harmless_acc, yy, "o", color=C["b250"], label="harmless (out of domain)", zorder=2)
ref_line(ax, 0.5, "chance", len(d6))
ax.set_yticks(yy, d6.model)
ax.set_xlim(0.3, 0.72)
ax.set_xlabel("accuracy")
ax.grid(axis="y", visible=False)
legend_below(ax)
ax.set_title("Helpful-trained models fall below chance on harmless prompts")
save(fig, "fig6_domain_shift")

# %% [markdown]
# ## Table 6 — Does model uncertainty flag human disagreement? (SHP test, AUROC; 0.5 = no signal)

# %%
rows = []
for label, run in [("BERT hard, MC dropout", "t2_bert-mc_hard_hh_seed42"), ("BERT hard, 3-seed ensemble", "t2_bert-ens3_hard_hh_seed42"),
                   ("DeBERTa hard, MC dropout", "t3_deberta-mc_hard_hh_seed42"), ("DeBERTa hard, 3-seed ensemble", "t3_deberta-ens3_hard_hh_seed42"),
                   ("BERT soft (SHP), MC dropout", "t2_bert-mc_soft_shp_seed42"), ("DeBERTa soft (SHP), MC dropout", "t3_deberta-mc_soft_shp_seed42")]:
    u = load(run, "uncertainty")
    if u:
        s = u["shp_test"]
        rows.append({"model / method": label, "AUROC errors (spread)": s["auroc_error"],
                     "AUROC disagreement (spread)": s["auroc_hd_std"], "AUROC disagreement (|p-0.5|)": s["auroc_hd_conf"],
                     "disagreement rate in 0.4-0.6 band": s["frac_hd_in_flagged_0.4_0.6"], "base rate": s["frac_hd_overall"]})
t6 = pd.DataFrame(rows)
t6.to_csv(TAB / "table6_uncertainty.csv", index=False)
(TAB / "table6_uncertainty.md").write_text(t6.round(3).to_markdown(index=False))
print(t6.round(3).to_string(index=False))

# %% [markdown]
# ## Table 7 — Significance tests (paired bootstrap), if src.evaluation.significance has been run

# %%
if Path("results/significance.csv").exists():
    sig = pd.read_csv("results/significance.csv")
    (TAB / "table7_significance.md").write_text(
        sig[["question", "metric", "A", "B", "diff", "ci_low", "ci_high", "p", "n", "winner"]].round(4).to_markdown(index=False))
    print(sig[["question", "metric", "diff", "p", "winner"]].round(4).to_string(index=False))
print(f"\nTables -> {TAB}/   Figures -> {FIG}/")
