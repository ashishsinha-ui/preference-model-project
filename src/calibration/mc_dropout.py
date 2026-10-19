"""MC Dropout (Gal & Ghahramani, 2016): keep dropout active at inference, run N passes; the mean is
the prediction and the std is an uncertainty estimate.

  python -m src.calibration.mc_dropout t2_bert_hard_hh_seed42 --passes 10

Saves results/<tier>_<model>-mc_<label>_<dataset>_seed42/ in the same layout as a normal run
(so calibrate.py works on it) plus uncertainty.json:
  auroc_error            does high std flag the model's own mistakes?
  auroc_hd_std / _conf   does high std (or low confidence |p - 0.5|) flag pairs where humans
                         disagree (SHP 0.4 < p_A < 0.6)? -> the human-in-the-loop experiment
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import CrossCollator, PairDataset, load_split
from src.evaluation.evaluate import EVAL_SETS, evaluate_set, swap
from src.models.cross_encoder import load_checkpoint
from src.training.trainer import to_device
from src.utils import tracking
from src.utils.tokenization import MAX_RESP


def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if "Dropout" in type(m).__name__:  # nn.Dropout and DeBERTa's StableDropout
            m.train()


@torch.no_grad()
def mc_predict(model, loader, passes, bf16=True):
    enable_dropout(model)
    runs = []
    for _ in range(passes):
        out = []
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                out.append(torch.sigmoid(model(to_device(batch, "cuda")).float()).cpu())
        runs.append(torch.cat(out).numpy())
    runs = np.stack(runs)
    return runs.mean(0), runs.std(0)


def mc_run_name(run):
    parts = run.split("_")
    parts[1] += "-mc"
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--sets", nargs="+", default=["hh_val", "hh_test", "shp_val", "shp_test"])
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    model, tok, cfg = load_checkpoint(f"checkpoints/{args.run}")
    collate = CrossCollator(tok, token_type=model.use_token_type)
    name = mc_run_name(args.run)
    out_dir = Path("results") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tracking.start_run(name, {"base_run": args.run, "passes": args.passes}, enabled=not args.no_track)

    metrics, unc = {}, {}
    for set_name in args.sets:
        df = load_split(*EVAL_SETS[set_name])
        preds = {}
        for tag, d in [("fwd", df), ("swp", swap(df))]:
            ds = PairDataset(d, tok, max_prompt=cfg["max_prompt"], max_resp=MAX_RESP)
            loader = DataLoader(ds, batch_size=cfg["eval_batch_size"], collate_fn=collate, num_workers=0)
            preds[tag] = mc_predict(model, loader, args.passes, cfg["bf16"])
        (p, std), (p_swp, _) = preds["fwd"], preds["swp"]
        pd.DataFrame({"uid": df.uid, "p_a": p, "p_a_swapped": p_swp, "std": std}).to_parquet(
            out_dir / f"preds_{set_name}.parquet", index=False)

        res = evaluate_set(df, p, p_swp)
        y = df.label.to_numpy()
        u = {"mean_std": float(std.mean()),
             "auroc_error": float(roc_auc_score((p > 0.5) != y, std))}
        if "high_disagreement" in df:
            hd = df.high_disagreement.to_numpy()
            u["auroc_hd_std"] = float(roc_auc_score(hd, std))
            u["auroc_hd_conf"] = float(roc_auc_score(hd, -np.abs(p - 0.5)))
            u["frac_hd_in_flagged_0.4_0.6"] = float(hd[(p > 0.4) & (p < 0.6)].mean())
            u["frac_hd_overall"] = float(hd.mean())
        metrics[set_name], unc[set_name] = res, u
        tracking.log({f"{set_name}/{k}": v for k, v in {**res, **u}.items()
                      if isinstance(v, (int, float)) and np.isfinite(v)})
        print(f"{set_name:10s} acc={res['accuracy']:.4f} ece={res['ece']:.4f} nll={res['nll']:.4f} "
              + " ".join(f"{k}={v:.4f}" for k, v in u.items()))

    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    json.dump(unc, open(out_dir / "uncertainty.json", "w"), indent=2)
    tracking.finish()


if __name__ == "__main__":
    main()
