"""Generic training loop shared by the BiLSTM and transformer models.

model(batch) -> logits of shape (B,). Targets are batch["labels"] (hard 0/1 or soft p_a).
Evaluates on the validation set every `eval_every` optimizer steps (default: once per epoch),
early-stops on validation loss, and saves only the best weights (fp16) to ckpt_path.
"""
import os
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import PairDataset
from src.evaluation.metrics import classification_metrics
from src.training.losses import SoftBCELoss
from src.utils import tracking

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict_loader(model, loader, device="cuda", bf16=True):
    model.eval()
    out = []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(to_device(batch, device))
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def make_predict_fn(model, tokenizer, collate, batch_size=128, bf16=True, device="cuda", **ds_kwargs):
    """Wraps a trained model as predict_fn(df) -> P(A) for evaluate_model()."""
    def predict(df):
        ds = PairDataset(df, tokenizer, **ds_kwargs)
        return predict_loader(model, DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                                                num_workers=0), device, bf16)
    return predict


def train(model, train_ds, val_ds, collate, ckpt_path, *, epochs=3, batch_size=32, lr=2e-5,
          weight_decay=0.01, warmup_ratio=0.1, grad_accum=1, eval_every=0, patience=2, bf16=True,
          num_workers=0, max_grad_norm=1.0, param_groups=None, device="cuda"):
    model.to(device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, collate_fn=collate, num_workers=num_workers)
    steps_per_epoch = len(train_dl) // grad_accum
    total = epochs * steps_per_epoch
    eval_every = eval_every or steps_per_epoch
    opt = torch.optim.AdamW(param_groups if param_groups is not None else model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    loss_fn = SoftBCELoss()
    target = val_ds.target
    state = {"best": float("inf"), "bad": 0, "weights": None}

    def evaluate(step, epoch):
        """Returns True when early stopping should trigger."""
        p = np.clip(predict_loader(model, val_dl, device, bf16), 1e-6, 1 - 1e-6)
        model.train()
        val_loss = float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
        m = classification_metrics(p, (target > 0.5).astype(int))
        tracking.log({"val/loss": val_loss, "val/accuracy": m["accuracy"], "val/auc": m["auc"],
                      "val/ece": m["ece"], "epoch": epoch}, step=step)
        improved = val_loss < state["best"] - 1e-4
        print(f"step {step:6d}  epoch {epoch:.2f}  val_loss={val_loss:.4f}  val_acc={m['accuracy']:.4f}  "
              f"val_auc={m['auc']:.4f}  val_ece={m['ece']:.4f}  {'*best*' if improved else ''}")
        if improved:
            state.update(best=val_loss, bad=0,
                         weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            state["bad"] += 1
        return state["bad"] >= patience

    step, t0, stop = 0, time.time(), False
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(train_dl, desc=f"epoch {epoch + 1}/{epochs}", leave=False,
                                       disable=not sys.stderr.isatty())):

            batch = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            (loss / grad_accum).backward()
            if (i + 1) % grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                tracking.log({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0]}, step=step)
            if step % eval_every == 0 and evaluate(step, epoch + (i + 1) / len(train_dl)):
                stop = True
                break
        if stop:
            print(f"early stopping at step {step}")
            break
    if step % eval_every != 0:
        evaluate(step, float(epochs))

    if state["weights"] is not None:
        model.load_state_dict(state["weights"])
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
               ckpt_path)
    stats = {"train_time_min": (time.time() - t0) / 60, "best_val_loss": state["best"],
             "optimizer_steps": step, "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    tracking.log(stats)
    print({k: round(v, 4) for k, v in stats.items()}, f"-> {ckpt_path}")
    return model
