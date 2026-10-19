"""T1: BiLSTM preference model (2 layers, hidden 256, attention pooling), trained from scratch.

mode=cross    one sequence [CLS] prompt [SEP] A [SEP] B [SEP] -> logit   (same input as the
              transformer cross-encoders; free to learn a position preference)
mode=siamese  a shared encoder scores (prompt, A) and (prompt, B); logit = s_A - s_B
              (position-invariant by construction)
Uses the bert-base-uncased WordPiece vocabulary and the plan's token budget.

  python -m src.models.bilstm --dataset hh --label_type hard --mode cross
"""
import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.data.dataset import CrossCollator, PairDataset, SiameseCollator, load_split
from src.evaluation.evaluate import evaluate_model
from src.training.trainer import make_predict_fn, train
from src.utils import tracking
from src.utils.seed import set_seed
from src.utils.tokenization import MAX_RESP


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden=256, layers=2, dropout=0.3, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.seg = nn.Embedding(2, emb_dim)  # segment 0 = prompt + A, 1 = B
        self.lstm = nn.LSTM(emb_dim, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids, mask, token_type=None):
        x = self.emb(ids)
        if token_type is not None:
            x = x + self.seg(token_type)
        packed = pack_padded_sequence(self.drop(x), mask.sum(1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=ids.size(1))
        scores = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return self.drop((w * h).sum(1))


class BiLSTMPreference(nn.Module):
    def __init__(self, vocab_size, mode="cross", pad_id=0, emb_dim=256, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.enc = BiLSTMEncoder(vocab_size, emb_dim, hidden, layers, dropout, pad_id)
        d = 2 * hidden
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, batch):
        if self.mode == "cross":
            h = self.enc(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
            return self.head(h).squeeze(-1)
        s_a = self.head(self.enc(batch["input_ids_a"], batch["attention_mask_a"])).squeeze(-1)
        s_b = self.head(self.enc(batch["input_ids_b"], batch["attention_mask_b"])).squeeze(-1)
        return s_a - s_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hh", "shp"], default="hh")
    ap.add_argument("--label_type", choices=["hard", "soft"], default="hard")
    ap.add_argument("--mode", choices=["cross", "siamese"], default="cross")
    ap.add_argument("--swap_prob", type=float, default=0.0, help="0.5 = A/B swap augmentation")
    ap.add_argument("--max_prompt", type=int, default=128, help="0 = response-only ablation")
    ap.add_argument("--subset_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()
    set_seed(args.seed)
    tag = (args.mode + ("-swapaug" if args.swap_prob > 0 else "")
           + ("-noprompt" if args.max_prompt == 0 else "") + (f"-dev{args.subset_size}" if args.subset_size else ""))
    run_name = f"t1_bilstm-{tag}_{args.label_type}_{args.dataset}_seed{args.seed}"
    tracking.start_run(run_name, vars(args), enabled=not args.no_track)

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds_kw = dict(max_prompt=args.max_prompt, max_resp=MAX_RESP)
    train_ds = PairDataset(load_split(args.dataset, "train", args.subset_size, args.seed), tok,
                           args.label_type, swap_prob=args.swap_prob, **ds_kw)
    val_ds = PairDataset(load_split(args.dataset, "val"), tok, args.label_type, **ds_kw)
    collate = CrossCollator(tok) if args.mode == "cross" else SiameseCollator(tok)

    model = BiLSTMPreference(tok.vocab_size, args.mode, tok.pad_token_id, args.emb_dim, args.hidden,
                             args.layers, args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{run_name}: {n_params / 1e6:.1f}M params | train={len(train_ds)} val={len(val_ds)}")
    tracking.log({"n_params_m": n_params / 1e6})

    model = train(model, train_ds, val_ds, collate, f"checkpoints/{run_name}/model.pt",
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, warmup_ratio=0.05,
                  patience=2, bf16=False)  # LSTMs run in fp32
    evaluate_model(make_predict_fn(model, tok, collate, bf16=False, **ds_kw), run_name)
    tracking.finish()


if __name__ == "__main__":
    main()
