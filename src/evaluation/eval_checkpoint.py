"""Evaluate already-trained checkpoints on extra eval sets (results are merged into metrics.json).

  python -m src.evaluation.eval_checkpoint t2_bert_hard_hh_seed42 t3_deberta_hard_hh_seed42 --sets hhh_test uf_test
"""
import argparse

import torch
from transformers.utils import logging as hf_logging

from src.evaluation.evaluate import EVAL_SETS, evaluate_model
from src.models.cross_encoder import load_checkpoint
from src.training.trainer import make_predict_fn
from src.utils.tokenization import MAX_RESP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sets", nargs="+", required=True, choices=list(EVAL_SETS))
    args = ap.parse_args()
    hf_logging.set_verbosity_error()
    for run in args.runs:
        print(f"\n### {run}")
        model, tok, cfg = load_checkpoint(f"checkpoints/{run}")
        predict = make_predict_fn(model, tok, model.make_collator(tok), batch_size=cfg["eval_batch_size"],
                                  bf16=cfg["bf16"], max_prompt=cfg["max_prompt"], max_resp=MAX_RESP)
        evaluate_model(predict, run, sets=args.sets)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
