"""T6: LLM-judge baseline for RQ1 (not fine-tuned).

The instruct LLM sees the conversation and both responses and is asked "Which response is better?".
P(A) = softmax over its next-token logits for "A" vs "B" (no sampling), so the judge gets exactly
the same metrics as every trained tier: accuracy, AUC, ECE, swap test (both orders are scored),
length bias, calibration. Few-shot examples come from the HH *train* split.
Each eval set is subsampled to --max_n pairs (fixed seed) to keep the run to about an hour.

  python -m src.models.llm_judge --model meta-llama/Meta-Llama-3-8B-Instruct --short_name llama3-8b
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.data.dataset import load_split
from src.evaluation.evaluate import EVAL_SETS, evaluate_model
from src.utils import tracking
from src.utils.seed import SEED

SYSTEM = ("You are an expert evaluator of AI assistant responses. You will see a conversation between a "
          "human and an AI assistant, followed by two candidate final assistant responses, A and B. "
          "Decide which response is better.")


def clip_words(text, n, keep_end=False):
    """Word-level stand-in for the encoders' token budget."""
    w = text.split()
    if len(w) <= n:
        return text
    return "... " + " ".join(w[-n:]) if keep_end else " ".join(w[:n]) + " ..."


def render(prompt, a, b):
    return (f"[Conversation]\n{clip_words(prompt, 300, keep_end=True)}\n\n"
            f"[Response A]\n{clip_words(a, 250)}\n\n[Response B]\n{clip_words(b, 250)}\n\n"
            "Which response is better? Answer with a single letter: A or B.")


class LLMJudge:
    def __init__(self, model_name, shots=3, batch_size=8):
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                          device_map="cuda").eval()
        self.batch_size = batch_size
        self.letter_ids = {c: sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in (c, " " + c)})
                           for c in "AB"}
        self.examples = self._few_shot(shots)

    def _few_shot(self, shots):
        """Short HH-train pairs, answers alternating A, B, A, ... so the examples carry no position cue."""
        if shots == 0:
            return []
        tr = load_split("hh", "train")
        short = tr[(tr.len_a <= 60) & (tr.len_b <= 60) & (tr.prompt.str.split().str.len() <= 80)]
        msgs = []
        for k, r in enumerate(short.sample(shots, random_state=SEED).itertuples()):
            chosen, rejected = (r.response_a, r.response_b) if r.label == 1 else (r.response_b, r.response_a)
            a, b, ans = (chosen, rejected, "A") if k % 2 == 0 else (rejected, chosen, "B")
            msgs += [{"role": "user", "content": render(r.prompt, a, b)}, {"role": "assistant", "content": ans}]
        return msgs

    @torch.no_grad()
    def predict(self, df):
        out = []
        for i in tqdm(range(0, len(df), self.batch_size), desc="judging", leave=False):
            chats = [[{"role": "system", "content": SYSTEM}, *self.examples,
                      {"role": "user", "content": render(r.prompt, r.response_a, r.response_b)}]
                     for r in df.iloc[i:i + self.batch_size].itertuples()]
            enc = self.tok.apply_chat_template(chats, add_generation_prompt=True, return_tensors="pt",
                                               padding=True, return_dict=True).to("cuda")
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
            la = torch.logsumexp(logits[:, self.letter_ids["A"]], dim=-1)
            lb = torch.logsumexp(logits[:, self.letter_ids["B"]], dim=-1)
            out.append(torch.sigmoid(la - lb).cpu())
        return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--short_name", default="llama3-8b")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--max_n", type=int, default=1000, help="pairs per eval set")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=list(EVAL_SETS))
    ap.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()
    # "judge" fills the label slot; the dataset slot "hh" makes calibrate.py fit on hh_val
    run_name = f"t6_{args.short_name}-{args.shots}shot_judge_hh_seed{SEED}"
    tracking.start_run(run_name, vars(args), enabled=not args.no_track)
    judge = LLMJudge(args.model, args.shots, args.batch_size)
    print(f"{run_name}: {len(judge.examples) // 2} few-shot examples, letter ids {judge.letter_ids}")
    evaluate_model(judge.predict, run_name, sets=args.sets, max_n=args.max_n)
    tracking.finish()


if __name__ == "__main__":
    main()
