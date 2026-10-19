"""Test B — length perturbation (plan 5.2). Built from 500 HH-test pairs (chosen 40-150 words).

Variants per base pair (c = chosen, r = rejected, c' = LLM-compressed c with ~40% fewer words,
f = content-free filler sentences); A/B order is randomized per row:
  orig             (c,  r)      label = chosen         reference accuracy on the same 500 pairs
  compress_chosen  (c', r)      label = chosen         does the model still pick the (now shorter) better answer?
  pad_rejected     (c,  r + f)  label = chosen         does padding the worse answer fool the model?
  equal_compress   (c,  c')     no label               same content: P(model prefers the longer one), ideal ~0.5
  equal_pad        (c,  c + f)  no label               filler only:  P(model prefers the padded one), ideal <= 0.5

  python -m src.evaluation.bias_tests generate --n 500                 # GPU, uses an instruct LLM
  python -m src.evaluation.bias_tests evaluate t2_bert_hard_hh_seed42  # GPU, any cross-encoder run
The compressor is Qwen2.5-7B-Instruct by default, deliberately not the LLaMA-3 judge (T6), so the
judge is never scoring text it wrote itself.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import load_split
from src.evaluation.evaluate import swap
from src.utils import tracking
from src.utils.seed import SEED
from src.utils.tokenization import MAX_RESP

OUT = Path("data/bias_test")
LP_PATH = OUT / "length_perturbed.parquet"
FILLERS = ["I hope this helps!", "Let me know if you have any other questions.",
           "Thanks for asking, this is a great question.", "Please feel free to reach out if you need anything else.",
           "I'm always happy to help with questions like this.", "Feel free to ask if anything is unclear."]
COMPRESS_PROMPT = ("Rewrite the following response so that it is about 40% shorter. Keep every piece of "
                   "information, every instruction and the same tone. Do not add anything new. "
                   "Output only the rewritten response.\n\nResponse:\n{text}")


# ----------------------------------------------------------------------------- generation
def compress(texts, model_name, batch_size=16, max_new_tokens=300):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="compressing"):
        msgs = [[{"role": "user", "content": COMPRESS_PROMPT.format(text=t)}] for t in texts[i:i + batch_size]]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      padding=True, return_dict=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


def generate(n, gen_model):
    rng = np.random.default_rng(SEED)
    test = load_split("hh", "test")
    test["chosen"] = np.where(test.label == 1, test.response_a, test.response_b)
    test["rejected"] = np.where(test.label == 1, test.response_b, test.response_a)
    words = test.chosen.str.split().str.len()
    cand = test[(words >= 40) & (words <= 150)].sample(int(n * 1.5), random_state=SEED).reset_index(drop=True)

    cand["compressed"] = compress(list(cand.chosen), gen_model)
    cand["ratio"] = cand.compressed.str.split().str.len() / cand.chosen.str.split().str.len()
    ok = cand[(cand.ratio >= 0.35) & (cand.ratio <= 0.8) & (cand.compressed.str.len() > 0)].head(n)
    print(f"kept {len(ok)}/{len(cand)} compressions with 35-80% of the original length "
          f"(median ratio {ok.ratio.median():.2f})")

    rows = []
    for _, r in ok.iterrows():
        filler = " ".join(rng.choice(FILLERS, size=2, replace=False))
        variants = {  # (first, second, label: 1 = first is better, -1 = no label), first = the longer for equal_*
            "orig": (r.chosen, r.rejected, 1),
            "compress_chosen": (r.compressed, r.rejected, 1),
            "pad_rejected": (r.chosen, f"{r.rejected} {filler}", 1),
            "equal_compress": (r.chosen, r.compressed, -1),
            "equal_pad": (f"{r.chosen} {filler}", r.chosen, -1),
        }
        for v, (x, y, lab) in variants.items():
            flip = rng.random() < 0.5
            a, b = (y, x) if flip else (x, y)
            rows.append(dict(uid=f"{r.uid}-{v}", base_uid=r.uid, variant=v, source="hh_rlhf", domain="helpful",
                             prompt=r.prompt, response_a=a, response_b=b,
                             label=(-1 if lab < 0 else int(not flip)), p_a=0.5 if lab < 0 else float(not flip),
                             first_is_a=not flip))
    df = pd.DataFrame(rows)
    df["len_a"] = df.response_a.str.split().str.len()
    df["len_b"] = df.response_b.str.split().str.len()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LP_PATH, index=False)
    ok[["uid", "chosen", "compressed", "ratio"]].head(50).to_csv(OUT / "compression_review.csv", index=False)
    print(f"saved {len(df)} rows -> {LP_PATH}; spot-check 50 compressions in {OUT / 'compression_review.csv'}")


# ----------------------------------------------------------------------------- evaluation
def evaluate_run(run, df, track=True):
    from src.models.cross_encoder import load_checkpoint
    from src.training.trainer import make_predict_fn
    model, tok, cfg = load_checkpoint(f"checkpoints/{run}")
    collate = model.make_collator(tok)
    predict = make_predict_fn(model, tok, collate, batch_size=cfg["eval_batch_size"], bf16=cfg["bf16"],
                              max_prompt=cfg["max_prompt"], max_resp=MAX_RESP)
    p, p_swp = predict(df), predict(swap(df))
    p_sym = (p + 1 - p_swp) / 2

    res = {}
    for v, g in df.groupby("variant"):
        i = g.index.to_numpy()
        if v.startswith("equal"):
            first_a = g.first_is_a.to_numpy()  # "first" = the longer response
            pl, psl = np.where(first_a, p[i], 1 - p[i]), np.where(first_a, p_sym[i], 1 - p_sym[i])
            res[v] = {"p_prefers_longer": float((pl > 0.5).mean()), "mean_p_longer": float(pl.mean()),
                      "p_prefers_longer_sym": float((psl > 0.5).mean())}
        else:
            y = g.label.to_numpy()
            res[v] = {"accuracy": float(((p[i] > 0.5) == y).mean()),
                      "accuracy_sym": float(((p_sym[i] > 0.5) == y).mean())}
    res["compress_drop"] = res["orig"]["accuracy"] - res["compress_chosen"]["accuracy"]
    res["pad_drop"] = res["orig"]["accuracy"] - res["pad_rejected"]["accuracy"]

    json.dump(res, open(Path("results") / run / "length_perturbation.json", "w"), indent=2)
    tracking.start_run(f"{run}+biasB", {"base_run": run}, enabled=track)
    tracking.log({f"{v}/{k}": x for v, d in res.items() if isinstance(d, dict) for k, x in d.items()})
    tracking.log({"compress_drop": res["compress_drop"], "pad_drop": res["pad_drop"]})
    tracking.finish()
    print(f"\n### {run}")
    for v, d in res.items():
        print(f"  {v:16s} " + (" ".join(f"{k}={x:.3f}" for k, x in d.items()) if isinstance(d, dict) else f"{d:+.3f}"))
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=500)
    g.add_argument("--gen_model", default="Qwen/Qwen2.5-7B-Instruct")
    e = sub.add_parser("evaluate")
    e.add_argument("runs", nargs="+")
    e.add_argument("--no_track", action="store_true")
    args = ap.parse_args()

    if args.cmd == "generate":
        generate(args.n, args.gen_model)
    else:
        df = pd.read_parquet(LP_PATH).reset_index(drop=True)
        for run in args.runs:
            evaluate_run(run, df, track=not args.no_track)


if __name__ == "__main__":
    main()
