"""Token-budget formatter shared by all transformer models (plan section 3).

  [CLS] prompt [SEP] response_a [SEP] response_b [SEP]   (<= 512 tokens)
  prompt:    last 128 tokens (left-truncated, keeps the final user turn)
  responses: first 190 tokens each (right-truncated)
"""
MAX_PROMPT, MAX_RESP = 128, 190


def build_input(p_ids, a_ids, b_ids, tokenizer, max_prompt=MAX_PROMPT, max_resp=MAX_RESP):
    p_ids = p_ids[-max_prompt:] if max_prompt > 0 else []  # max_prompt=0 -> response-only ablation
    a_ids, b_ids = a_ids[:max_resp], b_ids[:max_resp]
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    input_ids = [cls] + p_ids + [sep] + a_ids + [sep] + b_ids + [sep]
    # segment 0 = prompt + response A, segment 1 = response B (used by BERT, ignored by DeBERTa-v3)
    n_first = len(p_ids) + len(a_ids) + 3
    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": [0] * n_first + [1] * (len(input_ids) - n_first)}


def format_input(prompt, resp_a, resp_b, tokenizer, max_prompt=MAX_PROMPT, max_resp=MAX_RESP):
    """Single example (used by the demo and bias tests)."""
    enc = lambda t: tokenizer.encode(t, add_special_tokens=False)
    return build_input(enc(prompt), enc(resp_a), enc(resp_b), tokenizer, max_prompt, max_resp)


def _tokenize(texts, tokenizer, batch_size=2000):
    texts, ids = list(texts), []
    for i in range(0, len(texts), batch_size):
        ids += tokenizer(texts[i:i + batch_size], add_special_tokens=False)["input_ids"]
    return ids


def encode_pairs(prompts, resp_a, resp_b, tokenizer, max_prompt=MAX_PROMPT, max_resp=MAX_RESP):
    """Batched format_input for whole splits. Returns (encodings, truncation stats)."""
    P, A, B = (_tokenize(t, tokenizer) for t in (prompts, resp_a, resp_b))
    enc = [build_input(p, a, b, tokenizer, max_prompt, max_resp) for p, a, b in zip(P, A, B)]
    n = max(len(P), 1)
    stats = {"prompt_truncated": sum(len(p) > max_prompt for p in P) / n,
             "resp_a_truncated": sum(len(a) > max_resp for a in A) / n,
             "resp_b_truncated": sum(len(b) > max_resp for b in B) / n}
    return enc, stats


if __name__ == "__main__":
    import pandas as pd
    from transformers import AutoTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()

    for name in ["bert-base-uncased", "microsoft/deberta-v3-base"]:
        tok = AutoTokenizer.from_pretrained(name)
        for path in ["data/processed/hh_rlhf_hard/val.parquet", "data/processed/shp_soft/test.parquet"]:
            df = pd.read_parquet(path).head(2000)
            enc, stats = encode_pairs(df.prompt, df.response_a, df.response_b, tok)
            lens = [len(e["input_ids"]) for e in enc]
            assert max(lens) <= 512, max(lens)
            print(f"{name:28s} {path.split('/')[-2]:13s} max_len={max(lens)} mean_len={sum(lens)/len(lens):.0f} "
                  + " ".join(f"{k}={v:.1%}" for k, v in stats.items()))
        print(tok.decode(enc[0]["input_ids"])[:400], "\n")
