"""T2-T4: pretrained transformer cross-encoder (BERT-base, DeBERTa-v3-base/large).

Input  [CLS] prompt [SEP] A [SEP] B [SEP]  (token budget from src/utils/tokenization.py)
Output one logit for P(A preferred), from a linear head on the [CLS] vector.
"""
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoConfig, AutoModel, AutoTokenizer


class CrossEncoderModel(nn.Module):
    def __init__(self, model_name, dropout=0.1, gradient_checkpointing=False, pretrained=True):
        super().__init__()
        # force fp32 weights: some Hub checkpoints are stored in fp16, which otherwise mismatches
        # fp32 autocast-disabled training (needed for DeBERTa-v3, see configs/deberta_*.yaml)
        self.encoder = (AutoModel.from_pretrained(model_name, dtype=torch.float32) if pretrained
                        else AutoModel.from_config(AutoConfig.from_pretrained(model_name)))
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # BERT has 2 segment embeddings; DeBERTa-v3 has none (type_vocab_size = 0)
        self.use_token_type = getattr(self.encoder.config, "type_vocab_size", 0) > 1
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, batch):
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if self.use_token_type and "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        cls = self.encoder(**kwargs).last_hidden_state[:, 0]
        return self.head(self.drop(cls)).squeeze(-1)

    def param_groups(self, weight_decay):
        """No weight decay on biases and LayerNorm weights."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


def load_checkpoint(run_dir, device="cuda"):
    """Rebuild a trained cross-encoder from checkpoints/<run_name>/ (config.yaml + model.pt)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = CrossEncoderModel(cfg["model_name"], cfg["dropout"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(cfg["model_name"]), cfg
