"""T5: bi-encoder (shared pretrained encoder) — the speed/accuracy and position-bias counterpart
of the cross-encoder.

Each response is scored on its own:  s(r) = head([CLS] of  [CLS] prompt [SEP] r [SEP])
and P(A preferred) = sigmoid(s(A) - s(B)). Swapping A and B exactly negates the logit, so the model
is position-invariant by construction, but A and B never attend to each other (no cross-attention).
Uses SiameseCollator; trained with the same loop as the cross-encoder (arch: bi in the config).
"""
from src.data.dataset import SiameseCollator
from src.models.cross_encoder import CrossEncoderModel


class BiEncoderModel(CrossEncoderModel):
    def score(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.drop(cls)).squeeze(-1)

    def forward(self, batch):
        return (self.score(batch["input_ids_a"], batch["attention_mask_a"])
                - self.score(batch["input_ids_b"], batch["attention_mask_b"]))

    def make_collator(self, tokenizer):
        return SiameseCollator(tokenizer)
