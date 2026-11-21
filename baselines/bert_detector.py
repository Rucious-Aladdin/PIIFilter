"""BERT baseline: gravitee-io/bert-small-pii-detection, or a fine-tuned copy of it (question 3).

The model gives each token a BIO tag (B-PERSON for the first token of a name, I-PERSON for
the next ones, O outside any value). The transformers pipeline groups the tags into spans
with character positions (aggregation_strategy="simple", as in the model card). We keep
the six classes that match our labels (proposal, Appendix A.2) and drop the other 19.

The model files do not set a maximum length, so long texts would not be split and the
model would fail beyond 512 tokens. We read texts in windows of 256 tokens, the max_length
in the model's tokenizer config, that overlap by 64 tokens. The pipeline puts the spans
back together, keeping the longer one where two windows disagree.

The tokenizer lowercases every text first (do_lower_case), so this model never sees case.
"""

from PIIFilter.device import pick_device
from PIIFilter.masking import Span

# Model class -> our label
LABEL_MAP = {
    "PERSON": "NAME",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "ADDRESS",
    "DATE_TIME": "DATE_OF_BIRTH",
    "CREDIT_CARD": "CARD_NUMBER",
}


class BertDetector:
    default_model = "gravitee-io/bert-small-pii-detection"
    default_revision = "f8c27a85c51c0168f07b9dcf00265bf0a4097939"  # version used in the benchmark
    window_tokens = 256
    overlap_tokens = 64

    def __init__(self, model: str | None = None, device: str | None = None, dates: bool = False):
        """model: a Hugging Face id or a local folder (our fine-tuned versions).
        dates: comparison C, DATE_TIME becomes DATE instead of DATE_OF_BIRTH."""
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        self.model_id = model or self.default_model
        self.label_map = (LABEL_MAP | {"DATE_TIME": "DATE"}) if dates else LABEL_MAP
        self.device = pick_device(device)
        self.revision = self.default_revision if self.model_id == self.default_model else None
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        tokenizer.model_max_length = self.window_tokens
        self.pipe = pipeline(
            "token-classification",
            model=AutoModelForTokenClassification.from_pretrained(self.model_id, revision=self.revision),
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            stride=self.overlap_tokens,
            device=self.device,
        )

    def detect(self, text: str) -> list[Span]:
        return [
            Span(e["start"], e["end"], self.label_map[e["entity_group"]], round(float(e["score"]), 3))
            for e in self.pipe(text)
            if e["entity_group"] in self.label_map
        ]
