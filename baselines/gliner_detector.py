"""GLiNER baseline: urchade/gliner_multi_pii-v1, given our labels as plain-word names.

No label mapping is needed beyond translating the label names (proposal, Appendix A.3).
"""

from PIIFilter.masking import Span

# GLiNER label name -> our label
LABEL_NAMES = {
    "person": "NAME",
    "email": "EMAIL",
    "phone number": "PHONE",
    "street address": "ADDRESS",
    "date of birth": "DATE_OF_BIRTH",
    "credit card number": "CARD_NUMBER",
}


class GlinerDetector:
    default_model = "urchade/gliner_multi_pii-v1"

    def __init__(self, model: str | None = None, threshold: float = 0.5, device: str = "cuda"):
        from gliner import GLiNER

        self.model_id = model or self.default_model
        self.threshold = threshold
        self.model = GLiNER.from_pretrained(self.model_id).to(device)

    def detect(self, text: str, threshold: float | None = None) -> list[Span]:
        entities = self.model.predict_entities(
            text, list(LABEL_NAMES), threshold=self.threshold if threshold is None else threshold
        )
        return [Span(e["start"], e["end"], LABEL_NAMES[e["label"]], round(e["score"], 3)) for e in entities]
