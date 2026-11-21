"""GLiNER baseline: urchade/gliner_multi_pii-v1, given our labels as plain-word names.

No label mapping is needed beyond translating the label names (proposal, Appendix A.3).
GLiNER reads at most 384 words (max_len in the model's config) and ignores the rest of a
longer text, so long texts are read in windows of 384 words that overlap by 64 words
(baselines/windows.py). A value is at most 12 words long for GLiNER (max_width), so it is
always whole in at least one window.
"""

from PIIFilter.device import pick_device
from PIIFilter.masking import Span

from .windows import keep_longest, word_windows

# GLiNER label name -> our label
LABEL_NAMES = {
    "person": "NAME",
    "email": "EMAIL",
    "phone number": "PHONE",
    "street address": "ADDRESS",
    "date of birth": "DATE_OF_BIRTH",
    "credit card number": "CARD_NUMBER",
}

# Comparison C: "date" in place of "date of birth", at the same place in the list.
DATE_LABEL_NAMES = {
    "person": "NAME",
    "email": "EMAIL",
    "phone number": "PHONE",
    "street address": "ADDRESS",
    "date": "DATE",
    "credit card number": "CARD_NUMBER",
}


class GlinerDetector:
    default_model = "urchade/gliner_multi_pii-v1"
    default_revision = "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"  # version used in the benchmark
    window_words = 384
    overlap_words = 64

    def __init__(self, model: str | None = None, threshold: float = 0.5, device: str | None = None, dates: bool = False):
        """dates: comparison C, GLiNER is asked for "date" instead of "date of birth"."""
        from gliner import GLiNER

        self.model_id = model or self.default_model
        self.label_names = DATE_LABEL_NAMES if dates else LABEL_NAMES
        self.threshold = threshold
        self.device = pick_device(device)
        self.revision = self.default_revision if self.model_id == self.default_model else None
        self.model = GLiNER.from_pretrained(self.model_id, revision=self.revision).to(self.device)

    def detect(self, text: str, threshold: float | None = None) -> list[Span]:
        threshold = self.threshold if threshold is None else threshold
        windows = word_windows(text, self.window_words, self.overlap_words)
        outputs = self.model.inference(
            [text[start:end] for start, end in windows], list(self.label_names), threshold=threshold
        )
        spans = [
            Span(start + e["start"], start + e["end"], self.label_names[e["label"]], round(e["score"], 3))
            for (start, _), entities in zip(windows, outputs)
            for e in entities
        ]
        return keep_longest(spans)
