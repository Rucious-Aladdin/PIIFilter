"""The three detectors compared in the benchmark. Each one has detect(text) -> list[Span],
with character positions in the text and our six labels."""

from .bert_detector import BertDetector
from .gliner_detector import GlinerDetector
from .presidio_detector import PresidioDetector

DETECTORS = {"presidio": PresidioDetector, "bert": BertDetector, "gliner": GlinerDetector}
