"""Presidio baseline, rules only: Microsoft Presidio's pattern recognizers, no NLP model.

Presidio usually runs a spaCy pipeline whose NER model finds names and places. We give it
its NoOpNlpEngine instead, which loads no model at all, so only self-contained rules run:
regular expressions, the phonenumbers library for phone numbers, and a checksum (Luhn)
for card numbers. We ask for the four types that match our labels (proposal, Appendix A.1).
Presidio finds no names or addresses, by design.

Without a model there are no tokens, so the context words ("phone", "email", ...) cannot
raise the scores. Presidio keeps every match whatever its score (default threshold 0), so
this only changes the scores, and, rarely, which of two overlapping matches of the same
type it keeps.
"""

from importlib.metadata import version

from PIIFilter.masking import Span

# Presidio entity -> our label. DATE_TIME is any date, broader than a date of birth.
LABEL_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD_NUMBER",
    "DATE_TIME": "DATE_OF_BIRTH",
}


class PresidioDetector:
    def __init__(self, dates: bool = False):
        """dates: comparison C, DATE_TIME becomes DATE instead of DATE_OF_BIRTH."""
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NoOpNlpEngine

        self.model_id = f"presidio-analyzer {version('presidio-analyzer')} (rules only)"
        self.label_map = (LABEL_MAP | {"DATE_TIME": "DATE"}) if dates else LABEL_MAP
        self.engine = AnalyzerEngine(nlp_engine=NoOpNlpEngine(models=[{"lang_code": "en", "model_name": "none"}]))

    def detect(self, text: str) -> list[Span]:
        results = self.engine.analyze(text, language="en", entities=list(self.label_map))
        return [Span(r.start, r.end, self.label_map[r.entity_type], round(r.score, 3)) for r in results]
