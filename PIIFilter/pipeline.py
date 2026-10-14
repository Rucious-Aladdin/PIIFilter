"""The whole filter: detect -> mask -> LLM -> unmask."""

from dataclasses import dataclass, field

from .masking import Span, mask, unmask


@dataclass
class FilterTrace:
    """Output of every step for one prompt."""

    prompt: str
    spans: list[Span]
    masked_prompt: str  # the only text that leaves the device
    mapping: dict[str, str] = field(repr=False)  # stays on the device
    masked_answer: str
    answer: str  # shown to the user


def run_filter(prompt: str, detector, llm) -> FilterTrace:
    """detector.detect(text) -> list[Span]; llm.chat(text) -> str."""
    spans = detector.detect(prompt)
    masked_prompt, mapping = mask(prompt, spans)
    masked_answer = llm.chat(masked_prompt)
    return FilterTrace(prompt, spans, masked_prompt, mapping, masked_answer, unmask(masked_answer, mapping))
