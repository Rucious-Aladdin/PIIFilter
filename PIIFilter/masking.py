"""Masking and unmasking with numbered placeholders.

mask() replaces each detected span with a placeholder such as <NAME_1> and returns
the placeholder-to-value mapping, which stays on the device. unmask() puts the stored
values back by simple replacement.
"""

import re
from dataclasses import dataclass

LABELS = ("NAME", "EMAIL", "PHONE", "ADDRESS", "DATE_OF_BIRTH", "CARD_NUMBER")

PLACEHOLDER_RE = re.compile(r"<(?:%s)_\d+>" % "|".join(LABELS))


@dataclass(frozen=True)
class Span:
    """Character span [start, end) with one of our labels."""

    start: int
    end: int
    label: str
    score: float | None = None


def drop_overlaps(spans: list[Span]) -> list[Span]:
    """Keep spans left to right, dropping any that overlap an already kept span."""
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if not kept or span.start >= kept[-1].end:
            kept.append(span)
    return kept


def mask(text: str, spans: list[Span]) -> tuple[str, dict[str, str]]:
    """Replace each span with <LABEL_n>, numbered per label in order of appearance.

    The same value with the same label always gets the same placeholder.
    """
    mapping: dict[str, str] = {}
    placeholder_of: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    parts, pos = [], 0
    for span in drop_overlaps(spans):
        value = text[span.start : span.end]
        key = (span.label, value)
        if key not in placeholder_of:
            counts[span.label] = counts.get(span.label, 0) + 1
            placeholder = f"<{span.label}_{counts[span.label]}>"
            placeholder_of[key] = placeholder
            mapping[placeholder] = value
        parts += [text[pos : span.start], placeholder_of[key]]
        pos = span.end
    parts.append(text[pos:])
    return "".join(parts), mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Replace every known placeholder in text with its stored value."""
    return PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
