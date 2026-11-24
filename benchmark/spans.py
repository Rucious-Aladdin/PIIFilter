"""Clean a list of spans before scoring.

The same rules are used for the gold spans of every dataset and for the spans found by
every detector, so both sides are compared in the same way:

1. drop spans outside the text, and move span edges inward past whitespace and past a pair
   of brackets or quotes around the whole span ("[x@y.com]" -> "x@y.com");
2. remove a leading title from NAME spans ("Dr. Kim" -> "Kim", proposal Section 4.3);
3. drop overlapping spans, keeping the one that starts first (as the masking step does);
4. join spans with the same label that only spaces separate ("Minji" + "Park" -> "Minji Park").
"""

import re
from dataclasses import replace

from PIIFilter.masking import Span, drop_overlaps

# Mister, Madame, Master, Mayoress and Mstr: ai4privacy labels them TITLE as often as Mr.
TITLE_RE = re.compile(
    r"(?:mr|mrs|ms|miss|mx|dr|prof|sir|madam|mister|madame|master|mayoress|mstr)\b\.?\s*", re.IGNORECASE
)

# Opening -> closing character of the brackets and quotes that can enclose a value.
# Only a pair around the whole span is removed, so "(555) 123-4567" keeps its "(".
BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">", '"': '"', "'": "'", "“": "”", "‘": "’"}


def trim(text: str, span: Span) -> Span:
    """Move the span edges inward past whitespace and enclosing brackets or quotes."""
    start, end = span.start, span.end
    while True:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start >= 2 and BRACKETS.get(text[start]) == text[end - 1]:
            start, end = start + 1, end - 1
        else:
            return replace(span, start=start, end=end)


def strip_title(text: str, span: Span) -> Span:
    """Remove a leading title (Mr, Mrs, Ms, Dr, ...) from a NAME span."""
    if span.label != "NAME":
        return span
    match = TITLE_RE.match(text, span.start, span.end)
    return replace(span, start=match.end()) if match else span


def merge_neighbours(text: str, spans: list[Span]) -> list[Span]:
    """Join sorted, non-overlapping spans with the same label when only spaces separate them.

    A comma or a line break keeps them apart: "Doe, Jane" or two lines of a form.
    """
    merged: list[Span] = []
    for span in spans:
        last = merged[-1] if merged else None
        if last and last.label == span.label and text[last.end : span.start].strip(" ") == "":
            scores = [s for s in (last.score, span.score) if s is not None]
            merged[-1] = replace(last, end=span.end, score=min(scores) if scores else None)
        else:
            merged.append(span)
    return merged


def normalize_spans(text: str, spans: list[Span]) -> list[Span]:
    """Apply rules 1 to 4 above. Returns spans sorted by position."""
    spans = [strip_title(text, trim(text, s)) for s in spans if 0 <= s.start < s.end <= len(text)]
    spans = [s for s in spans if s.start < s.end]
    return merge_neighbours(text, drop_overlaps(spans))
