"""Split long texts into overlapping windows, for models with a length limit.

GLiNER reads at most 384 words and ignores the rest of a longer text, so the detector runs
it on overlapping windows and puts the spans found back at their place in the full text.
(BERT does the same through the transformers pipeline, see bert_detector.py.)
"""

import re

from PIIFilter.masking import Span

# GLiNER's default word splitter: words, and each other non-space character on its own.
WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def word_windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges (start, end) of windows of at most `size` words.

    Each window shares `overlap` words with the next one, so a value cut at the end of a
    window is whole in the next. A text of `size` words or fewer is a single window.
    """
    words = [m.span() for m in WORD_RE.finditer(text)]
    if len(words) <= size:
        return [(0, len(text))]
    windows = []
    for first in range(0, len(words), size - overlap):
        last = min(first + size, len(words)) - 1
        windows.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return windows


def keep_longest(spans: list[Span]) -> list[Span]:
    """Resolve spans that overlap because two windows found them: keep the longer one, or
    the one with the higher score if they are as long. It is the rule the transformers
    pipeline uses for its own windows.
    """
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        last = kept[-1] if kept else None
        if last and last.start <= span.start < last.end:
            longer = span.end - span.start > last.end - last.start
            as_long_better = span.end - span.start == last.end - last.start and (span.score or 0) > (last.score or 0)
            if longer or as_long_better:
                kept[-1] = span
        else:
            kept.append(span)
    return kept
