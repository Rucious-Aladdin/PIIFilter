"""Noisy copies of a text: the mistakes people make by accident when typing quickly.

The kinds of noise follow the proposal (Section 4.2), plus a mix of them:

    typo    a letter added, dropped, swapped with the next one, or replaced by a
            neighbouring key on a QWERTY keyboard (example -> exmaple)
    space   a space dropped between two words, or added inside a word (isminji.park@...)
    case    a word with capitals typed in lowercase (Minji Park -> minji park)
    punct   punctuation at the end of a sentence or clause dropped, so sentences run together
    number  the separators in a number dropped or mixed (010-2345-6789 -> 01023456789, 010 2345-6789)
    mixed   each edit is one of the five kinds above, chosen at random

A copy makes about `rate` edits per word of the text (round(rate * words), at least one),
fixed before any scoring. A text with fewer places for that kind of noise gets fewer edits;
a text with none (a case copy of a text without capitals) is left as it is.

Every edit replaces a range [start, end) of the original text, never empty, with a new
string, and edits never overlap. An edit is either inside a gold span or outside every gold
span: an edit that would cross a span edge is not made, and neither is one that would leave
only spaces between two values with the same label (the clean-up would join them, changing
the gold spans themselves: "Roger Leblanc, Roger Leblanc" without its comma). So the gold
spans move with the
text: a span start moves by the length change of the edits before it, and a span end also
by the length change of the edits inside the span. A changed value keeps its label: a
misspelled email is still an email. The moved spans are then cleaned with the same rules as
every gold span (benchmark/spans.py).

Each edit is returned with its kind and whether it falls inside a gold span of the first
span list, so the loss can be split by noise inside a value or only in the words around it
(proposal, Section 5.2).
"""

import random
import re
from dataclasses import dataclass

from PIIFilter.masking import Span

from .spans import normalize_spans

NOISE_KINDS = ("typo", "space", "case", "punct", "number")
RATE = 0.10  # edits per word, chosen before any scoring

WORD_RE = re.compile(r"[A-Za-z]{3,}")  # typo targets
LONG_WORD_RE = re.compile(r"[A-Za-z]{4,}")  # a space can be added inside these
SPACE_RE = re.compile(r"(?<=\S) (?=\S)")  # one space between two words
CAPITALIZED_RE = re.compile(r"[A-Za-z]*[A-Z][A-Za-z]*")
PUNCT_RE = re.compile(r"[.,;:!?](?=\s|$)")  # end of a sentence or clause, not "x@y.com" or "1,234"
NUMBER_RE = re.compile(r"\d+(?:[ \-./]\d+)+")  # digit groups with separators
SEPARATOR_RE = re.compile(r"[ \-./]")

KEYBOARD = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _neighbour_keys() -> dict[str, str]:
    """Letter -> the letters next to it on a QWERTY keyboard (same row, and above and below)."""
    keys = {}
    for r, row in enumerate(KEYBOARD):
        for i, c in enumerate(row):
            near = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
            near += [KEYBOARD[q][i] for q in (r - 1, r + 1) if 0 <= q < len(KEYBOARD) and i < len(KEYBOARD[q])]
            keys[c] = "".join(near)
    return keys


NEIGHBOUR_KEYS = _neighbour_keys()


@dataclass(frozen=True)
class Edit:
    """Replace text[start:end] (never empty) with new. Positions are in the original text."""

    start: int
    end: int
    new: str
    kind: str

    @property
    def delta(self) -> int:
        return len(self.new) - (self.end - self.start)


def _near_key(rng: random.Random, c: str) -> str:
    near = rng.choice(NEIGHBOUR_KEYS[c.lower()])
    return near.upper() if c.isupper() else near


def _typo(rng: random.Random, text: str, start: int, end: int) -> Edit:
    word = text[start:end]
    op = rng.choice(("add", "drop", "swap", "key"))
    swappable = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
    if op == "swap" and swappable:
        i = rng.choice(swappable)
        return Edit(start + i, start + i + 2, word[i + 1] + word[i], "typo")
    i = rng.randrange(len(word))
    c = word[i]
    if op == "drop":
        return Edit(start + i, start + i + 1, "", "typo")
    if op == "add":  # the same letter twice, or a neighbouring key pressed too
        return Edit(start + i, start + i + 1, c + rng.choice((c, _near_key(rng, c))), "typo")
    return Edit(start + i, start + i + 1, _near_key(rng, c), "typo")


def _space(rng: random.Random, text: str, start: int, end: int) -> Edit:
    if text[start:end] == " ":  # drop a space between two words
        return Edit(start, end, "", "space")
    i = rng.randrange(start + 1, end - 2)  # add a space inside a word, not at its edges
    return Edit(i, i + 1, text[i] + " ", "space")


def _number(rng: random.Random, text: str, start: int, end: int) -> Edit:
    number = text[start:end]
    groups = SEPARATOR_RE.split(number)
    separators = SEPARATOR_RE.findall(number)
    if rng.random() < 0.5:  # drop every separator
        return Edit(start, end, "".join(groups), "number")
    while True:  # mix the separators
        mixed = [rng.choice((" ", "-", "")) for _ in separators]
        if mixed != separators:
            break
    new = groups[0] + "".join(sep + group for sep, group in zip(mixed, groups[1:]))
    return Edit(start, end, new, "number")


def _candidates(kind: str, text: str) -> list[tuple[int, int]]:
    """Ranges of the text where this kind of noise can be made."""
    if kind == "typo":
        return [m.span() for m in WORD_RE.finditer(text)]
    if kind == "space":
        return [m.span() for m in SPACE_RE.finditer(text)] + [m.span() for m in LONG_WORD_RE.finditer(text)]
    if kind == "case":
        return [m.span() for m in CAPITALIZED_RE.finditer(text)]
    if kind == "punct":
        return [m.span() for m in PUNCT_RE.finditer(text)]
    if kind == "number":
        return [m.span() for m in NUMBER_RE.finditer(text)]
    raise ValueError(f"unknown noise kind: {kind}")


def _make_edit(kind: str, rng: random.Random, text: str, start: int, end: int) -> Edit:
    if kind == "typo":
        return _typo(rng, text, start, end)
    if kind == "space":
        return _space(rng, text, start, end)
    if kind == "case":
        return Edit(start, end, text[start:end].lower(), "case")
    if kind == "punct":
        return Edit(start, end, "", "punct")
    return _number(rng, text, start, end)


def _crosses_a_span(edit: Edit, spans: list[Span]) -> bool:
    """True if the edit is partly inside and partly outside a span."""
    return any(
        not (edit.end <= s.start or edit.start >= s.end or (s.start <= edit.start and edit.end <= s.end))
        for s in spans
    )


def _overlaps(edit: Edit, edits: list[Edit]) -> bool:
    return any(edit.start < e.end and e.start < edit.end for e in edits)


def _joins_spans(edit: Edit, edits: list[Edit], span_lists: list[list[Span]], text: str) -> bool:
    """True if the edit would leave only spaces between two spans with the same label.

    The span clean-up would then join them ("Roger Leblanc, Roger Leblanc" without its comma
    becomes one NAME span), so the noise would change the gold spans, not only the text.
    """
    for spans in span_lists:
        ordered = sorted(spans, key=lambda s: s.start)
        for a, b in zip(ordered, ordered[1:]):
            if a.label == b.label and a.end <= edit.start and edit.end <= b.start:
                in_gap = [e for e in edits + [edit] if a.end <= e.start and e.end <= b.start]
                gap = apply_edits(text[: b.start], in_gap)[a.end :]
                if gap.strip(" ") == "":
                    return True
    return False


def choose_edits(text: str, span_lists: list[list[Span]], kind: str, rng: random.Random, rate: float = RATE) -> list[Edit]:
    """Pick the edits of one noisy copy, sorted by position. span_lists: every version of the
    gold spans of this text (e.g. the clean and dates versions); no edit crosses any of them."""
    if kind != "mixed" and kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind: {kind}")
    n_words = len(text.split())
    wanted = max(1, round(rate * n_words)) if rate > 0 and n_words else 0
    kinds = NOISE_KINDS if kind == "mixed" else (kind,)
    candidates = {k: _candidates(k, text) for k in kinds}
    for k in kinds:
        rng.shuffle(candidates[k])
    spans = [s for spans in span_lists for s in spans]
    edits: list[Edit] = []
    while len(edits) < wanted:
        available = [k for k in kinds if candidates[k]]
        if not available:
            break
        k = rng.choice(available)
        start, end = candidates[k].pop()
        edit = _make_edit(k, rng, text, start, end)
        if (
            edit.new == text[edit.start : edit.end]
            or _crosses_a_span(edit, spans)
            or _overlaps(edit, edits)
            or _joins_spans(edit, edits, span_lists, text)
        ):
            continue
        edits.append(edit)
    return sorted(edits, key=lambda e: e.start)


def apply_edits(text: str, edits: list[Edit]) -> str:
    parts, pos = [], 0
    for e in sorted(edits, key=lambda e: e.start):
        parts += [text[pos : e.start], e.new]
        pos = e.end
    parts.append(text[pos:])
    return "".join(parts)


def move_spans(spans: list[Span], edits: list[Edit]) -> list[Span]:
    """Move gold spans with the text. No edit crosses a span edge (choose_edits)."""
    moved = []
    for s in spans:
        start = s.start + sum(e.delta for e in edits if e.end <= s.start)
        end = s.end + sum(e.delta for e in edits if e.end <= s.end)
        moved.append(Span(start, end, s.label, s.score))
    return moved


def noisy_copy(
    text: str, span_lists: list[list[Span]], kind: str, rng: random.Random, rate: float = RATE
) -> tuple[str, list[list[Span]], list[dict]]:
    """A noisy copy of a text and of each list of its gold spans.

    Returns the new text, the moved and cleaned span lists (same order as span_lists), and
    the edits as JSON objects: kind, position in the original text, old and new string, and
    whether the edit is inside a span of span_lists[0].
    """
    edits = choose_edits(text, span_lists, kind, rng, rate)
    new_text = apply_edits(text, edits)
    new_lists = [normalize_spans(new_text, move_spans(spans, edits)) for spans in span_lists]
    first = span_lists[0] if span_lists else []
    records = [
        {
            "kind": e.kind, "start": e.start, "end": e.end, "old": text[e.start : e.end], "new": e.new,
            "inside": any(s.start <= e.start and e.end <= s.end for s in first),
        }
        for e in edits
    ]
    return new_text, new_lists, records


def example_rng(seed: int, kind: str, example_id: str) -> random.Random:
    """One random generator per example, so a copy does not depend on the order of the examples."""
    return random.Random(f"{seed}:{kind}:{example_id}")
