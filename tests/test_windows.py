"""Tests for the windows used on long texts (baselines/windows.py)."""

from baselines.windows import WORD_RE, keep_longest, word_windows
from PIIFilter.masking import Span


def test_short_text_is_one_window():
    text = "I am Minji Park."
    assert word_windows(text, size=384, overlap=64) == [(0, len(text))]


def test_long_text_windows_cover_everything_and_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    windows = word_windows(text, size=384, overlap=64)
    words = [WORD_RE.findall(text[start:end]) for start, end in windows]
    assert len(windows) == 3  # windows start at words 0, 320 and 640
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    assert all(len(w) <= 384 for w in words)
    assert words[0][-64:] == words[1][:64]  # 64 shared words
    assert words[1][-64:] == words[2][:64]


def test_keep_longest_prefers_the_whole_value():
    cut = Span(10, 15, "NAME", 0.9)  # "Minji", cut at the end of a window
    whole = Span(10, 20, "NAME", 0.8)  # "Minji Park", whole in the next one
    assert keep_longest([cut, whole]) == [whole]


def test_keep_longest_keeps_one_copy_and_separate_spans():
    a, b = Span(0, 5, "NAME", 0.7), Span(0, 5, "NAME", 0.9)  # found by two windows
    other = Span(8, 12, "PHONE", 0.5)
    assert keep_longest([a, other, b]) == [b, other]
