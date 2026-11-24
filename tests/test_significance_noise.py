"""The seed-averaged randomization test of experiments/significance_noise.py."""

import sys
from pathlib import Path

import numpy as np

from benchmark.analysis import randomization_test

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from significance_noise import TESTED, randomization  # noqa: E402


def counts(rng, n):
    tp = rng.integers(0, 6, n)
    fn = rng.integers(0, 3, n)
    return np.stack([tp, rng.integers(0, 3, n), fn, rng.integers(0, 1 + fn)], axis=1)


def test_one_seed_is_the_question_1_test():
    rng = np.random.default_rng(1)
    x, y = counts(rng, 300), counts(rng, 300)
    ours = randomization(x[:, None], y[:, None], rounds=2_000)
    for metric in TESTED:
        assert ours[metric] == randomization_test(x, y, metric, rounds=2_000)


def test_seeds_are_averaged_and_swapped_together():
    rng = np.random.default_rng(2)
    x = np.stack([counts(rng, 200) for _ in range(3)], axis=1)
    y = np.stack([counts(rng, 200) for _ in range(3)], axis=1)
    ours = randomization(x, y, rounds=500)
    from benchmark.analysis import scores
    for metric in TESTED:
        expected = np.mean(scores(x.sum(0))[metric]) - np.mean(scores(y.sum(0))[metric])
        assert abs(ours[metric][0] - expected) < 1e-12
        assert 0 < ours[metric][1] <= 1
    same = randomization(x, x, rounds=500)
    assert all(same[m] == (0.0, 1.0) for m in TESTED)
