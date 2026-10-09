import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from matrix_solver import find_minimum_matrix_sum, random_matrix  # noqa: E402


def brute_force(a):
    best = None
    for r, c in itertools.product(range(len(a)), range(len(a[0]))):
        s = sum(v for i, row in enumerate(a) for j, v in enumerate(row) if i != r and j != c)
        if best is None or s < best[0]:
            best = (s, r, c)
    return best


def test_known_example():
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # Removing row 2 and column 2 eliminates 7+8+9+3+6 = 33 → remaining 12.
    assert tuple(find_minimum_matrix_sum(a)) == (12, 2, 2)


@pytest.mark.parametrize("seed", range(200))
def test_matches_brute_force_including_negatives(seed):
    rng = random.Random(seed)
    n, m = rng.randint(1, 6), rng.randint(1, 6)
    a = [[rng.randint(-50, 50) for _ in range(m)] for _ in range(n)]
    assert tuple(find_minimum_matrix_sum(a)) == brute_force(a)


def test_single_cell():
    assert tuple(find_minimum_matrix_sum([[42]])) == (0, 0, 0)


@pytest.mark.parametrize("bad", [[], [[]], [[1, 2], [3]]])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        find_minimum_matrix_sum(bad)


def test_random_matrix_is_reproducible():
    assert random_matrix(seed=7) == random_matrix(seed=7)
