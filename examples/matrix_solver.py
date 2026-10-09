"""Zero out exactly one row and one column of a matrix to minimise the remaining sum.

Not dynamic programming (as the original docstring claimed): with precomputed row
and column sums every (row, col) choice is scored in O(1), so the whole search is
O(N·M) time and O(N + M) extra space.

Removing row r and column c eliminates  row_sum[r] + col_sum[c] − A[r][c]
(the intersection would otherwise be subtracted twice), so minimising the remainder
means maximising that quantity. This also holds for negative entries, where zeroing
can *increase* the total and the best choice is the least-bad one.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import NamedTuple


class Solution(NamedTuple):
    remaining_sum: int
    row: int
    col: int


def find_minimum_matrix_sum(matrix: Sequence[Sequence[int]]) -> Solution:
    """Return the minimum achievable sum and the (row, col) that achieves it.

    Ties resolve to the smallest row, then the smallest column.
    Raises ValueError for empty or ragged input (the original silently mis-indexed).
    """
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    n_cols = len(matrix[0])
    if any(len(row) != n_cols for row in matrix):
        raise ValueError("matrix must be rectangular")

    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(col) for col in zip(*matrix, strict=True)]
    total = sum(row_sums)

    best: Solution | None = None
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            remaining = total - (row_sums[r] + col_sums[c] - value)
            if best is None or remaining < best.remaining_sum:
                best = Solution(remaining, r, c)
    assert best is not None
    return best


def random_matrix(n: int = 3, m: int = 3, *, lo: int = 10, hi: int = 99, seed: int | None = None) -> list[list[int]]:
    rng = random.Random(seed)  # noqa: S311 - demo data, not security sensitive
    return [[rng.randint(lo, hi) for _ in range(m)] for _ in range(n)]


if __name__ == "__main__":
    a = random_matrix()
    print("Input matrix:")
    for row in a:
        print(" ", row)
    total = sum(map(sum, a))
    result = find_minimum_matrix_sum(a)
    print(f"\nInitial sum:            {total}")
    print(f"Zero row / column:      {result.row} / {result.col}")
    print(f"Intersection element:   {a[result.row][result.col]}")
    print(f"Minimum remaining sum:  {result.remaining_sum}")
    print(f"Eliminated:             {total - result.remaining_sum}")
