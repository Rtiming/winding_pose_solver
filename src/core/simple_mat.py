from __future__ import annotations

from typing import Iterable, Sequence


class SimpleMat:
    """Minimal 4x4 matrix wrapper for offline SixAxisIK evaluation."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        normalized_rows = tuple(tuple(float(value) for value in row) for row in rows)
        if len(normalized_rows) != 4 or any(len(row) != 4 for row in normalized_rows):
            raise ValueError("SimpleMat expects a 4x4 row-major matrix.")
        self._rows = normalized_rows

    @property
    def rows(self) -> list[list[float]]:
        return [list(row) for row in self._rows]

    def size(self, dim: int) -> int:
        if dim in (0, 1):
            return 4
        raise IndexError(f"SimpleMat only supports dimensions 0 and 1, got {dim}.")

    def tolist(self) -> list[list[float]]:
        return self.rows

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_index, col_index = key
            return self._rows[row_index][col_index]
        return self._rows[key]

    def __iter__(self) -> Iterable[tuple[float, ...]]:
        return iter(self._rows)

    def __len__(self) -> int:
        return 4
