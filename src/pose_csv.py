from __future__ import annotations

import csv
import math
from pathlib import Path


REQUIRED_COLUMNS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "r11",
    "r12",
    "r13",
    "r21",
    "r22",
    "r23",
    "r31",
    "r32",
    "r33",
)


def load_pose_rows(csv_path: str | Path) -> list[dict[str, float]]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                "CSV file is missing required column(s): " + ", ".join(missing_columns)
            )

        pose_rows: list[dict[str, float]] = []
        for line_number, row in enumerate(reader, start=2):
            if _row_is_empty(row) or _row_is_marked_invalid(row):
                continue

            values: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                raw_value = row.get(column, "")
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value for '{column}' at CSV line {line_number}: {raw_value!r}"
                    ) from exc

                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Non-finite value for '{column}' at CSV line {line_number}: {raw_value!r}"
                    )

                values[column] = numeric_value

            for optional_column in ("source_row", "index"):
                raw_value = row.get(optional_column, "")
                if raw_value == "":
                    continue
                try:
                    values[optional_column] = float(raw_value)
                except (TypeError, ValueError):
                    continue

            pose_rows.append(values)

    if not pose_rows:
        raise ValueError(f"No valid pose rows found in CSV file: {csv_path}")

    return pose_rows


def _row_is_empty(row: dict[str, str | None]) -> bool:
    return not any((value or "").strip() for value in row.values())


def _row_is_marked_invalid(row: dict[str, str | None]) -> bool:
    raw_valid = row.get("valid")
    if raw_valid is None:
        return False
    return raw_valid.strip().lower() in {"0", "false", "no"}
