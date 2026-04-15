from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# 中心线点坐标列。
CENTER_COLUMNS = ("x", "y", "z")

# 输入数据中的切向列，后续会作为局部坐标系的 +Y。
TANGENT_COLUMNS = ("tx", "ty", "tz")

# 输入数据中的法向列，后续会作为局部坐标系的 +Z 候选方向。
NORMAL_COLUMNS = ("nx", "ny", "nz")

# 如果原始数据里已经带有侧向列，会一并读取，但当前算法仍会重新正交化。
SIDE_COLUMNS = ("sx", "sy", "sz")

# 可视化时使用的左右边界列。
LEFT_BOUNDARY_COLUMNS = ("left_x", "left_y", "left_z")
RIGHT_BOUNDARY_COLUMNS = ("right_x", "right_y", "right_z")


class CenterlineDataError(ValueError):
    """中心线 CSV 缺少必要数据时抛出的错误。"""


@dataclass(frozen=True)
class FrameBuildOptions:
    """构建局部坐标系时使用的数值参数。"""

    zero_tolerance: float = 1e-9  # 判定零向量的阈值
    parallel_tolerance: float = 1e-6  # 切向与法向近平行时的阈值
    continuity_dot_threshold: float = 0.0  # 与上一有效行的连续性阈值


@dataclass
class FrameRecord:
    """保存单行中心线数据及其构建出的局部坐标系结果。"""

    source_row: int  # 原始 DataFrame 行号
    row_id: int  # 对外展示或输出使用的行号
    center: np.ndarray  # 中心线点坐标
    raw_tangent: np.ndarray  # 原始切向
    raw_normal: np.ndarray  # 原始法向
    raw_side: np.ndarray | None  # 原始侧向，可为空
    tangent: np.ndarray | None = None  # 修正后的切向 +Y
    normal: np.ndarray | None = None  # 修正后的法向 +Z
    side: np.ndarray | None = None  # 修正后的侧向 +X
    transform_tool_proc: np.ndarray | None = None  # 工艺坐标系在工具系下的变换
    valid: bool = True  # 当前行是否有效
    issues: list[str] = field(default_factory=list)  # 当前行的问题说明

    @property
    def issue_text(self) -> str:
        """把问题列表拼接成便于输出的单行文本。"""
        return "; ".join(self.issues)


@dataclass
class CenterlineDataset:
    """封装中心线 CSV 路径、原始表格和构建后的记录列表。"""

    csv_path: Path  # 当前数据源路径
    frame: pd.DataFrame  # 原始 DataFrame
    records: list[FrameRecord]  # 每一行对应的解析结果

    def valid_records(self) -> list[FrameRecord]:
        """返回所有有效记录。"""
        return [record for record in self.records if record.valid]

    def invalid_records(self) -> list[FrameRecord]:
        """返回所有无效记录。"""
        return [record for record in self.records if not record.valid]


def load_centerline_dataset(
    csv_path: str | Path,
    *,
    require_boundaries: bool,
    build_options: FrameBuildOptions,
) -> CenterlineDataset:
    """读取中心线 CSV，并构建每一行的局部坐标系记录。"""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)

    # 基础计算必须包含中心点、切向和法向。
    required_columns = list(CENTER_COLUMNS + TANGENT_COLUMNS + NORMAL_COLUMNS)
    if require_boundaries:
        # 可视化模式还要求包含左右边界。
        required_columns.extend(LEFT_BOUNDARY_COLUMNS)
        required_columns.extend(RIGHT_BOUNDARY_COLUMNS)

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise CenterlineDataError(
            f"Missing required column(s) in {csv_path}: {', '.join(missing_columns)}"
        )

    records = build_frame_records(frame, build_options=build_options)
    return CenterlineDataset(csv_path=csv_path, frame=frame, records=records)


def build_frame_records(
    frame: pd.DataFrame,
    *,
    build_options: FrameBuildOptions,
) -> list[FrameRecord]:
    """把 DataFrame 中的每一行转换为局部坐标系记录。"""
    records: list[FrameRecord] = []

    # previous_valid_record 用来保证相邻有效行的方向尽量连续。
    previous_valid_record: FrameRecord | None = None

    for source_row, (_, row) in enumerate(frame.iterrows()):
        # row_id 优先取 CSV 中的 index 列；没有时退回到行号。
        row_id = _row_identifier(row, source_row)

        # 读取当前行的中心点、切向、法向和可选侧向。
        center = _vector_from_row(row, CENTER_COLUMNS)
        raw_tangent = _vector_from_row(row, TANGENT_COLUMNS)
        raw_normal = _vector_from_row(row, NORMAL_COLUMNS)
        raw_side = _vector_from_row(row, SIDE_COLUMNS) if all(
            column in frame.columns for column in SIDE_COLUMNS
        ) else None

        record = FrameRecord(
            source_row=source_row,
            row_id=row_id,
            center=center,
            raw_tangent=raw_tangent,
            raw_normal=raw_normal,
            raw_side=raw_side,
        )

        # 第一步：尝试归一化原始切向和法向。
        tangent = _try_normalize(raw_tangent, build_options.zero_tolerance)
        if tangent is None:
            record.valid = False
            record.issues.append("zero tangent vector")

        normal_hint = _try_normalize(raw_normal, build_options.zero_tolerance)
        if normal_hint is None:
            record.valid = False
            record.issues.append("zero normal vector")

        if not record.valid:
            records.append(record)
            continue

        assert tangent is not None
        assert normal_hint is not None

        if previous_valid_record is not None:
            # 让当前切向和法向尽量与上一有效行保持同向，减少突然翻转。
            tangent = _align_with_previous(
                tangent,
                previous_valid_record.tangent,
                "tangent",
                record.issues,
                build_options.continuity_dot_threshold,
            )
            normal_hint = _align_with_previous(
                normal_hint,
                previous_valid_record.normal,
                "normal",
                record.issues,
                build_options.continuity_dot_threshold,
            )

        # 第二步：把法向从切向中正交化，避免两者不垂直。
        normal_remainder = normal_hint - np.dot(normal_hint, tangent) * tangent
        remainder_norm = np.linalg.norm(normal_remainder)
        tangent_normal_dot = float(np.dot(tangent, normal_hint))
        if remainder_norm < build_options.parallel_tolerance:
            record.valid = False
            record.issues.append(
                f"tangent and normal are near parallel (dot={tangent_normal_dot:.6f})"
            )
            records.append(record)
            continue

        normal = normal_remainder / remainder_norm

        # 第三步：用右手定则计算侧向 +X。
        side = _try_normalize(np.cross(tangent, normal), build_options.zero_tolerance)
        if side is None:
            record.valid = False
            record.issues.append("failed to build right-handed x-axis from tangent and normal")
            records.append(record)
            continue

        # 再用 side 和 tangent 回算一次 normal，保证三轴严格正交。
        normal = normalize_vector(np.cross(side, tangent), eps=build_options.zero_tolerance)

        if previous_valid_record is not None:
            assert previous_valid_record.side is not None

            # side 方向如果相对上一有效行发生翻转，记录提示信息。
            side_dot = float(np.dot(side, previous_valid_record.side))
            if side_dot < build_options.continuity_dot_threshold:
                record.issues.append(
                    f"computed x-axis flips relative to previous valid row (dot={side_dot:.6f})"
                )

        # 把最终结果写回记录对象。
        record.tangent = tangent
        record.normal = normal
        record.side = side
        record.transform_tool_proc = make_transform(center, side, tangent, normal)

        records.append(record)
        previous_valid_record = record

    return records


def normalize_vector(vector: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """归一化向量；如果向量太小则直接报错。"""
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """求刚体变换矩阵的逆。"""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def make_transform(
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    """根据原点和三轴方向构造 4x4 齐次变换矩阵。"""
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = origin
    return transform


def format_issue_report(records: list[FrameRecord], *, max_rows: int = 10) -> str:
    """把记录中的问题整理成便于终端查看的报告文本。"""
    rows_with_issues = [record for record in records if record.issues]
    if not rows_with_issues:
        return "No row issues detected."

    lines = [f"Rows with issues: {len(rows_with_issues)} / {len(records)}"]
    for record in rows_with_issues[:max_rows]:
        status = "invalid" if not record.valid else "repaired"
        lines.append(
            f"  row {record.row_id} (source row {record.source_row}, {status}): "
            f"{record.issue_text}"
        )

    remaining = len(rows_with_issues) - max_rows
    if remaining > 0:
        lines.append(f"  ... {remaining} more row(s)")

    return "\n".join(lines)


def default_verification_row_ids(records: list[FrameRecord]) -> list[int]:
    """自动挑选开头、中间、末尾三个有效行号用于验证。"""
    valid_records = [record for record in records if record.valid]
    if not valid_records:
        return []

    candidate_positions = [0, len(valid_records) // 2, len(valid_records) - 1]
    selected_positions: list[int] = []
    for position in candidate_positions:
        if position not in selected_positions:
            selected_positions.append(position)

    return [valid_records[position].row_id for position in selected_positions]


def _align_with_previous(
    vector: np.ndarray,
    previous_vector: np.ndarray | None,
    axis_name: str,
    issues: list[str],
    threshold: float,
) -> np.ndarray:
    """如果当前方向与上一有效行相反，则自动翻转以保持连续。"""
    if previous_vector is None:
        return vector

    dot_value = float(np.dot(vector, previous_vector))
    if dot_value < threshold:
        issues.append(
            f"{axis_name} direction flips relative to previous valid row "
            f"(dot={dot_value:.6f}); using the negated direction"
        )
        return -vector
    return vector


def _try_normalize(vector: np.ndarray, eps: float) -> np.ndarray | None:
    """尝试归一化向量；若过小则返回 None。"""
    norm = np.linalg.norm(vector)
    if norm < eps:
        return None
    return vector / norm


def _vector_from_row(row: pd.Series, columns: tuple[str, str, str]) -> np.ndarray:
    """从一行数据中提取三维向量。"""
    return row.loc[list(columns)].to_numpy(dtype=float)


def _row_identifier(row: pd.Series, source_row: int) -> int:
    """优先使用 index 列作为行号，否则退回到原始行号。"""
    if "index" not in row.index:
        return source_row

    value = row["index"]
    if pd.isna(value):
        return source_row

    try:
        return int(value)
    except (TypeError, ValueError):
        return source_row
