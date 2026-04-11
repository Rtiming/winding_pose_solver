from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import _rotation_matrix_from_xyz_offset_deg
from .frame_math import (
    FrameBuildOptions,
    FrameRecord,
    default_verification_row_ids,
    format_issue_report,
    invert_rigid_transform,
    load_centerline_dataset,
)


def solve_tool_poses(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    build_options: FrameBuildOptions,
    target_frame_origin_mm: tuple[float, float, float] | np.ndarray,
    target_frame_rotation_xyz_deg: tuple[float, float, float] | np.ndarray,
    verify_solution: bool,
    verification_row_ids: list[int] | None,
    verification_tolerance: float,
) -> pd.DataFrame:
    """求解每一行中心线对应的工具位姿，并输出为 CSV。"""
    # 先加载并检查中心线数据，同时构建每一行的工艺局部坐标系。
    dataset = load_centerline_dataset(
        input_csv_path,
        require_boundaries=False,
        build_options=build_options,
    )
    print(format_issue_report(dataset.records))

    # 目标坐标系 A 固定在 Frame 2 中。
    target_in_frame2 = build_target_frame_in_frame2(
        target_frame_origin_mm,
        target_frame_rotation_xyz_deg,
    )

    # 用行号保存工具位姿矩阵，后面做验证时会复用。
    tool_transforms_by_row_id: dict[int, np.ndarray] = {}

    # output_rows 会直接转成输出 CSV。
    output_rows: list[dict[str, object]] = []

    for record in dataset.records:
        # 每一行先写入基础信息，后面再补位姿结果。
        output_row: dict[str, object] = {
            "source_row": record.source_row,
            "index": record.row_id,
            "valid": record.valid,
            "issues": record.issue_text,
        }

        if record.valid:
            assert record.transform_tool_proc is not None

            # 工艺坐标系是以工具坐标系表达的：
            #   原点 = 当前中心线点
            #   +Y  = 切向
            #   +Z  = 表面法向
            #   +X  = Y x Z，保证右手系
            #
            # 目标坐标系 A 固定在 Frame 2 中。
            # 若希望工具上的工艺坐标系与 A 对齐，则满足：
            #   T_F2_tool * T_tool_proc = T_F2_A
            # 所以：
            #   T_F2_tool = T_F2_A * inv(T_tool_proc)
            transform_frame2_tool = target_in_frame2 @ invert_rigid_transform(
                record.transform_tool_proc
            )
            tool_transforms_by_row_id[record.row_id] = transform_frame2_tool

            # 从 4x4 变换矩阵中拆出旋转矩阵和平移向量。
            rotation = transform_frame2_tool[:3, :3]
            translation = transform_frame2_tool[:3, 3]
            output_row.update(
                {
                    "x_mm": translation[0],
                    "y_mm": translation[1],
                    "z_mm": translation[2],
                    "r11": rotation[0, 0],
                    "r12": rotation[0, 1],
                    "r13": rotation[0, 2],
                    "r21": rotation[1, 0],
                    "r22": rotation[1, 1],
                    "r23": rotation[1, 2],
                    "r31": rotation[2, 0],
                    "r32": rotation[2, 1],
                    "r33": rotation[2, 2],
                }
            )
        else:
            # 无效行仍然保留，但位姿字段写成 NaN，便于后处理筛选。
            output_row.update(
                {
                    "x_mm": np.nan,
                    "y_mm": np.nan,
                    "z_mm": np.nan,
                    "r11": np.nan,
                    "r12": np.nan,
                    "r13": np.nan,
                    "r21": np.nan,
                    "r22": np.nan,
                    "r23": np.nan,
                    "r31": np.nan,
                    "r32": np.nan,
                    "r33": np.nan,
                }
            )

        output_rows.append(output_row)

    # 生成结果表，并确保输出目录存在。
    result = pd.DataFrame(output_rows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv_path, index=False)

    valid_count = int(result["valid"].sum())
    print(f"Wrote {output_csv_path} with {valid_count} valid row(s) out of {len(result)}.")

    # 可选验证：随机或按指定行检查闭环误差。
    if verify_solution:
        print(
            verify_pose_solution(
                dataset.records,
                tool_transforms_by_row_id,
                target_in_frame2,
                requested_row_ids=verification_row_ids,
                tolerance=verification_tolerance,
            )
        )

    return result


def build_target_frame_in_frame2(
    origin_mm: tuple[float, float, float] | np.ndarray,
    rotation_xyz_deg: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """构造目标坐标系 A 在 Frame 2 下的齐次变换矩阵。"""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        _rotation_matrix_from_xyz_offset_deg(rotation_xyz_deg),
        dtype=float,
    )
    transform[:3, 3] = np.asarray(origin_mm, dtype=float)
    return transform


def verify_pose_solution(
    records: list[FrameRecord],
    tool_transforms_by_row_id: dict[int, np.ndarray],
    target_in_frame2: np.ndarray,
    *,
    requested_row_ids: list[int] | None,
    tolerance: float,
) -> str:
    """验证求解后的工具位姿是否能让工艺坐标系回到目标坐标系。"""
    # 如果用户没有指定验证行，就自动取开头 / 中间 / 末尾的有效点。
    selected_row_ids = requested_row_ids or default_verification_row_ids(records)
    if not selected_row_ids:
        return "Verification skipped: no valid rows available."

    # 只把有效记录放进映射表，后面按行号快速查找。
    valid_records = {record.row_id: record for record in records if record.valid}
    lines = [f"Verification samples: {selected_row_ids}"]
    missing_or_invalid: list[int] = []

    for row_id in selected_row_ids:
        record = valid_records.get(row_id)
        transform_frame2_tool = tool_transforms_by_row_id.get(row_id)
        if record is None or transform_frame2_tool is None:
            missing_or_invalid.append(row_id)
            continue

        assert record.transform_tool_proc is not None

        # 闭环检查：T_F2_tool * T_tool_proc 是否能回到 T_F2_A。
        closure = transform_frame2_tool @ record.transform_tool_proc
        translation_error_mm = float(np.linalg.norm(closure[:3, 3] - target_in_frame2[:3, 3]))
        rotation_error = float(np.max(np.abs(closure[:3, :3] - target_in_frame2[:3, :3])))
        passed = np.allclose(closure, target_in_frame2, atol=tolerance, rtol=0.0)

        status = "PASS" if passed else "FAIL"
        lines.append(
            f"  row {row_id}: {status} "
            f"(translation_error_mm={translation_error_mm:.3e}, "
            f"rotation_max_abs_error={rotation_error:.3e})"
        )

    if missing_or_invalid:
        lines.append(
            "  missing or invalid requested row(s): "
            + ", ".join(str(row_id) for row_id in missing_or_invalid)
        )

    return "\n".join(lines)
