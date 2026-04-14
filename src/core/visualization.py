from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .frame_math import (
    CENTER_COLUMNS,
    LEFT_BOUNDARY_COLUMNS,
    RIGHT_BOUNDARY_COLUMNS,
    CenterlineDataError,
    FrameBuildOptions,
    format_issue_report,
    load_centerline_dataset,
)


def plot_centerline_frames(
    csv_path: str | Path,
    *,
    build_options: FrameBuildOptions,
    step: int,
    vector_scale: float,
    show_tangent: bool,
    show_normal: bool,
    show_side: bool,
    show_bad_rows: bool,
) -> None:
    """绘制中心线、边界和局部坐标系方向。"""
    if step <= 0:
        raise ValueError("Visualization step must be a positive integer.")
    if vector_scale <= 0.0:
        raise ValueError("Visualization vector scale must be positive.")

    # 可视化时要求左右边界列也存在。
    dataset = load_centerline_dataset(
        csv_path,
        require_boundaries=True,
        build_options=build_options,
    )
    print(format_issue_report(dataset.records))

    valid_records = dataset.valid_records()
    if not valid_records:
        raise CenterlineDataError("No valid rows are available for visualization.")

    # 从原始表格中取出中心线和左右边界点云。
    frame = dataset.frame
    center = frame.loc[:, list(CENTER_COLUMNS)].to_numpy(dtype=float)
    left = frame.loc[:, list(LEFT_BOUNDARY_COLUMNS)].to_numpy(dtype=float)
    right = frame.loc[:, list(RIGHT_BOUNDARY_COLUMNS)].to_numpy(dtype=float)

    # 每隔 step 个有效点采样一次，避免箭头过密。
    sampled_records = valid_records[::step] or [valid_records[0]]
    origins = np.vstack([record.center for record in sampled_records])

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")

    # 绘制中心线和左右边界。
    axis.plot(center[:, 0], center[:, 1], center[:, 2], linewidth=2.0, label="centerline")
    axis.plot(
        left[:, 0],
        left[:, 1],
        left[:, 2],
        linestyle="--",
        linewidth=1.0,
        color="tab:green",
        label="left boundary",
    )
    axis.plot(
        right[:, 0],
        right[:, 1],
        right[:, 2],
        linestyle="--",
        linewidth=1.0,
        color="tab:red",
        label="right boundary",
    )

    if show_tangent:
        # 切向对应局部坐标系的 +Y。
        tangents = np.vstack([record.tangent for record in sampled_records]) * vector_scale
        axis.quiver(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            tangents[:, 0],
            tangents[:, 1],
            tangents[:, 2],
            color="tab:orange",
            normalize=False,
            label="tangent (+y)",
        )

    if show_normal:
        # 法向对应局部坐标系的 +Z。
        normals = np.vstack([record.normal for record in sampled_records]) * vector_scale
        axis.quiver(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            color="tab:blue",
            normalize=False,
            label="surface normal (+z)",
        )

    if show_side:
        # 侧向对应局部坐标系的 +X。
        sides = np.vstack([record.side for record in sampled_records]) * vector_scale
        axis.quiver(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            sides[:, 0],
            sides[:, 1],
            sides[:, 2],
            color="tab:purple",
            normalize=False,
            label="right-handed side (+x)",
        )

    if show_bad_rows:
        # 把无效点单独标出来，方便快速排查数据问题。
        invalid_records = dataset.invalid_records()
        if invalid_records:
            invalid_points = np.vstack([record.center for record in invalid_records])
            axis.scatter(
                invalid_points[:, 0],
                invalid_points[:, 1],
                invalid_points[:, 2],
                color="black",
                marker="x",
                s=40,
                label="invalid rows",
            )

    # 标出路径起点和终点，方便判断整体方向。
    axis.scatter(
        center[0, 0],
        center[0, 1],
        center[0, 2],
        color="tab:green",
        s=60,
        marker="o",
        label="start",
    )
    axis.scatter(
        center[-1, 0],
        center[-1, 1],
        center[-1, 2],
        color="tab:red",
        s=60,
        marker="^",
        label="end",
    )

    _set_axes_equal(axis, np.vstack((center, left, right)))
    axis.set_title(
        f"Centerline and validated local frames ({len(valid_records)} valid, "
        f"{len(dataset.invalid_records())} invalid)"
    )
    axis.set_xlabel("X [mm]")
    axis.set_ylabel("Y [mm]")
    axis.set_zlabel("Z [mm]")
    axis.legend(loc="best")
    plt.tight_layout()

    # 无界面后端（例如 Agg）下不弹窗，直接结束。
    backend_name = plt.get_backend().lower()
    if "agg" in backend_name:
        plt.close(figure)
        return

    plt.show()


def _set_axes_equal(axis, points: np.ndarray) -> None:
    """让 3D 图的三个坐标轴使用相同尺度，避免形状失真。"""
    minimums = points.min(axis=0)
    maximums = points.max(axis=0)
    center = (minimums + maximums) / 2.0
    radius = float(np.max(maximums - minimums) / 2.0)
    radius = max(radius, 1.0)

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect((1.0, 1.0, 1.0))
