from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .frame_math import FrameBuildOptions
from .motion_settings import RoboDKMotionSettings
from .pose_solver import solve_tool_poses
from .robodk_program import create_program_from_csv
from .visualization import plot_centerline_frames


@dataclass(frozen=True)
class AppRuntimeSettings:
    """封装命令行入口真正需要的运行参数。"""

    validation_centerline_csv: Path
    tool_poses_frame2_csv: Path
    target_frame_origin_mm: tuple[float, float, float]
    target_frame_rotation_xyz_deg: tuple[float, float, float]
    robot_name: str
    frame_name: str
    program_name: str
    frame_build_options: FrameBuildOptions
    motion_settings: RoboDKMotionSettings
    enable_solver_verification: bool
    verification_row_ids: tuple[int, ...] | None
    verification_tolerance: float
    visualization_step: int
    visualization_vector_scale: float
    show_tangent: bool
    show_normal: bool
    show_side: bool
    show_bad_rows: bool


def run_visualization(settings: AppRuntimeSettings) -> None:
    """只做中心线及局部坐标系可视化。"""

    plot_centerline_frames(
        settings.validation_centerline_csv,
        build_options=settings.frame_build_options,
        step=settings.visualization_step,
        vector_scale=settings.visualization_vector_scale,
        show_tangent=settings.show_tangent,
        show_normal=settings.show_normal,
        show_side=settings.show_side,
        show_bad_rows=settings.show_bad_rows,
    )


def run_pose_solver(settings: AppRuntimeSettings) -> None:
    """先求解工具位姿，并写出 pose CSV。"""

    solve_tool_poses(
        settings.validation_centerline_csv,
        settings.tool_poses_frame2_csv,
        build_options=settings.frame_build_options,
        target_frame_origin_mm=settings.target_frame_origin_mm,
        target_frame_rotation_xyz_deg=settings.target_frame_rotation_xyz_deg,
        verify_solution=settings.enable_solver_verification,
        verification_row_ids=list(settings.verification_row_ids)
        if settings.verification_row_ids is not None
        else None,
        verification_tolerance=settings.verification_tolerance,
    )


def run_robodk_program_generation(settings: AppRuntimeSettings) -> None:
    """先求解位姿，再根据最新 CSV 生成 RoboDK 程序。"""

    run_pose_solver(settings)
    create_program_from_csv(
        settings.tool_poses_frame2_csv,
        robot_name=settings.robot_name,
        frame_name=settings.frame_name,
        program_name=settings.program_name,
        motion_settings=settings.motion_settings,
    )
