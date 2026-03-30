from __future__ import annotations

import argparse
from pathlib import Path

from app_settings import build_app_runtime_settings
from src.app_runner import run_robodk_program_generation, run_visualization


# -----------------------------------------------------------------------------
# 主参数区
# 这里保留最常改的业务参数；更细的算法/可视化/RoboDK 调参都在 `app_settings.py`。
# -----------------------------------------------------------------------------

VALIDATION_CENTERLINE_CSV = Path("data/validation_centerline.csv")  # 输入中心线 CSV。
TOOL_POSES_FRAME2_CSV = Path("data/tool_poses_frame2.csv")  # 输出的法兰位姿 CSV。

TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -400.0, 1200.0)  # 目标工艺坐标系 A 在 Frame 2 下的原点位置，单位 mm。
ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION = True  # 是否启用本项目自定义的多解 IK / 全局选姿 / 平滑 / 桥接。

ROBOT_NAME = "KUKA"  # RoboDK 站点中的机器人对象名。
FRAME_NAME = "Frame 2"  # RoboDK 站点中的参考坐标系名称。
PROGRAM_NAME = "Path_From_CSV"  # 输出到 RoboDK 的程序对象名。


APP_RUNTIME_SETTINGS = build_app_runtime_settings(
    validation_centerline_csv=VALIDATION_CENTERLINE_CSV,
    tool_poses_frame2_csv=TOOL_POSES_FRAME2_CSV,
    target_frame_origin_mm=TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM,
    enable_custom_smoothing_and_pose_selection=ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
    robot_name=ROBOT_NAME,
    frame_name=FRAME_NAME,
    program_name=PROGRAM_NAME,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "默认执行位姿求解并生成 RoboDK 程序。"
            "如果只想检查数据可视化，请传入 --visualize。"
        )
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="只做数据可视化，不生成 RoboDK 程序。",
    )
    return parser.parse_args()


def main() -> int:
    """程序入口。"""

    args = parse_args()

    if args.visualize:
        label = "可视化数据"
        action = run_visualization
    else:
        label = "生成 RoboDK 程序（包含位姿求解）"
        action = run_robodk_program_generation

    print(f"Running: {label}")

    try:
        action(APP_RUNTIME_SETTINGS)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
