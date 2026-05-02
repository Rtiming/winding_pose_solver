"""Thin user control panel + entrypoint.

`src/runtime/main_entrypoint.py` owns implementation and CLI wiring.
This file intentionally keeps only the commonly adjusted project knobs.
"""

from __future__ import annotations

from typing import Any

from src.runtime import main_entrypoint as _entry


# ---------------------------------------------------------------------------
# Frequently adjusted project knobs
# ---------------------------------------------------------------------------

# Frame-A origin in Frame 2, in millimeters: (X, Y, Z).
# When origin search is enabled, this is also the smart-square search center.
TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -247.5, 977.5)
# Frame-A rotation in Frame 2, in degrees: (Rx, Ry, Rz).
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (0.0, 0.0, -180.0)

# Main mode: "single" | "online" | "origin_search".
# ENABLE_TARGET_ORIGIN_YZ_SEARCH=True takes priority and enters origin_search first.
RUN_MODE = "online"

# Local machine profile: "auto" | "mac" | "windows" | "linux".
# This only describes the coordinator/receiver machine; solver behavior remains shared.
LOCAL_MACHINE_PROFILE = "auto"

# Single-machine action: "solve" | "program" | "visualize".
SINGLE_ACTION = "program"

# Online role: "coordinator" | "server" | "receiver".
ONLINE_ROLE = "coordinator"
ONLINE_HOST = "master"
ONLINE_SERVER_DIR = "/home/tzwang/program/winding_pose_solver"
ONLINE_ENV_NAME = "winding_pose_solver"
ONLINE_FINAL_GENERATE_PROGRAM = True

# Online compute policy:
# Local coordinator handles sync and final RoboDK import; heavy continuity
# retry/repair runs in the server role under Slurm.
ONLINE_PROFILE_RETRY_CANDIDATE_LIMIT = 24
ONLINE_PROFILE_RETRY_REPAIR_LIMIT = 2
ONLINE_PROFILE_RETRY_MAX_ROUNDS = 3

# Local exact-profile evaluation parallelism (six_axis_ik only).
# Keep enabled for heavy continuity repair workloads.
ENABLE_LOCAL_MULTIPROCESS_PARALLEL = True
LOCAL_PARALLEL_WORKERS = 0  # 0 = auto
LOCAL_PARALLEL_MIN_BATCH_SIZE = 4

# IK branch policy:
# - lower config bit 1 is elbow-down for the embedded SixAxisIK backend.
# - Set REQUIRE_LOWER_CONFIG_FLAG = 1 to force elbow-down globally.
PREFERRED_LOWER_CONFIG_FLAG: int | None = None
LOWER_CONFIG_PREFERENCE_WEIGHT = 0.0
REQUIRE_LOWER_CONFIG_FLAG: int | None = None
ENABLE_PERIODIC_TRANSITION_UNWRAP = True
PERIODIC_CONTINUITY_JOINT_INDICES = (3, 5)  # A4/A6 are periodic in continuity scoring.
PERIODIC_CONTINUITY_EXPANSION_TURNS = 1

# Load-aware posture shaping for contact winding. These are soft costs used
# after reachability, so they prefer lower shoulder load without hiding failures.
POSTURE_STRESS_PENALTY_WEIGHT = 260.0
POSTURE_WRIST_REACH_SOFT_LIMIT_MM = 1750.0
POSTURE_WRIST_REACH_HARD_LIMIT_MM = 2000.0
POSTURE_ARM_EXTENSION_SOFT_RATIO = 0.86
POSTURE_ARM_EXTENSION_HARD_RATIO = 0.96
POSTURE_A2_UPPER_SOFT_DEG = 100.0
POSTURE_A2_UPPER_HARD_DEG = 115.0
POSTURE_WRIST_REACH_COMPONENT_WEIGHT = 1.0
POSTURE_ARM_EXTENSION_COMPONENT_WEIGHT = 1.2
POSTURE_A2_UPPER_COMPONENT_WEIGHT = 1.5

# Endpoint-locked path fallback:
# keep the configured Frame-A origin at the first and terminal winding rows,
# while allowing middle-row Y/Z profile offsets to improve motion continuity.
LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS = True
ENABLE_FIXED_POINT_PATH_FALLBACK = True
FIXED_POINT_PATH_FALLBACK_MAX_CANDIDATES_PER_CONFIG = 12
FIXED_POINT_PATH_FALLBACK_DISABLE_GUIDED_PATH = True
FIXED_POINT_PATH_FALLBACK_RUN_WINDOW_REPAIR = True
FIXED_POINT_PATH_FALLBACK_RUN_INSERTED_REPAIR = False

# Remote sync policy before online coordinator: "off" | "guard" | "push".
REMOTE_SYNC_MODE = "push"
ENFORCE_REMOTE_SYNC_GUARD = True

# Keep False for interactive runs so diagnostic artifacts can still be inspected.
STRICT_EXIT_ON_INVALID = False

# Run smart-square Y/Z search before SINGLE_ACTION / ONLINE_ROLE.
ENABLE_TARGET_ORIGIN_YZ_SEARCH = True
TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM = 400.0
TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM = 100.0
TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM = 5.0
TARGET_ORIGIN_YZ_SEARCH_MAX_ITERS = 5
TARGET_ORIGIN_YZ_SEARCH_BEAM_WIDTH = 4
TARGET_ORIGIN_YZ_SEARCH_USE_SERVER = True
TARGET_ORIGIN_YZ_SEARCH_POLISH_STEP_MM = 5.0

# Origin-search output policy:
# - report several official-gate candidates for comparison;
# - only materialize/import the best official candidate by default;
# - do not import debug/invalid candidates unless explicitly changed here or via CLI.
TARGET_ORIGIN_YZ_SEARCH_USABLE_COUNT = 5
TARGET_ORIGIN_YZ_SEARCH_POST_DISPATCH = "online_role"  # "none" | "single_action" | "online_role"
TARGET_ORIGIN_YZ_SEARCH_POST_TOP_N = 1
DEBUG_FALLBACK_IMPORT_COUNT = 0
ALLOW_INVALID_DEBUG_OUTPUTS = False

# If the configured square has no official deliverable, search outward from
# the square boundary and report the nearest official alternatives.
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_COUNT = 3
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_MAX_RINGS = 6
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_RING_STEP_MM = 25.0
TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_EDGE_STEP_MM = 75.0


def _override_map() -> dict[str, object]:
    return {
        "TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM": TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM,
        "TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG": (
            TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG
        ),
        "RUN_MODE": RUN_MODE,
        "LOCAL_MACHINE_PROFILE": LOCAL_MACHINE_PROFILE,
        "SINGLE_ACTION": SINGLE_ACTION,
        "ONLINE_ROLE": ONLINE_ROLE,
        "ONLINE_ACTION": ONLINE_ROLE,
        "ONLINE_HOST": ONLINE_HOST,
        "ONLINE_SERVER_DIR": ONLINE_SERVER_DIR,
        "ONLINE_ENV_NAME": ONLINE_ENV_NAME,
        "ONLINE_FINAL_GENERATE_PROGRAM": ONLINE_FINAL_GENERATE_PROGRAM,
        "ONLINE_PROFILE_RETRY_CANDIDATE_LIMIT": ONLINE_PROFILE_RETRY_CANDIDATE_LIMIT,
        "ONLINE_PROFILE_RETRY_REPAIR_LIMIT": ONLINE_PROFILE_RETRY_REPAIR_LIMIT,
        "ONLINE_PROFILE_RETRY_MAX_ROUNDS": ONLINE_PROFILE_RETRY_MAX_ROUNDS,
        "ENABLE_LOCAL_MULTIPROCESS_PARALLEL": ENABLE_LOCAL_MULTIPROCESS_PARALLEL,
        "LOCAL_PARALLEL_WORKERS": LOCAL_PARALLEL_WORKERS,
        "LOCAL_PARALLEL_MIN_BATCH_SIZE": LOCAL_PARALLEL_MIN_BATCH_SIZE,
        "PREFERRED_LOWER_CONFIG_FLAG": PREFERRED_LOWER_CONFIG_FLAG,
        "LOWER_CONFIG_PREFERENCE_WEIGHT": LOWER_CONFIG_PREFERENCE_WEIGHT,
        "REQUIRE_LOWER_CONFIG_FLAG": REQUIRE_LOWER_CONFIG_FLAG,
        "ENABLE_PERIODIC_TRANSITION_UNWRAP": ENABLE_PERIODIC_TRANSITION_UNWRAP,
        "PERIODIC_CONTINUITY_JOINT_INDICES": PERIODIC_CONTINUITY_JOINT_INDICES,
        "PERIODIC_CONTINUITY_EXPANSION_TURNS": PERIODIC_CONTINUITY_EXPANSION_TURNS,
        "POSTURE_STRESS_PENALTY_WEIGHT": POSTURE_STRESS_PENALTY_WEIGHT,
        "POSTURE_WRIST_REACH_SOFT_LIMIT_MM": POSTURE_WRIST_REACH_SOFT_LIMIT_MM,
        "POSTURE_WRIST_REACH_HARD_LIMIT_MM": POSTURE_WRIST_REACH_HARD_LIMIT_MM,
        "POSTURE_ARM_EXTENSION_SOFT_RATIO": POSTURE_ARM_EXTENSION_SOFT_RATIO,
        "POSTURE_ARM_EXTENSION_HARD_RATIO": POSTURE_ARM_EXTENSION_HARD_RATIO,
        "POSTURE_A2_UPPER_SOFT_DEG": POSTURE_A2_UPPER_SOFT_DEG,
        "POSTURE_A2_UPPER_HARD_DEG": POSTURE_A2_UPPER_HARD_DEG,
        "POSTURE_WRIST_REACH_COMPONENT_WEIGHT": POSTURE_WRIST_REACH_COMPONENT_WEIGHT,
        "POSTURE_ARM_EXTENSION_COMPONENT_WEIGHT": POSTURE_ARM_EXTENSION_COMPONENT_WEIGHT,
        "POSTURE_A2_UPPER_COMPONENT_WEIGHT": POSTURE_A2_UPPER_COMPONENT_WEIGHT,
        "LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS": (
            LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS
        ),
        "ENABLE_FIXED_POINT_PATH_FALLBACK": ENABLE_FIXED_POINT_PATH_FALLBACK,
        "FIXED_POINT_PATH_FALLBACK_MAX_CANDIDATES_PER_CONFIG": (
            FIXED_POINT_PATH_FALLBACK_MAX_CANDIDATES_PER_CONFIG
        ),
        "FIXED_POINT_PATH_FALLBACK_DISABLE_GUIDED_PATH": (
            FIXED_POINT_PATH_FALLBACK_DISABLE_GUIDED_PATH
        ),
        "FIXED_POINT_PATH_FALLBACK_RUN_WINDOW_REPAIR": (
            FIXED_POINT_PATH_FALLBACK_RUN_WINDOW_REPAIR
        ),
        "FIXED_POINT_PATH_FALLBACK_RUN_INSERTED_REPAIR": (
            FIXED_POINT_PATH_FALLBACK_RUN_INSERTED_REPAIR
        ),
        "REMOTE_SYNC_MODE": REMOTE_SYNC_MODE,
        "ENFORCE_REMOTE_SYNC_GUARD": ENFORCE_REMOTE_SYNC_GUARD,
        "STRICT_EXIT_ON_INVALID": STRICT_EXIT_ON_INVALID,
        "ENABLE_TARGET_ORIGIN_YZ_SEARCH": ENABLE_TARGET_ORIGIN_YZ_SEARCH,
        "TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM": TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM,
        "TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM": TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM,
        "TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM": TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM,
        "TARGET_ORIGIN_YZ_SEARCH_MAX_ITERS": TARGET_ORIGIN_YZ_SEARCH_MAX_ITERS,
        "TARGET_ORIGIN_YZ_SEARCH_BEAM_WIDTH": TARGET_ORIGIN_YZ_SEARCH_BEAM_WIDTH,
        "TARGET_ORIGIN_YZ_SEARCH_USE_SERVER": TARGET_ORIGIN_YZ_SEARCH_USE_SERVER,
        "TARGET_ORIGIN_YZ_SEARCH_POLISH_STEP_MM": TARGET_ORIGIN_YZ_SEARCH_POLISH_STEP_MM,
        "TARGET_ORIGIN_YZ_SEARCH_USABLE_COUNT": TARGET_ORIGIN_YZ_SEARCH_USABLE_COUNT,
        "TARGET_ORIGIN_YZ_SEARCH_POST_DISPATCH": TARGET_ORIGIN_YZ_SEARCH_POST_DISPATCH,
        "TARGET_ORIGIN_YZ_SEARCH_POST_TOP_N": TARGET_ORIGIN_YZ_SEARCH_POST_TOP_N,
        "DEBUG_FALLBACK_IMPORT_COUNT": DEBUG_FALLBACK_IMPORT_COUNT,
        "ALLOW_INVALID_DEBUG_OUTPUTS": ALLOW_INVALID_DEBUG_OUTPUTS,
        "TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_COUNT": TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_COUNT,
        "TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_MAX_RINGS": TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_MAX_RINGS,
        "TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_RING_STEP_MM": TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_RING_STEP_MM,
        "TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_EDGE_STEP_MM": TARGET_ORIGIN_YZ_OUTSIDE_FALLBACK_EDGE_STEP_MM,
    }


def _apply_main_overrides() -> None:
    _entry.apply_overrides(_override_map())
    globals()["APP_RUNTIME_SETTINGS"] = _entry.APP_RUNTIME_SETTINGS


# Apply once on import so legacy scripts using `from main import ...` keep working.
_apply_main_overrides()


def __getattr__(name: str) -> Any:
    return getattr(_entry, name)


def main() -> int:
    _apply_main_overrides()
    return _entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
