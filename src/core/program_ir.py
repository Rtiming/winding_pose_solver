from __future__ import annotations

from typing import Any


def program_runtime_capabilities() -> dict[str, Any]:
    """Return solver-owned reserved program/runtime capability metadata.

    The runtime endpoints are intentionally reserved here instead of being
    implemented in adapter packages. This keeps KUKA/RoboDK export and future
    simulation execution under the solver/runtime ownership boundary.
    """

    return {
        "owner": "winding_pose_solver",
        "status": "reserved",
        "simulation_control": {
            "implemented": False,
            "reserved_commands": [
                "play",
                "pause",
                "stop",
                "seek",
                "step",
                "set_speed",
                "run_from_instruction",
            ],
        },
        "program_plan": {
            "implemented": False,
            "instruction_ir_reserved": True,
        },
        "program_export": {
            "implemented": False,
            "targets": [
                "robodk_station",
                "kuka_krl_src_dat",
                "kuka_mxautomation_plan",
                "kuka_srci_plan",
            ],
            "first_concrete_target": "kuka_krl_src_dat",
            "controller_family": "kuka_kr_c5",
        },
        "collision_policy": {
            "implemented": False,
            "owner": "winding_pose_solver",
        },
    }
