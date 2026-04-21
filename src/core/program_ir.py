from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProgramInstructionKind = Literal[
    "move_j",
    "move_l",
    "set_frame",
    "set_tool",
    "set_speed",
    "set_rounding",
    "set_do",
    "wait_di",
    "pause",
    "call_program",
    "message",
    "custom_code",
]

SimulationCommand = Literal[
    "play",
    "pause",
    "stop",
    "seek",
    "step",
    "set_speed",
    "run_from_instruction",
]

ProgramExportTarget = Literal[
    "robodk_station",
    "kuka_krl_src_dat",
    "kuka_mxautomation_plan",
    "kuka_srci_plan",
]


@dataclass(frozen=True)
class ProgramInstruction:
    """Controller-neutral instruction record for future RoboDK-parity work.

    This is only an interface contract. Execution, RoboDK station mutation, and
    KUKA export remain owned by runtime/robodk_runtime implementations.
    """

    kind: ProgramInstructionKind
    label: str | None = None
    target_name: str | None = None
    pose_frame: list[list[float]] | None = None
    joints_deg: list[float] | None = None
    speed_mm_s: float | None = None
    speed_deg_s: float | None = None
    accel_mm_s2: float | None = None
    accel_deg_s2: float | None = None
    rounding_mm: float | None = None
    io_name: str | int | None = None
    io_value: str | int | float | bool | None = None
    timeout_ms: int | None = None
    duration_ms: int | None = None
    program_name: str | None = None
    message: str | None = None
    code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationControlRequest:
    command: SimulationCommand
    program_name: str | None = None
    instruction_index: int | None = None
    speed_ratio: float | None = None
    run_on_robot: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgramPlanRequest:
    program_name: str
    instructions: tuple[ProgramInstruction, ...] = ()
    source_result_id: str | None = None
    controller_family: str = "kuka_kr_c5"
    controller_runtime: str = "kss_8_7"
    export_target: ProgramExportTarget = "kuka_krl_src_dat"
    post_processor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgramExportRequest:
    program_name: str
    export_target: ProgramExportTarget = "kuka_krl_src_dat"
    output_dir: str | None = None
    controller_family: str = "kuka_kr_c5"
    controller_runtime: str = "kss_8_7"
    post_processor: str | None = None
    include_dat_file: bool = True
    include_src_file: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def program_runtime_capabilities() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "reserved_interfaces",
        "instruction_ir": {
            "supported_kinds": [
                "move_j",
                "move_l",
                "set_frame",
                "set_tool",
                "set_speed",
                "set_rounding",
                "set_do",
                "wait_di",
                "pause",
                "call_program",
                "message",
                "custom_code",
            ],
        },
        "simulation_control": {
            "status": "reserved_not_implemented",
            "commands": [
                "play",
                "pause",
                "stop",
                "seek",
                "step",
                "set_speed",
                "run_from_instruction",
            ],
        },
        "program_export": {
            "status": "reserved_not_implemented",
            "targets": [
                "robodk_station",
                "kuka_krl_src_dat",
                "kuka_mxautomation_plan",
                "kuka_srci_plan",
            ],
            "default_controller_family": "kuka_kr_c5",
            "default_controller_runtime": "kss_8_7",
        },
    }
