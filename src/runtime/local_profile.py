from __future__ import annotations

import os
import platform
from pathlib import Path


LOCAL_PROFILE_CHOICES = ("auto", "mac", "windows", "linux")
LOCAL_PROFILE_ALIASES = {
    "darwin": "mac",
    "macos": "mac",
    "osx": "mac",
    "win": "windows",
    "windows_robodk": "windows",
    "mac_robodk": "mac",
    "mac_server": "mac",
    "server": "linux",
}
DEFAULT_LOCAL_PROFILE_ENV = "WPS_LOCAL_MACHINE_PROFILE"


def normalize_local_profile(
    raw_value: str | None,
    *,
    option_name: str = "local profile",
) -> str:
    normalized = str(raw_value or "auto").strip().lower()
    normalized = LOCAL_PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in LOCAL_PROFILE_CHOICES:
        expected = "/".join(LOCAL_PROFILE_CHOICES)
        raise ValueError(f"Unsupported {option_name}={raw_value!r}; expected {expected}.")
    return normalized


def detect_local_profile() -> str:
    system_name = platform.system().strip().lower()
    if system_name == "darwin":
        return "mac"
    if system_name == "windows":
        return "windows"
    if system_name == "linux":
        return "linux"
    return system_name or "unknown"


def resolve_local_profile(
    configured_profile: str | None,
    *,
    env_name: str = DEFAULT_LOCAL_PROFILE_ENV,
) -> str:
    configured = normalize_local_profile(
        configured_profile if configured_profile else os.getenv(env_name, "auto"),
    )
    return detect_local_profile() if configured == "auto" else configured


def local_profile_metadata(
    configured_profile: str | None,
    *,
    env_name: str = DEFAULT_LOCAL_PROFILE_ENV,
) -> dict[str, object]:
    configured = (
        str(configured_profile)
        if configured_profile
        else str(os.getenv(env_name, "auto"))
    )
    return {
        "local_machine_profile": resolve_local_profile(configured_profile, env_name=env_name),
        "local_machine_profile_config": configured,
        "local_platform_system": platform.system(),
        "local_platform_machine": platform.machine(),
    }


def local_profile_status_text(
    configured_profile: str | None,
    *,
    env_name: str = DEFAULT_LOCAL_PROFILE_ENV,
) -> str:
    resolved = resolve_local_profile(configured_profile, env_name=env_name)
    return (
        f"{resolved} "
        f"(system={platform.system()}, machine={platform.machine()})"
    )


def local_conda_python_candidates(
    local_profile: str,
    *,
    env_name: str = "winding_pose_solver",
) -> tuple[Path, ...]:
    resolved_profile = resolve_local_profile(local_profile)
    roots = (
        Path.home() / "miniforge3",
        Path.home() / "mambaforge",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    )
    executable = "python.exe" if resolved_profile == "windows" else "python"
    subdir = "" if resolved_profile == "windows" else "bin"
    candidates: list[Path] = []
    for root in roots:
        env_dir = root / "envs" / env_name
        candidates.append(
            env_dir / executable if not subdir else env_dir / subdir / executable
        )
    return tuple(candidates)
