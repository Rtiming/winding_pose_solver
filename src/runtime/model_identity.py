from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _rounded_rows(value: Any, shape: tuple[int, ...]) -> list[Any]:
    array = np.asarray(value, dtype=float).reshape(shape)
    rounded = np.round(array, decimals=9)
    rounded[np.isclose(rounded, 0.0, atol=1e-12)] = 0.0
    return rounded.tolist()


def kinematics_hash(
    *,
    axes: Any,
    points: Any,
    senses: Any,
    home_flange: Any,
) -> str:
    """Return a stable identity hash for a six-axis kinematic model.

    The hash is intentionally based only on geometry/sense fields that affect
    FK/IK identity. Tool, frame, current joint value, and display assets are not
    part of this identity.
    """

    payload = {
        "schema": "winding_pose_solver.kinematics.v1",
        "joint_axes_base": _rounded_rows(axes, (6, 3)),
        "joint_points_base_mm": _rounded_rows(points, (6, 3)),
        "joint_senses": _rounded_rows(senses, (6,)),
        "home_flange": _rounded_rows(home_flange, (4, 4)),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
