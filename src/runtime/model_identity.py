from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def kinematics_hash(
    *,
    axes: Any,
    points: Any,
    senses: Any,
    home_flange: Any,
) -> str:
    payload = {
        "joint_axes_base": np.asarray(axes, dtype=float).round(9).tolist(),
        "joint_points_base_mm": np.asarray(points, dtype=float).round(6).tolist(),
        "joint_senses": np.asarray(senses, dtype=float).round(9).tolist(),
        "home_flange": np.asarray(home_flange, dtype=float).round(6).tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
