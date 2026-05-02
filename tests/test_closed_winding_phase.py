from __future__ import annotations

import unittest
import importlib.util
import sys
import types

if importlib.util.find_spec("pandas") is None:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    sys.modules["pandas"] = pandas_stub

from src.core.types import _IKCandidate, _IKLayer, _PathOptimizerSettings
from src.search.path_optimizer import _optimize_joint_path, _summarize_selected_path


def _candidate(a6_deg: float) -> _IKCandidate:
    return _IKCandidate(
        joints=(0.0, 0.0, 0.0, 0.0, 30.0, float(a6_deg)),
        config_flags=(0, 1, 0),
        joint_limit_penalty=0.0,
        singularity_penalty=0.0,
        branch_id=None,
    )


def _layer(a6_values: tuple[float, ...]) -> _IKLayer:
    return _IKLayer(
        pose=object(),
        candidates=tuple(_candidate(a6_deg) for a6_deg in a6_values),
    )


def _settings() -> _PathOptimizerSettings:
    return _PathOptimizerSettings(
        joint_delta_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        max_joint_step_deg=(5.0, 5.0, 5.0, 45.0, 30.0, 130.0),
        preferred_joint_step_deg=(5.0, 5.0, 5.0, 25.0, 25.0, 130.0),
        closed_path_joint6_direction_sample_count=4,
        closed_path_joint6_direction_min_delta_deg=1.0,
    )


class ClosedWindingJoint6PhaseTests(unittest.TestCase):
    def test_closed_path_redistributes_joint6_phase_across_equivalent_rows(self) -> None:
        selected_path, _total_cost = _optimize_joint_path(
            (
                _layer((0.0,)),
                _layer((120.0,)),
                _layer((-120.0,)),
                _layer((0.0,)),
            ),
            robot=None,
            move_type="MoveJ",
            start_joints=(0.0, 0.0, 0.0, 0.0, 30.0, 0.0),
            optimizer_settings=_settings(),
            require_terminal_match_start=True,
            selection_bridge_trigger_joint_delta_deg=170.0,
        )

        self.assertEqual([entry.joints[5] for entry in selected_path], [0.0, 120.0, 240.0, 360.0])
        self.assertAlmostEqual(selected_path[-1].joints[5] - selected_path[0].joints[5], 360.0)
        _config_switches, bridge_like_segments, worst_joint_step_deg, _mean = _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=170.0,
        )
        self.assertEqual(bridge_like_segments, 0)
        self.assertLessEqual(worst_joint_step_deg, 130.0)


if __name__ == "__main__":
    unittest.main()
