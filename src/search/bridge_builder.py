from __future__ import annotations

from typing import Sequence

from src.core.geometry import (
    _pose_row_from_rotation_translation,
    _pose_row_to_rotation_translation,
    _quaternion_to_rotation_matrix,
    _rotation_matrix_to_quaternion,
    _slerp_quaternion,
)


def _insert_interpolated_transition_rows(
    reference_pose_rows: Sequence[dict[str, float]],
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
    row_labels: Sequence[str],
    inserted_flags: Sequence[bool],
    *,
    segment_index: int,
    insertion_count: int,
) -> tuple[
    tuple[dict[str, float], ...],
    tuple[tuple[float, float], ...],
    tuple[str, ...],
    tuple[bool, ...],
]:
    """Insert process-frame transition samples between two neighboring rows.

    The inserted rows are reconstructed from interpolated reference poses, not
    from joint interpolation. Their Frame-2 Y/Z profile values are linearly
    interpolated from the already solved neighboring profile samples.
    """

    if insertion_count <= 0:
        return (
            tuple(dict(row) for row in reference_pose_rows),
            tuple((float(dy), float(dz)) for dy, dz in frame_a_origin_yz_profile_mm),
            tuple(str(label) for label in row_labels),
            tuple(bool(flag) for flag in inserted_flags),
        )

    if segment_index < 0 or segment_index + 1 >= len(reference_pose_rows):
        raise IndexError(f"Invalid insertion segment index: {segment_index}")

    previous_row = reference_pose_rows[segment_index]
    current_row = reference_pose_rows[segment_index + 1]
    previous_rotation, previous_translation = _pose_row_to_rotation_translation(previous_row)
    current_rotation, current_translation = _pose_row_to_rotation_translation(current_row)
    previous_quaternion = _rotation_matrix_to_quaternion(previous_rotation)
    current_quaternion = _rotation_matrix_to_quaternion(current_rotation)
    previous_profile = frame_a_origin_yz_profile_mm[segment_index]
    current_profile = frame_a_origin_yz_profile_mm[segment_index + 1]

    augmented_rows: list[dict[str, float]] = []
    augmented_profile: list[tuple[float, float]] = []
    augmented_labels: list[str] = []
    augmented_flags: list[bool] = []

    for row_index, row in enumerate(reference_pose_rows):
        augmented_rows.append(dict(row))
        augmented_profile.append(
            (
                float(frame_a_origin_yz_profile_mm[row_index][0]),
                float(frame_a_origin_yz_profile_mm[row_index][1]),
            )
        )
        augmented_labels.append(str(row_labels[row_index]))
        augmented_flags.append(bool(inserted_flags[row_index]))

        if row_index != segment_index:
            continue

        for insertion_index in range(1, insertion_count + 1):
            interpolation_ratio = insertion_index / (insertion_count + 1)
            interpolated_rotation = _quaternion_to_rotation_matrix(
                _slerp_quaternion(
                    previous_quaternion,
                    current_quaternion,
                    interpolation_ratio,
                )
            )
            interpolated_translation = tuple(
                float(previous_value + (current_value - previous_value) * interpolation_ratio)
                for previous_value, current_value in zip(
                    previous_translation,
                    current_translation,
                )
            )
            interpolated_profile = tuple(
                float(previous_value + (current_value - previous_value) * interpolation_ratio)
                for previous_value, current_value in zip(previous_profile, current_profile)
            )
            augmented_rows.append(
                _pose_row_from_rotation_translation(
                    interpolated_rotation,
                    interpolated_translation,
                )
            )
            augmented_profile.append(interpolated_profile)
            augmented_labels.append(f"{row_labels[row_index]}_INS_{insertion_index:02d}")
            augmented_flags.append(True)

    return (
        tuple(augmented_rows),
        tuple(augmented_profile),
        tuple(augmented_labels),
        tuple(augmented_flags),
    )
