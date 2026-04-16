from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence


def _build_pose(row: dict[str, float], mat_type) -> object:
    """把一行 CSV 位姿组装成 RoboDK 使用的 4x4 齐次矩阵。"""

    return mat_type(
        [
            [row["r11"], row["r12"], row["r13"], row["x_mm"]],
            [row["r21"], row["r22"], row["r23"], row["y_mm"]],
            [row["r31"], row["r32"], row["r33"], row["z_mm"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _pose_row_to_rotation_translation(
    pose_row: dict[str, float],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    """把 CSV row 解析成 3x3 旋转矩阵和 3x1 平移向量。"""

    rotation = (
        (float(pose_row["r11"]), float(pose_row["r12"]), float(pose_row["r13"])),
        (float(pose_row["r21"]), float(pose_row["r22"]), float(pose_row["r23"])),
        (float(pose_row["r31"]), float(pose_row["r32"]), float(pose_row["r33"])),
    )
    translation = (
        float(pose_row["x_mm"]),
        float(pose_row["y_mm"]),
        float(pose_row["z_mm"]),
    )
    return rotation, translation


def _pose_row_from_rotation_translation(
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> dict[str, float]:
    """把旋转矩阵和平移向量重新打包成 pose row。"""

    return {
        "x_mm": float(translation[0]),
        "y_mm": float(translation[1]),
        "z_mm": float(translation[2]),
        "r11": float(rotation[0][0]),
        "r12": float(rotation[0][1]),
        "r13": float(rotation[0][2]),
        "r21": float(rotation[1][0]),
        "r22": float(rotation[1][1]),
        "r23": float(rotation[1][2]),
        "r31": float(rotation[2][0]),
        "r32": float(rotation[2][1]),
        "r33": float(rotation[2][2]),
    }


def _rotation_matrix_from_xyz_offset_deg(
    rotation_offset_deg: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """构造基于 Frame 2 固定轴的 XYZ 旋转矩阵。"""

    normalized_offset = (
        float(rotation_offset_deg[0]),
        float(rotation_offset_deg[1]),
        float(rotation_offset_deg[2]),
    )
    return _rotation_matrix_from_xyz_offset_deg_cached(normalized_offset)


@lru_cache(maxsize=512)
def _rotation_matrix_from_xyz_offset_deg_cached(
    rotation_offset_deg: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """缓存常用姿态偏置旋转矩阵，避免在搜索中重复构造。"""

    rx_deg, ry_deg, rz_deg = rotation_offset_deg
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    rotation_x = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(rx), -math.sin(rx)),
        (0.0, math.sin(rx), math.cos(rx)),
    )
    rotation_y = (
        (math.cos(ry), 0.0, math.sin(ry)),
        (0.0, 1.0, 0.0),
        (-math.sin(ry), 0.0, math.cos(ry)),
    )
    rotation_z = (
        (math.cos(rz), -math.sin(rz), 0.0),
        (math.sin(rz), math.cos(rz), 0.0),
        (0.0, 0.0, 1.0),
    )
    return _multiply_rotation_matrices(
        rotation_z,
        _multiply_rotation_matrices(rotation_y, rotation_x),
    )


def _multiply_rotation_matrices(
    first_rotation: Sequence[Sequence[float]],
    second_rotation: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """3x3 旋转矩阵相乘。"""

    return tuple(
        tuple(
            float(
                sum(
                    first_rotation[row_index][inner_index]
                    * second_rotation[inner_index][column_index]
                    for inner_index in range(3)
                )
            )
            for column_index in range(3)
        )
        for row_index in range(3)
    )


def _multiply_rotation_vector(
    rotation: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    """3x3 旋转矩阵乘三维向量。"""

    return tuple(
        float(
            sum(rotation[row_index][column_index] * vector[column_index] for column_index in range(3))
        )
        for row_index in range(3)
    )


def _rotation_matrix_to_quaternion(
    rotation: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """把 3x3 旋转矩阵转成单位四元数。"""

    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace_value = m00 + m11 + m22

    if trace_value > 0.0:
        scale = math.sqrt(trace_value + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    return _normalize_quaternion((qw, qx, qy, qz))


def _quaternion_to_rotation_matrix(
    quaternion: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """把单位四元数转成 3x3 旋转矩阵。"""

    qw, qx, qy, qz = _normalize_quaternion(quaternion)
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def _slerp_quaternion(
    first_quaternion: Sequence[float],
    second_quaternion: Sequence[float],
    interpolation_ratio: float,
) -> tuple[float, float, float, float]:
    """对两个单位四元数做球面插值。"""

    first = _normalize_quaternion(first_quaternion)
    second = _normalize_quaternion(second_quaternion)
    dot_product = sum(a * b for a, b in zip(first, second))

    if dot_product < 0.0:
        second = tuple(-value for value in second)
        dot_product = -dot_product

    if dot_product > 0.9995:
        blended = tuple(
            (1.0 - interpolation_ratio) * first_value + interpolation_ratio * second_value
            for first_value, second_value in zip(first, second)
        )
        return _normalize_quaternion(blended)

    theta_0 = math.acos(max(-1.0, min(1.0, dot_product)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * interpolation_ratio
    sin_theta = math.sin(theta)
    scale_first = math.sin(theta_0 - theta) / sin_theta_0
    scale_second = sin_theta / sin_theta_0
    return _normalize_quaternion(
        tuple(
            scale_first * first_value + scale_second * second_value
            for first_value, second_value in zip(first, second)
        )
    )


def _normalize_quaternion(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    """把四元数归一化到单位长度。"""

    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quaternion)


def _rotation_from_pose(pose_matrix) -> tuple[tuple[float, float, float], ...]:
    """从 4x4 位姿中提取 3x3 旋转矩阵。"""

    return tuple(
        tuple(float(pose_matrix[row, column]) for column in range(3))
        for row in range(3)
    )


def _build_pose_from_rotation_translation(reference_pose, rotation, translation):
    """根据旋转和平移重建一个 RoboDK 位姿。"""

    pose_type = type(reference_pose)
    return pose_type(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2], float(translation[0])],
            [rotation[1][0], rotation[1][1], rotation[1][2], float(translation[1])],
            [rotation[2][0], rotation[2][1], rotation[2][2], float(translation[2])],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _translation_from_pose(pose_matrix) -> tuple[float, float, float]:
    """从 4x4 位姿矩阵中提取平移部分。"""

    return (float(pose_matrix[0, 3]), float(pose_matrix[1, 3]), float(pose_matrix[2, 3]))


def _add_vectors(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> tuple[float, float, float]:
    """三维向量相加。"""

    return (
        first_vector[0] + second_vector[0],
        first_vector[1] + second_vector[1],
        first_vector[2] + second_vector[2],
    )


def _subtract_vectors(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> tuple[float, float, float]:
    """三维向量相减。"""

    return (
        first_vector[0] - second_vector[0],
        first_vector[1] - second_vector[1],
        first_vector[2] - second_vector[2],
    )


def _normalized_cross_measure(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> float:
    """返回两个向量叉积模长的归一化值。

    值越接近 0，说明越接近共线。
    """

    first_norm = math.sqrt(sum(component * component for component in first_vector))
    second_norm = math.sqrt(sum(component * component for component in second_vector))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return 1.0

    cross_x = first_vector[1] * second_vector[2] - first_vector[2] * second_vector[1]
    cross_y = first_vector[2] * second_vector[0] - first_vector[0] * second_vector[2]
    cross_z = first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
    cross_norm = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    return cross_norm / (first_norm * second_norm)


def _pose_translation_distance_mm(first_pose, second_pose) -> float:
    """计算两个位姿之间的平移偏差。"""

    delta_x = float(first_pose[0, 3] - second_pose[0, 3])
    delta_y = float(first_pose[1, 3] - second_pose[1, 3])
    delta_z = float(first_pose[2, 3] - second_pose[2, 3])
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


def _pose_rotation_distance_deg(first_pose, second_pose) -> float:
    """计算两个位姿之间的旋转偏差角。"""

    rotation_error = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            rotation_error[row][column] = sum(
                float(first_pose[k, row]) * float(second_pose[k, column])
                for k in range(3)
            )

    trace_value = rotation_error[0][0] + rotation_error[1][1] + rotation_error[2][2]
    cosine_value = max(-1.0, min(1.0, (trace_value - 1.0) * 0.5))
    return math.degrees(math.acos(cosine_value))


def _interpolate_pose_with_locked_translation(
    *,
    previous_pose,
    current_pose,
    interpolation_ratio: float,
    locked_translation: Sequence[float],
):
    """在固定平移的前提下，对两个姿态做球面插值。"""

    previous_quaternion = _rotation_matrix_to_quaternion(_rotation_from_pose(previous_pose))
    current_quaternion = _rotation_matrix_to_quaternion(_rotation_from_pose(current_pose))
    interpolated_quaternion = _slerp_quaternion(
        previous_quaternion,
        current_quaternion,
        interpolation_ratio,
    )
    interpolated_rotation = _quaternion_to_rotation_matrix(interpolated_quaternion)
    return _build_pose_from_rotation_translation(
        previous_pose,
        interpolated_rotation,
        locked_translation,
    )


def _interpolate_joints(
    start_joints: Sequence[float],
    end_joints: Sequence[float],
    ratio: float,
) -> tuple[float, ...]:
    """在两组关节之间做线性插值。"""

    return tuple(
        start + (end - start) * ratio
        for start, end in zip(start_joints, end_joints)
    )


def _mean_abs_joint_delta(
    first_joints: Sequence[float],
    second_joints: Sequence[float],
) -> float:
    """计算两组关节之间的平均绝对差值。"""

    deltas = [abs(second - first) for first, second in zip(first_joints, second_joints)]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _normalize_angle_range(first_deg: float, second_deg: float) -> tuple[float, float]:
    """把角度范围归一化为 [lower, upper]。"""

    return (min(first_deg, second_deg), max(first_deg, second_deg))


def _normalize_step_limits(step_limits: Sequence[float], joint_count: int) -> tuple[float, ...]:
    """把连续性阈值补齐/裁剪到机器人实际轴数长度。"""

    if not step_limits:
        raise ValueError("Joint continuity step limits must not be empty.")

    normalized = [float(value) for value in step_limits]
    if len(normalized) >= joint_count:
        return tuple(normalized[:joint_count])

    # 如果提供的阈值数量少于轴数，则用最后一个阈值补齐剩余关节。
    normalized.extend([normalized[-1]] * (joint_count - len(normalized)))
    return tuple(normalized)


def _extract_joint_tuple(raw_joints, joint_count: int) -> tuple[float, ...]:
    """把 RoboDK 返回的 Mat / list 统一提取成固定长度关节元组。"""

    if hasattr(raw_joints, "list"):
        values = list(raw_joints.list())
    else:
        values = list(raw_joints)

    if not values:
        return ()

    trimmed = _trim_joint_vector(values, joint_count)
    if len(trimmed) != joint_count or not all(math.isfinite(value) for value in trimmed):
        return ()
    return trimmed


def _trim_joint_vector(values: Sequence[float], joint_count: int) -> tuple[float, ...]:
    """把关节向量裁成机器人实际轴数长度。"""

    return tuple(float(values[index]) for index in range(min(joint_count, len(values))))


def _clip_seed_to_limits(
    seed: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[float, ...]:
    """把 seed 限制在机器人关节硬限位内。"""

    return tuple(
        min(max(float(value), lower), upper)
        for value, lower, upper in zip(seed, lower_limits, upper_limits)
    )


def _is_within_joint_limits(
    joints: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """检查是否落在机器人自身的关节限位内。"""

    return all(
        lower - tolerance <= joint <= upper + tolerance
        for joint, lower, upper in zip(joints, lower_limits, upper_limits)
    )


def _passes_user_joint_constraints(
    joints: Sequence[float],
    *,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    tolerance_deg: float,
) -> bool:
    """检查是否满足用户指定的 A1 / A2 硬约束。"""

    if len(joints) < 2:
        return False

    a1_value = joints[0]
    a2_value = joints[1]

    if a1_value < a1_lower_deg - tolerance_deg or a1_value > a1_upper_deg + tolerance_deg:
        return False

    # 用户要求 A2 小于 115。这里允许极小容差，避免 114.999999 / 115.0000001 这类数值噪声。
    if a2_value >= a2_max_deg + tolerance_deg:
        return False

    return True
