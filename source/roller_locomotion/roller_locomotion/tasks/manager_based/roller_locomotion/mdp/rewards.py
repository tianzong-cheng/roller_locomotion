# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import math


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def two_joint_min_abduction_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_joint_name: str = "left_hip_roll_joint",
    right_joint_name: str = "right_hip_roll_joint",
) -> torch.Tensor:
    """take the min abduction of two joints as the l2 penalty"""

    asset: Articulation = env.scene[asset_cfg.name]

    hip_roll_left_idx = asset.find_joints(left_joint_name)[0][0]
    hip_roll_right_idx = asset.find_joints(right_joint_name)[0][0]

    joint_pos = asset.data.joint_pos

    left_abduction = torch.abs(joint_pos[:, hip_roll_left_idx])
    right_abduction = torch.abs(joint_pos[:, hip_roll_right_idx])

    min_abduction = torch.min(left_abduction, right_abduction)

    return torch.square(min_abduction)


def foot_clearance_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_wheel_body_names: list = ["left_wheel_1", "left_wheel_4"],
    right_wheel_body_names: list = ["right_wheel_1", "right_wheel_4"],
    target_height: float = 0.1,
    std: float = 0.05
) -> torch.Tensor:
    """奖励轮子离地高度，鼓励抬腿动作

    使用高斯核函数，基于每只脚最低的轮子高度来计算奖励
    只有当所有轮子都抬起时才能获得高奖励，避免只抬起部分轮子

    Args:
        env: 环境实例
        asset_cfg: 场景实体配置
        left_wheel_body_names: 左脚轮子的body名称列表
        right_wheel_body_names: 右脚轮子的body名称列表
        target_height: 目标离地高度(米)
        std: 高斯核标准差

    Returns:
        每个环境的奖励值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取左脚所有轮子的高度，取最小值
    left_wheel_heights = []
    for wheel_name in left_wheel_body_names:
        wheel_idx = asset.body_names.index(wheel_name)
        wheel_height = asset.data.body_pos_w[:, wheel_idx, 2]
        left_wheel_heights.append(wheel_height)

    left_min_height = torch.stack(left_wheel_heights, dim=1).min(dim=1)[0]  # [num_envs]

    # 获取右脚所有轮子的高度，取最小值
    right_wheel_heights = []
    for wheel_name in right_wheel_body_names:
        wheel_idx = asset.body_names.index(wheel_name)
        wheel_height = asset.data.body_pos_w[:, wheel_idx, 2]
        right_wheel_heights.append(wheel_height)

    right_min_height = torch.stack(right_wheel_heights, dim=1).min(dim=1)[0]  # [num_envs]

    # 使用高斯奖励，分别对左右脚的最低轮子高度进行奖励
    left_reward = torch.exp(-torch.square(left_min_height - target_height) / (2 * std**2))
    right_reward = torch.exp(-torch.square(right_min_height - target_height) / (2 * std**2))

    # 返回左右脚奖励之和
    return left_reward + right_reward


def alternating_leg_lift(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: list = ["left_foot", "right_foot"],
    velocity_threshold: float = 0.3
) -> torch.Tensor:
    """奖励交替抬腿动作，模拟人类行走/滑行的腿部运动"""
    # TODO: validation and threshold tuning

    asset: Articulation = env.scene[asset_cfg.name]

    # 获取左右脚的垂直速度
    left_foot_vel = asset.data.body_vel_w[:, asset.body_names.index(foot_body_names[0]), 2]
    right_foot_vel = asset.data.body_vel_w[:, asset.body_names.index(foot_body_names[1]), 2]

    # 检测一只脚抬起（正速度）而另一只脚放下（负速度或静止）
    # 使用速度的乘积：当一个为正一个为负时，乘积为负，取绝对值
    alternating_motion = torch.abs(left_foot_vel * right_foot_vel)

    # 只有当至少一只脚有足够的垂直速度时才奖励
    has_motion = (torch.abs(left_foot_vel) > velocity_threshold) | \
                 (torch.abs(right_foot_vel) > velocity_threshold)

    reward = alternating_motion * has_motion.float()

    return reward


def heading_deviation_l2(
    env: ManagerBasedRLEnv,
    threshold_deg: float = 30.0,
    clamp: float = 1.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    当偏离目标heading超过阈值时惩罚

    Args:
        threshold_deg: 角度阈值（度），只有超过此阈值才惩罚
        clamp: 最大惩罚值
        command_name: 速度指令名称（包含heading信息）

    Returns:
        torch.Tensor: 惩罚值 [num_envs]，30度内为0，超过30度则为平方惩罚
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取当前朝向（yaw角）
    quat = asset.data.root_quat_w  # [num_envs, 4] - [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # 从四元数提取yaw角（绕Z轴旋转）
    current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # 获取目标heading（从速度指令中获取）
    # base_velocity指令的格式通常是 [vx, vy, omega, heading] 或类似
    command = env.command_manager.get_command(command_name)

    # 目标heading通常在最后一维，具体取决于UniformVelocityCommand的实现
    # 常见情况是 command[:, -1] 或 command[:, 3]
    target_heading = command[:, -1]  # 尝试最后一维
    # 如果不对，可以尝试: target_heading = command[:, 3]

    # 计算角度差异（考虑角度环绕 -π 到 π）
    heading_error = angle_difference(current_yaw, target_heading)
    heading_error_abs = torch.abs(heading_error)

    # 转换阈值为弧度
    threshold_rad = math.radians(threshold_deg)

    # 只惩罚超过阈值的部分：max(0, |error| - threshold)
    excess_error = torch.relu(heading_error_abs - threshold_rad)

    # 对超出部分进行平方惩罚
    penalty = torch.square(excess_error)
    penalty = torch.clamp(penalty, max=clamp)

    return penalty


def angle_difference(angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
    """
    计算两个角度之间的最小差异（考虑角度环绕）

    返回值范围: [-π, π]
    """
    diff = angle1 - angle2
    # 将差异归一化到 [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff


def check_wheel_rotation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_joint_names: list = ["left_wheel_joint_4", "right_wheel_joint_4"],
) -> torch.Tensor:
    """检查轮子关节是否在转动，用于调试"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取轮子关节索引
    wheel_indices = [asset.joint_names.index(name) for name in wheel_joint_names]

    # 获取轮子的关节速度（rad/s）
    wheel_velocities = asset.data.joint_vel[:, wheel_indices]

    # 获取轮子的关节位置（rad）
    wheel_positions = asset.data.joint_pos[:, wheel_indices]

    # 打印信息（仅在第一个环境）
    if env.episode_length_buf[0] % 100 == 0:  # 每100步打印一次
        print(f"Wheel velocities: {wheel_velocities[0]}")
        print(f"Wheel positions: {wheel_positions[0]}")
        print(f"Wheel speed (rad/s): {torch.abs(wheel_velocities[0]).mean()}")

    return torch.abs(wheel_velocities).mean(dim=1)


def wheel_lateral_drag_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_body_names: list = ["left_wheel_1", "right_wheel_1", "left_wheel_4", "right_wheel_4"],
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["left_wheel_1", "right_wheel_1", "left_wheel_4", "right_wheel_4"]),
    contact_force_threshold: float = 1.0,  # 接触力阈值（N）
    lateral_velocity_threshold: float = 0.01,  # 允许的最小侧向速度（m/s）
    debug: bool = False,
) -> torch.Tensor:
    """惩罚轮子与地面接触时的侧向拖动（lateral drag）

    当轮子接触地面时，惩罚其轴向（侧向）速度，鼓励轮子只沿滚动方向移动

    Args:
        wheel_body_names: 轮子body名称列表
        contact_sensor_cfg: 接触力传感器配置
        contact_force_threshold: 判定为接触的力阈值
        lateral_velocity_threshold: 低于此值不惩罚（允许小的侧滑）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取接触力
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w

    total_penalty = torch.zeros(env.num_envs, device=env.device)

    for i, wheel_name in enumerate(wheel_body_names):
        # 获取轮子body索引
        wheel_body_idx = asset.body_names.index(wheel_name)

        # 获取轮子的世界坐标系速度
        wheel_vel_w = asset.data.body_vel_w[:, wheel_body_idx, :3]  # [num_envs, 3]

        # 获取轮子的方向（假设轮子轴向是y轴，滚动方向是x轴）
        # 需要根据你的机器人坐标系调整
        wheel_quat = asset.data.body_quat_w[:, wheel_body_idx, :]

        # 将速度转换到轮子局部坐标系
        from isaaclab.utils.math import quat_apply_inverse
        wheel_vel_local = quat_apply_inverse(wheel_quat, wheel_vel_w)

        # 轮子的侧向速度（假设y轴是轮子轴向）
        # 如果你的轮子轴向是x轴，改为 wheel_vel_local[:, 0]
        lateral_velocity = torch.abs(wheel_vel_local[:, 1])

        # 检测接触
        contact_force = torch.norm(net_contact_forces[:, wheel_body_idx, :], dim=-1)
        is_in_contact = contact_force > contact_force_threshold

        # 只在接触时惩罚侧向速度
        # 使用平方惩罚，使大的侧滑受到更重的惩罚
        lateral_drag = torch.where(
            is_in_contact & (lateral_velocity > lateral_velocity_threshold),
            torch.square(lateral_velocity),
            torch.zeros_like(lateral_velocity)
        )

        total_penalty += lateral_drag

        # debug info
        if debug and lateral_drag[0] > 0:
            print(f"Wheel {wheel_name} lateral drag penalty: {lateral_drag[0]:.4f} (Lat Vel: {lateral_velocity[0]:.4f} m/s, Contact Force: {contact_force[0]:.2f} N)")

    return total_penalty
