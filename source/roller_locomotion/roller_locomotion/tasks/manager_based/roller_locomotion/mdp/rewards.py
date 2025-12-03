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
    foot_body_names: list = ["left_foot", "right_foot"],
    target_height: float = 0.1,
    std: float = 0.05
) -> torch.Tensor:
    """奖励足部离地高度，鼓励抬腿动作

    使用高斯核函数，当足部高度接近目标高度时给予最大奖励

    TODO: 使用轮子平均高度，避免踮脚
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取足部高度（相对于地面）
    foot_heights = []
    for foot_name in foot_body_names:
        foot_pos = asset.data.body_pos_w[:, asset.body_names.index(foot_name), 2]
        foot_heights.append(foot_pos)

    foot_heights = torch.stack(foot_heights, dim=1)  # [num_envs, num_feet]

    # 使用高斯奖励，鼓励足部达到目标高度
    reward = torch.exp(-torch.square(foot_heights - target_height) / (2 * std**2))

    return torch.sum(reward, dim=1)


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
