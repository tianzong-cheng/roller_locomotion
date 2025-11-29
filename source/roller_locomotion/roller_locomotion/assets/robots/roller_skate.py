from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

ROLLER_SKATE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/robots/roller_skating_robot/roller_skating_robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit={".*": 20.0},
            stiffness={
                ".*_hip.*": 10.0,
                ".*_knee_joint": 10.0,
                ".*_ankle.*": 10.0,
                ".*_wheel.*": 0.0,
            },
            damping={
                ".*_hip.*": 1.0,
                ".*_knee_joint": 1.0,
                ".*_ankle.*": 1.0,
                ".*_wheel.*": 0.0,
            },
            velocity_limit_sim={".*": 100.0},
        ),
    },
)
"""Configuration for the Roller Skate robot."""
