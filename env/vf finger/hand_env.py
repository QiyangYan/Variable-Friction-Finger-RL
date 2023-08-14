# This is the summary of the structure of hand_env.py and fetch_env.py
'''
This is the summary of the structure of hand_env.py and fetch_env.py
"fetch_env" includes
1. _step_callback
2. _set_action
3. generate_mujoco_observations
4. _get_gripper_xpos
5. _render_callback
6. _reset_sim
7. _env_setup

"hand_env" includes
1. _set_action

"MujocoManipulateEnv" includes
1. _get_achieved_goal
2. _env_setup
3. _reset_sim
4. _sample_goal
5. _render_callback
6. _get_obs
'''

# Information regarding _set_action
'''
Information regarding _set_action
Overview: This function gets the action input from high-layer and set the joint based on these

fetch_env: 
1. Get action input, check shape is valid
2. Extract specific info from action, since last element corresponds to two joint
3. Apply modifications to those info_s
4. Return modified action
Note:
    1. Always do a check to the array before use it
    2. Include an end_effector direction quaternion
    3. When multiple joints are set based on single input infor, we can modify based on that
    , for example concatenate, to control those joints.
    
hand_env:
1. Get action input, check shape is valid
2. Get actuator control range from model
3. If relative control mode: create actuation_center, fill it with data read from joint pos
4. If absolute control mode: actuation_center = the half of control range
5. Output is obtained by applying input action to actuation center
6. Add clip to ensure output is within range
Note: 
    1. Actuation center: means the reference point for control, mid position (absolute) or current position (relative)

'''

# This is the action space
'''
Box(-1.0, 1.0, (3,), float32), haven't confirm other parameters

| Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
| --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
| 0   | Angular position of the left finger                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
| 1   | Angular position of the right finger                | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
| 2   | Friction States                                     | -1          | 1           | -1.571 (rad) | 1.571 (rad) | robot0:A_FFJ3 & robot0:A_FFJ4    | hinge | angle (rad) |

Third element decide, LF&HF, or HF&LF, or HF&HF
Friction State          -1          0           1
Left friction servo     L(-90)      H           H
Right friction servo    H           H           L (90)

-- On Dynamixel
For left finger: Low friction is 60 degree (204.8), High friction is 150 degree (512)
For right finger: Low friction is 240 degree (818.4), High friction is 150 degree (512)

-- On Mujoco (Define the 0 degree is the centre, clockwise is negative)
For left finger: Low friction is -90 degree (-1.571 rad), High friction is 0 degree (0 rad)
For right finger: Low friction is 90 degree (1.571 rad), High friction is 0 degree (0 rad)
Note: + 150 degree for Dynamixel Control


CONSIDER FOLLOWING PROBLEMS
1. How you want to control the gripper? Both position control or switching between pos and torque control ✅
2. Is it possible to include both continuous and discrete actions? ✅
3. What parameters should I set for the table? ✅
4. What to do with the mid-layer I implemented before? 
'''

from typing import Union
import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
import math

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.5,
    "azimuth": 55.0,
    "elevation": -25.0,
    "lookat": np.array([1, 0.96, 0.14]),
}


def get_base_hand_env(
    RobotEnvClass: Union[MujocoPyRobotEnv, MujocoRobotEnv]
    # 它表示 RobotEnvClass 参数的类型是 Union[MujocoPyRobotEnv, MujocoRobotEnv]
    # 并且它是一个必需的参数，调用函数时需要传递这个参数的值
) -> Union[MujocoPyRobotEnv, MujocoRobotEnv]:
    """Factory function that returns a BaseHandEnv class that inherits from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings."""

    class BaseHandEnv(RobotEnvClass):
        """Base class for all robotic hand environments."""

        def __init__(self, relative_control, **kwargs):
            self.relative_control = relative_control
            super().__init__(n_actions=3, **kwargs)

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (3,)

    return BaseHandEnv

'''Read me'''
'''
Both fingers follow position control, try this version first.

If this doesn't work, modify this so that 
1. the high friction finger follows position control
2. the low friction finger follows torque control

CONSIDER:
1. What to do with the mid-layer I implemented before? 
'''
class MujocoHandEnv(get_base_hand_env(MujocoRobotEnv)):
    def __init__(
        self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs
    ) -> None:
        super().__init__(default_camera_config=default_camera_config, **kwargs)
        self.last_friction = None
        self.current_friction = None
        self.action_count = 0

    def _set_action(self, action):
        super()._set_action(action)  # check if action has the right shape: 3 dimension

        # Calculate the half of each actuator's control range
        ctrlrange = self.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        gripper_pos_ctrl, friction_state = action[:1], action[2]

        if self.action_count == 0:
            self.current_friction = friction_state
        else:
            self.last_friction = self.current_friction
            self.current_friction = friction_state
        self.action_count += 1

        # Friction control follows this formate: [left right]
        '''
        Third element decide, LF&HF, or HF&LF, or HF&HF
        Friction State          -1          0           1
        Left friction servo     L(-90)      H(0)        H(0)
        Right friction servo    H(0)        H(0)        L(90)

        -- On Dynamixel
        For left finger: Low friction is 60 degree (204.8), High friction is 150 degree (512)
        For right finger: Low friction is 240 degree (818.4), High friction is 150 degree (512)

        -- On Mujoco (Define the 0 degree is the centre, clockwise is negative)
        For left finger: Low friction is -90 degree (-1.571 rad), High friction is 0 degree (0 rad)
        For right finger: Low friction is 90 degree (1.571 rad), High friction is 0 degree (0 rad)
        Note: + 150 degree for Dynamixel Control
        '''
        friction_state_mapping = {
            -1: [-math.pi / 2, 0],
            0: [0, 0],
            1: [0, math.pi / 2]
        }
        friction_ctrl = friction_state_mapping.get(friction_state, None)
        if friction_ctrl is None:
            raise ValueError("Invalid Action with Invalid Friction State")
        assert friction_ctrl.shape == (2,)

        # Obtain actuation center (movement reference point)
        if self.relative_control:
            # Move related to the current position
            actuation_center = np.zeros_like(action)  # To store the reference position for relative control mode
            for i in range(self.data.ctrl.shape[0]):
                actuation_center[i] = self.data.get_joint_qpos(
                    self.model.actuator_names[i].replace(":A_", ":")  # convert actuator name to joint name
                )
            '''Add this if the joints that control small servos are not included'''
            '''
            for joint_name in ["LFF", "RFF"]: 
                act_idx = self.model.actuator_name2id(f"robot0:A_{joint_name}J1")
                actuation_center[act_idx] += self.data.get_joint_qpos(
                    f"robot0:{joint_name}J0"
                )
            '''
        else:
            # Move related to the center
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

        self.data.ctrl = np.concatenate([actuation_center + gripper_pos_ctrl * actuation_range, friction_ctrl])
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


# class MujocoPyHandEnv(get_base_hand_env(MujocoPyRobotEnv)):
#     """Base class for all Hand environments that use mujoco-py as the python bindings."""
#
#     def _set_action(self, action):
#         super()._set_action(action)
#
#         ctrlrange = self.sim.model.actuator_ctrlrange
#         actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
#         if self.relative_control:
#             actuation_center = np.zeros_like(action)
#             for i in range(self.sim.data.ctrl.shape[0]):
#                 actuation_center[i] = self.sim.data.get_joint_qpos(
#                     self.sim.model.actuator_names[i].replace(":A_", ":")
#                 )
#             for joint_name in ["FF", "MF", "RF", "LF"]:
#                 act_idx = self.sim.model.actuator_name2id(f"robot0:A_{joint_name}J1")
#                 actuation_center[act_idx] += self.sim.data.get_joint_qpos(
#                     f"robot0:{joint_name}J0"
#                 )
#         else:
#             actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
#         self.sim.data.ctrl[:] = actuation_center + action * actuation_range
#         self.sim.data.ctrl[:] = np.clip(
#             self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
#         )
#
#     def _get_palm_xpos(self):
#         body_id = self.sim.model.body_name2id("robot0:palm")
#         return self.sim.data.body_xpos[body_id]
#
#     def _viewer_setup(self):
#         lookat = self._get_palm_xpos()
#         for idx, value in enumerate(lookat):
#             self.viewer.cam.lookat[idx] = value
#         self.viewer.cam.distance = 0.5
#         self.viewer.cam.azimuth = 55.0
#         self.viewer.cam.elevation = -25.0
