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
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
import math

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.5,
    "azimuth": 55.0,
    "elevation": -25.0,
    "lookat": np.array([1, 0.96, 0.14]),
}


def get_base_hand_env(
    RobotEnvClass: MujocoRobotEnv
    # 它表示 RobotEnvClass 参数的类型是 Union[MujocoPyRobotEnv, MujocoRobotEnv]
    # 并且它是一个必需的参数，调用函数时需要传递这个参数的值
) -> MujocoRobotEnv:
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
        actuation_range = actuation_range[:4]
        gripper_pos_ctrl, friction_state = action[:2], action[2]

        if self.action_count == 0:
            self.current_friction = friction_state
        else:
            self.last_friction = self.current_friction
            self.current_friction = friction_state
        self.action_count += 1

        if self.action_acount > 20:
            pick_up = 3
        else:
            pick_up = 0

        # Friction control follows this formate: [left right]
        '''
        Third element decide, LF&HF, or HF&LF, or HF&HF
        Friction State          -1          0           1
        Left friction servo     L(-90)      H(-90)        H(0)
        Right friction servo    H(0)        H(90)        L(90)

        -- On Dynamixel
        For left finger: Low friction is 60 degree (204.8), High friction is 150 degree (512)
        For right finger: Low friction is 240 degree (818.4), High friction is 150 degree (512)

        -- On Mujoco (Define the 0 degree is the centre, clockwise is negative)
        For left finger: Low friction is -90 degree (-1.571 rad), High friction is 0 degree (0 rad)
        For right finger: Low friction is 90 degree (1.571 rad), High friction is 0 degree (0 rad)
        Note: + 150 degree for Dynamixel Control
        '''
        friction_state_mapping = {
            -1: [2 * actuation_range[3], 0],
            0: [2 * actuation_range[3], 2 * actuation_range[3]],
            1: [0, 2 * actuation_range[3]]
        }
        friction_ctrl = friction_state_mapping.get(friction_state, [0,0])
        friction_ctrl = np.array(friction_ctrl)
        if friction_ctrl is None:
            raise ValueError("Invalid Action with Invalid Friction State")
        assert friction_ctrl.shape == (2,)

        # Obtain actuation center (movement reference point)
        if self.relative_control:
            # Move related to the current position
            actuation_center = np.zeros_like(action)  # To store the reference position for relative control mode
            for i in range(self.data.ctrl.shape[0]-1):
                actuation_center[i] = self.data.get_joint_qpos(
                    self.model.actuator_names[i].replace(":_A",":")  # convert actuator name to joint name
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
            actuation_center = actuation_center[:4]

        self.data.ctrl[:4] = np.concatenate([actuation_center[:2] + gripper_pos_ctrl * actuation_range[:2], friction_ctrl])
        self.data.ctrl[4] = pick_up
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

