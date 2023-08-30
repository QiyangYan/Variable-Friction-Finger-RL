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
    "distance": -0.5,
    "azimuth": 90,
    "elevation": -50,
    "lookat": np.array([0, -0.4, 0.2]),
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
            super().__init__(n_actions=7, **kwargs)

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (7,)

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
        self.last_friction = 0
        self.current_friction = 0
        self.action_count = 0
        self.last_motor_pos = 0
        self.current_motor_pos = 0
        self.motor_direction = 0  # 0 for clockwise, 1 for anticlockwise
        self.last_motor_direction = 0
        self.same_friction = True
        self.same_motor_direction = True
        self.pick_up = False
        self.closing = False
        self.count = 0
        self.torque_ctrl = 1

    '''
    check if position is reached, before reach to next position
    
    The action space follows the framework for manipulation. So the neutral network will obtain the amount of movement
    needed for each stage within the framework. And it will read the trajectory in this way.
    For stage 1: agent store the achieved goal of action [action1,0,0,0,0,0]
    For stage 2: agent store the achieved goal of action [action1,action2,0,0,0,0]
    ...
    For stage 7: agent stores the achieved goal of action [action1, ... , action7]
    So now, we have 7 times more transision tuple being sampled
    
    network input: current and goal position
    
    Framework:
    Pre-process the information to help agent to learn the model quicker
    Slide - Slide to left end if angle < 0, slide to right end if angle > 0
        (if pos > threshold, slide down, if pos < threshold, slide up)
        
        
    so... only need these five element from neural network
    Rotate1 - CW/ACW (needed if n = 2 with m > 0)
    Slide1: rotate angle
    Slide2: rotate angle
    Slide3: rotate angle
    Rotate2 - angle needed to rotate to compensate/extra
    '''

    '''
    _set_action should obtain 8 parameters every time
    [Slide, Rotate1, Slide1, Slide2, Slide3, Rotate2, onlyChangeFriction]
    Rotate90: -1 means to left, 1 means to right, 90 degrees
    negative means LF
    position means HF
    '''
    def _set_action(self, action): # 负责一直跑就行, 以及friction, 看有没有跑到由外面决定
        # print(action.shape)
        super()._set_action(action)  # check if action has the right shape: 7 dimension
        pick_up = 0
        self.data.ctrl[4] = pick_up
        ctrlrange = self.model.actuator_ctrlrange
        # print("ctrl_range",ctrlrange[0,1])

        if action[6] != 0:
            if action[6] == 2:
                if self.pick_up == False:
                    self.data.ctrl[0] = 1.05
                    self.data.ctrl[1] = 0
                    if self.action_complete():
                        # print("action complete")
                        self.pick_up = True
                else:
                    # print("self.pick_up", self.pick_up)
                    if self.closing == False:
                        self.data.ctrl[0] = 1.05
                        self.data.ctrl[1] = self.torque_ctrl
                        self.count += 1
                        # print("closing", self.count)
                        if self.count == 50:
                            self.closing = True
                    else:
                        '''change friction'''
                        # print("should appear twice", action)
                        self.data.ctrl[0] = self.data.qpos[1]
                        self.data.ctrl[2:4] = self.friction_state_mapping(0)
            else:
                self.data.ctrl[0] = self.data.qpos[1]
                self.data.ctrl[2:4] = self.friction_state_mapping(action[6])
        else:
            current_action_index = self.find_current_action_index(action)
            # print(current_action_index)
            if current_action_index == 1 or 5:
                # print("rotate")
                '''Rotate
                1. negative means ACW
                2. positive means CW
                '''
                # self.data.ctrl[1] = self.torque_ctrl
                self.data.ctrl[0] = ctrlrange[0,1] * (action[current_action_index] + 1) / 2
            else:
                # print("slide")
                '''Slide to side
                1. magnitude of action[0] is the pos of left finger, sign of action[0] is the friction state
                2. negative means LF
                3. position means HF
                '''
                # self.data.ctrl[1] = self.torque_ctrl
                self.data.ctrl[0] = ctrlrange[0,1] * (action[current_action_index] + 1) / 2

        self.same_action_check()
        self.action_count += 1
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def find_current_action_index(self,action):
        current_action_index = []
        for i, num in enumerate(action):
            if num != 0:
                current_action_index.append(i)
        # assert len(current_action_index) == 1
        if not current_action_index:
            return 0
        return current_action_index[0]

    def friction_state_mapping(self,action):
        '''
                Friction control follows this formate: [left right]
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
        if -1 <= action < 0:
            friction_ctrl = [0.34, 0]
        elif 0 < action <= 1:
            friction_ctrl = [0, 0.34]
        elif action == 0:
            friction_ctrl = [0, 0]
        friction_ctrl = np.array(friction_ctrl)
        # print(friction_ctrl)
        if friction_ctrl is None:
            raise ValueError("Invalid Action with Invalid Friction State")
        assert friction_ctrl.shape == (2,)
        return friction_ctrl

    def same_action_check(self):
        # update last-variables value
        if self.action_count > 0:
            self.last_friction = self.current_friction
            self.last_motor_pos = self.current_motor_pos
            self.last_motor_direction = self.motor_direction

        # get new values
        self.current_friction = self.data.ctrl[2:4]
        self.current_motor_pos = self.data.ctrl[0]
        if self.current_motor_pos - self.last_motor_pos > 0:
            self.motor_direction = 0
        elif self.current_motor_pos - self.last_motor_pos < 0:
            self.motor_direction = 1

        # check if same
        self.same_friction = False
        self.same_motor_direction = False
        if np.array_equal(self.last_friction, self.current_friction):
            self.same_friction = True
        if np.array_equal(self.motor_direction,self.last_motor_direction):
            self.same_motor_direction = True


