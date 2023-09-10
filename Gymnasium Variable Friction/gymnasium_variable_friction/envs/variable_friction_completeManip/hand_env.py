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
Box(-1.0, 1.0, (2,), float32), haven't confirm other parameters

| Num | Action                                              | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
| --- | ----------------------------------------------------| ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
| 0   | Angular position of the left finger                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
# | 1   | Angular position of the right finger                | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
| 1   | Friction States                                     | -1          | 1           | -1.571 (rad) | 1.571 (rad) | robot0:A_FFJ3 & robot0:A_FFJ4    | hinge | angle (rad) |

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
            super().__init__(n_actions=2, **kwargs)

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (2,)


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
        self.torque_ctrl = 1
        self.pick_up = False
        self.closing = False
        self.count = 0
        self.last_motor_diff = 0
        self.torque_high_indicator = 0
        self.terminate_count = 0
        self.stuck_terminate = False
        # self.last_friction = 0
        # self.current_friction = 0
        # self.action_count = 0
        # self.last_motor_pos = 0
        # self.current_motor_pos = 0
        # self.motor_direction = 0 # 0 for clockwise, 1 for anticlockwise
        # self.last_motor_direction = 0
        # self.same_friction = True
        # self.same_motor_direction = True

    def _set_action(self, action):
        # print("set action")
        super()._set_action(action)  # check if action has the right shape: 3 dimension
        # print(self.data.ctrl[:5])

        '''pick up to air'''
        # if self.action_count > 95:
        #     pick_up = 3
        # else:
        pick_up = 0
        self.data.ctrl[4] = pick_up
        gripper_pos_ctrl, friction_state = action[0], action[1]
        # print(gripper_pos_ctrl)
        ctrlrange = self.model.actuator_ctrlrange
        # gripper_pos_ctrl = -1~1, 2, friction_state = -1, 0, 1, 2

        '''grasp object'''
        if friction_state == 2:
            # print("grasping")
            if self.pick_up == False:
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = 0
                if self.action_complete(): # when hh, 0.001
                    # print("action complete")
                    self.pick_up = True
            elif self.closing == True and self.pick_up == True:
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = self.torque_ctrl
            else:
                # print(self.pick_up, self.closing)
                assert self.closing == False and self.pick_up == True
                self.data.ctrl[0] = 1.05
                self.data.ctrl[1] = self.torque_ctrl
                self.count += 1
                # print("closing", self.count)
                if self.count == 50:
                    self.closing = True


        elif gripper_pos_ctrl == 2:
            '''change friction'''
            assert friction_state == -1 or 0 or 1

            '''complete the last action before the change of friction, so until here, there is no change of friction'''
            # if not self.check_action_complete():
            '''If motor_diff is high, meaning that torque is too big so that it stops the manipulation'''
            '''Check this bit, it's not very appropriate'''
            if abs(abs(self.motor_diff) - self.last_motor_diff) < 0.002:
                self.torque_high_indicator += 1
            else:
                self.torque_high_indicator -= 1

            if self.torque_high_indicator > 10:
                self.torque_high_indicator = 10
                self.torque_ctrl = self.torque_ctrl - 0.1
                if self.torque_ctrl < 0:
                    self.torque_ctrl = 0.1
                    '''if situation doesn't get better after several steps, then terminate'''
                self.terminate_count += 1
                # print(self.terminate_count,self.torque_ctrl)
                print("stucking===========", self.terminate_count)
                if self.terminate_count == 20:
                    self.stuck_terminate = True
            else:
                # self.torque_ctrl = self.torque_ctrl + 0.1
                # if self.torque_ctrl > 1:
                self.torque_ctrl = 1
                self.terminate_count = 0

            self.last_motor_diff = self.motor_diff
            self.data.ctrl[1] = self.torque_ctrl
            # print("torque:", self.torque_ctrl, "torque high indicator:", abs(abs(self.motor_diff) - self.last_motor_diff), self.torque_high_indicator)

            # else:
            if self.check_action_complete(): # when hl, 0.0001
                '''if action is completed, friction change can start'''
                self.data.ctrl[0] = self.data.qpos[1]
                self.data.ctrl[2:4] = self.friction_state_mapping(friction_state)
                self.switchFriction_count += 1

        else:
            '''manipulation'''
            # print("manipulation")
            # assert self.pick_up == True and self.closing == True, "pick up and closing is not complete"
            self.data.ctrl[1] = 1 # change friction if stuck
            self.data.ctrl[0] = ctrlrange[0, 1] * (gripper_pos_ctrl + 1) / 2
            # ctrlrange[0, 1] is because only single motor is controlled by action

        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


        '''Before is the old version'''
        # Calculate the half of each actuator's control range
        # ctrlrange = self.model.actuator_ctrlrange
        # actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        # actuation_range = actuation_range[:4]
        # gripper_pos_ctrl, friction_state = action[:2], action[2]

        # if friction_state >= 0.33:
        #     friction_state = 1
        # elif friction_state <= -0.33:
        #     friction_state = -1
        # elif -0.33 < friction_state < 0.33:
        #     friction_state = 0
        # elif self.action_count == 0:
        #     friction_state = 0
        # else:
        #     print(friction_state)
        #     assert(friction_state,2) # 随便写个东西来报错

        # if self.action_count > 0:
        #     self.last_friction = self.current_friction
        #     self.last_motor_pos = self.current_motor_pos
        #     self.last_motor_direction = self.motor_direction

        # self.current_friction = friction_state
        # self.current_motor_pos = gripper_pos_ctrl[0] # Rotate in same direction
        # if self.current_motor_pos - self.last_motor_pos > 0:
        #     self.motor_direction = 0
        # elif self.current_motor_pos - self.last_motor_pos < 0:
        #     self.motor_direction = 1

        # if self.action_count > 0:
        #     self.same_action_check()

        # self.action_count += 1

        # print("last_friction:", self.last_friction, "current_friction", self.current_friction)

        # Friction control follows this formate: [left right]

        # if self.last_friction != self.current_friction:
        #     # print(self.last_friction, self.current_friction)
        #     friction_state_switch_steps = 600
        #     self.update_mujoco_step(friction_state_switch_steps)
        #     # self.change_friction(friction_ctrl,action)
        # elif self.last_friction == self.current_friction:
        #     friction_state_switch_steps = 20
        #     self.update_mujoco_step(friction_state_switch_steps)

        # Obtain actuation center (movement reference point)
        # if self.relative_control:
        #     # Move related to the current position
        #     actuation_center = np.zeros_like(action)  # To store the reference position for relative control mode
        #     for i in range(self.data.ctrl.shape[0]-1):
        #         actuation_center[i] = self.data.get_joint_qpos(
        #             self.model.actuator_names[i].replace(":_A",":")  # convert actuator name to joint name
        #         )
        #         print(actuation_center)
        #     '''Add this if the joints that control small servos are not included'''
        #     '''
        #     for joint_name in ["LFF", "RFF"]:
        #         act_idx = self.model.actuator_name2id(f"robot0:A_{joint_name}J1")
        #         actuation_center[act_idx] += self.data.get_joint_qpos(
        #             f"robot0:{joint_name}J0"
        #         )
        #     '''
        # else:
        #     # Move related to the center
        #     actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        #     actuation_center = actuation_center[:4]
        #
        # # self.data.ctrl[:4] = np.concatenate([actuation_center[:2] + gripper_pos_ctrl * actuation_range[:2], friction_ctrl])
        # self.data.ctrl[0] = actuation_center[0] + gripper_pos_ctrl[0] * actuation_range[0]
        # self.data.ctrl[1] = 1
        # self.data.ctrl[2:4] = friction_ctrl
        # self.data.ctrl[4] = pick_up
        # self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    # def change_friction(self,friction_ctrl,action):
    #     friction_state_switch_steps = 600
    #     self.update_mujoco_step(friction_state_switch_steps)
    #     self.data.ctrl[2:4] = friction_ctrl
    #     self.data.ctrl[:] = np.clip(self.data.ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
    #     self._mujoco_step(action)
    #     self._step_callback()
    #     if self.render_mode == "human":
    #         self.render()

    # def same_action_check(self):
    #     self.same_friction = False
    #     self.same_motor_direction = False
    #     if self.last_friction == self.current_friction:
    #         self.same_friction = True
    #     if self.motor_direction == self.last_motor_direction:
    #         self.same_motor_direction = True

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
