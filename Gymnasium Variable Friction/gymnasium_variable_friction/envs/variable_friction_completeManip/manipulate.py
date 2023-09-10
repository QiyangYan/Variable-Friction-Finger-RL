# Information regarding get_base_manipulate_env
'''

'''

# Action Space and Observation Space Definitions
'''Action Space and Observation Space Definitions.

Action Space:
l stands for rotate to left
r stands for rotate to right
0 stands for low_fric
1 stands for high_fric
1 unit angle for 1 time step
        L   R   L_s  R_s
1       \   l   0   1
2       \   l   1   0
3       \   l   1   1
4       r   \   0   1
5       r   \   1   0
6       r   \   1   1

Observation Space:
State: Instead of using predefined feature vector, I will use image as input
# 1   x-coordinate of the object
# 2   y-coordinate of the object
# 3   z-coordinate of the object
# 4   angular velocity of the left finger
# 5   angular velocity of the right finger
# 6   angle of the left finger
# 7   angle of the right finger
# 8

Reward:
1 Next state +5
2 object has no z-axis angle/distance-shift +10
3 object has small z-axis angle/distance-shift +1
4 Reach goal_pos +100
5 Below Velocity max_limit +1
6 Below (x,y,z) Velocity desired_limit +10
7 Same Action +10
8 Different Action +5
9 Below z-axis Angular velocity limit + 10

Done: Object fall off finger/angle>certain degree, exceed max_limit

CONSIDER FOLLOWING QUESTIONS
1. How to set the reward? cause the dense reward in the robot hand example also only considers the position
and angular difference between goal and current position, without considering healthy.

'''


from typing import Union

import numpy as np
from gymnasium import error

from gymnasium_robotics.envs.variable_friction_completeManip import MujocoHandEnv
from gymnasium_robotics.utils import rotations
from scipy.spatial.transform import Rotation

import xml.etree.ElementTree as ET
import os


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def get_base_manipulate_env(HandEnvClass: MujocoHandEnv):
    """Factory function that returns a BaseManipulateEnv class that inherits from MujocoPyHandEnv or MujocoHandEnv depending on the mujoco python bindings."""

    '''基于这个BaseManipulateEnv, 按需求进行添加'''
    class BaseManipulateEnv(HandEnvClass):
        def __init__(
            self,
            target_position,  # with a randomized target position (x,y,z), where z is fixed but with tolerance
            target_rotation,  # with a randomized target rotation around z, no rotation around x and y
            reward_type,  # more likely dense
            initial_qpos=None,
            randomize_initial_position=False,
            randomize_initial_rotation=False,
            distance_threshold=0.005,
            rotation_threshold=0.1,
            slip_pos_threshold = 0.005,  # this is the tolerance for displacement along z-axis due to slipping
            slip_rot_threshold = 0.2,  # this is the tolerance for rotation around z-axis due to slipping
            n_substeps=20,
            relative_control=False,
            # ignore_z_target_rotation=False,
            **kwargs,
        ):
            """Initializes a new Hand manipulation environment.

            Args:
                model_path (string): path to the environments XML file
                target_position (string): the type of target position:
                    - fixed: target position is set to the initial position of the object
                    - random: target position is fully randomized according to target_position_range
                target_rotation (string): the type of target rotation:
                    - z: fully randomized target rotation around the Z axis
                target_position_range (np.array of shape (3, 2)): range of the target_position randomization
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                randomize_initial_position (boolean): whether or not to randomize the initial position of the object
                randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
                distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
                rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
                n_substeps (int): number of substeps the simulation runs on every call to step
                relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state

                Removed:
                ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            """
            self.target_position = target_position
            self.target_rotation = target_rotation
            # 这段代码用于将一组 “欧拉角表示的旋转” (“平行旋转”) 转换为 “四元数表示”
            self.parallel_quats = [
                rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
            ]
            self.randomize_initial_rotation = randomize_initial_rotation
            self.randomize_initial_position = randomize_initial_position
            self.distance_threshold = distance_threshold
            self.rotation_threshold = rotation_threshold
            self.r_threshold = 0.001
            self.d_threshold = 0.01
            self.reward_type = reward_type
            self.slip_pos_threshold = slip_pos_threshold
            self.slip_rot_threshold = slip_rot_threshold
            self.switchFriction_count = 0;
            self.terminate_r_limit = [0.07,0.14]
            self.L = 0.015
            self.success = False
            self.left_contact_idx = None;
            self.right_contact_idx = None;
            # self.successSlide = False
            # self.ignore_z_target_rotation = ignore_z_target_rotation

            assert self.target_position in ["fixed", "random"]
            assert self.target_rotation in ["fixed", "z"]
            initial_qpos = initial_qpos or {}

            super().__init__(
                n_substeps=n_substeps,
                initial_qpos=initial_qpos,
                relative_control=relative_control,
                **kwargs,
            )


        def _is_success_radi(self,achieved_goal,desired_goal):
            '''find the actual goal, 3 = 2+1'''
            # print("object:target_corner1",self.model.site("object:target_corner1").pos)  # before get new goal
            # print("_is_success_radi")
            # d_radi, r_left, r_right = self._radi_error(achieved_goal[:7], desired_goal[:7])
            if len(achieved_goal.shape) == 1:
                '''each step'''
                # print("each step")
                # desired_goal = desired_goal[:8]
                d_radi = abs(achieved_goal[7:] - desired_goal[7:])
                # print("d_radi:",np.mean(d_radi))
                d_pos = self._goal_distance(achieved_goal[:7],desired_goal[:7])
                success_radi = (np.mean(d_radi) < self.r_threshold).astype(np.float32)
                success_pos = (d_pos < self.d_threshold).astype(np.float32)

            elif len(achieved_goal.shape) > 1:
                '''train'''
                # print("train")
                # desired_goal = desired_goal[:,:8]
                # print("achieved:",achieved_goal[:,7:])
                # print("desired:",desired_goal[:,7:])
                d_radi = abs(achieved_goal[:,7:] - desired_goal[:,7:])
                # print("d_radi:", d_radi)
                # print(achieved_goal[:,:7].shape,desired_goal[:,:7].shape)
                d_pos = self._goal_distance(achieved_goal[:,:7],desired_goal[:,:7])
                d_radi_mean = np.mean(d_radi, axis=1).reshape(-1, 1)
                # print("train:",d_radi_mean)
                d_radi_mean = np.where(d_radi_mean == 0, 1, d_radi_mean)
                success_radi = np.where(d_radi_mean < self.r_threshold, 1, 0)
                success_pos = np.where(d_pos < self.d_threshold, 1, 0)

            else:
                raise ValueError("Unsupported array shape.")

            return success_radi, success_pos, d_radi, d_pos

        def _goal_distance(self, goal_a, goal_b):
            ''' get pos difference and rotation difference
            left motor pos: 0.037012 -0.1845 0.002
            right motor pos: -0.037488 -0.1845 0.002
            '''
            assert goal_a.shape == goal_b.shape
            assert goal_a.shape[-1] == 7

            d_pos = np.zeros_like(goal_a[..., 0])

            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*

            # assert d_pos.shape == d_rot.shape
            # return d_pos, d_rot
            return d_pos

        def _compute_radi(self, a):
            # left motor pos: 0.037012 -0.1845 0.002
            # right motor pos: -0.037488 -0.1845 0.002
            a[2] = 0.002

            left_motor = [0.037012,-0.1845,0.002]
            right_motor = [-0.037488,-0.1845,0.002]

            assert a.shape[-1] == 7

            radius_al = np.zeros_like(a[..., 0])
            radius_ar = np.zeros_like(a[..., 0])

            delta_r_a_left_motor = a[..., :3] - left_motor # pos of motor left
            delta_r_a_right_motor = a[..., :3] - right_motor # pos of motor right
            radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)
            radius_ar = np.linalg.norm(delta_r_a_right_motor, axis=-1)

            return radius_al,radius_ar

        def compute_reward(self, achieved_goal, goal, info):
            self.reward_type = "dense"
            # print("reward_type:",self.reward_type)
            if self.reward_type == "sparse":
                '''success是 0, unsuccess是 1'''
                success, _, _ = self._is_success_radi(achieved_goal, goal)
                print("sparse")
                return success.astype(np.float32) - 1.0
            else:
                ''' dense: distance and angle dependent
                the negative summation of the Euclidean distance to the block’s target x 10
                + the theta angle difference to the target orientation	
                '''

                success_radi, success_pos, d_radi, d_pos = self._is_success_radi(achieved_goal, goal)

                success = success_radi + success_pos
                success *= 0

                if d_radi.ndim == 1:
                    # print("each step")
                    r_left, r_right = self._compute_radi(achieved_goal[:7])
                    # r_left = achieved_goal[7]
                    # r_right = achieved_goal[8]

                    if (not self.terminate_r_limit[0]+0.005 < r_left < self.terminate_r_limit[1]-0.005) or (
                            not self.terminate_r_limit[0]+0.005 < r_right < self.terminate_r_limit[1]-0.005):
                        outOfRange = -10
                    else:
                        outOfRange = 0

                    # print("success_radi:",success_radi)
                    if success_radi:
                        print(d_radi)
                        self.successSlide = True
                    elif success_radi and success_pos:
                        self.success = True

                    if success_radi != 1:
                        # if goal[9] <= 0:
                        #     return - (20 * d_radi[0] + 10 * d_radi[1]) + success
                        # else:
                        return - (20 * d_radi[1] + 10 * d_radi[0]) + outOfRange
                    else:
                        return - (20 * d_radi[1] + 10 * d_radi[0]) - d_pos * 20
                else:
                    # print("train")
                    '''compare center with this, not point of contact'''
                    r_left = achieved_goal[:,7]
                    r_right = achieved_goal[:,8]
                    # print("x is an array (list)")
                    reward = []
                    # print(d_radi.shape)
                    # print(d_pos.shape)
                    for i, radius in enumerate(np.array(d_radi)):
                        # if radius < 0.075:
                        #     success[i] = success[i] * 3
                        # if r_left[i] > 0.135 or r_right[i] > 0.135 or r_left[i] < 0.0535 or r_right[i] < 0.0535:
                        if (not self.terminate_r_limit[0] + 0.005 < r_left[i] < self.terminate_r_limit[1] - 0.005) or (
                                not self.terminate_r_limit[0] + 0.005 < r_right[i] < self.terminate_r_limit[1] - 0.005):
                            outOfRange = -10
                        else:
                            outOfRange = 0

                        if success_radi[i]:
                            ''' during rotation '''
                            # successSlide[i] = True
                            # if goal[i][9] <= 0:
                            #     slide_reward = - d_pos[i] * 20
                            # else:
                            reward.append(- (20 * d_radi[i][1] + 10 * d_radi[i][0]) - d_pos[i] * 20)
                        elif success_radi[i] and success_pos[i]:
                            # success[i] = True
                            reward.append(0)
                        else:
                            ''' during sliding '''
                            assert success_radi[i] != 1
                            # if goal[i][9] <= 0:
                            #     slide_reward = - (20 * d_radi[i][0] + 10 * d_radi[i][1])
                            # else:
                            # print(d_radi[i][0])
                            # print(d_radi[i][1])
                            reward.append(- (20 * d_radi[i][1] + 10 * d_radi[i][0]) + outOfRange)
                    return reward

                '''Add more reward term'''
                # slip penalty
                # d_slip, drop = self._slip_indicator(achieved_goal)  # d_slip is negative value

                # same action reward
                # if not self.same_friction and not self.same_motor_direction:
                #     d_action = -1
                #     self.switchFriction_count += 1
                # elif not self.same_friction or not self.same_motor_direction:
                #     self.switchFriction_count += 1
                #     d_action = -0.5
                # else:
                #     d_action = 0
                #
                # if self.switchFriction_count < 7:
                #     d_action = 0

        # RobotEnv methods
        # ----------------------------
        # def _slip_indicator(self,achieved_goal):
        #     start_position = self.initial_qpos
        #     qpos = self._utils.get_joint_qpos(self.model, self.data, "object:joint").copy() # 7 element
        #     slip_p = qpos[2] - start_position[2] # index: 0,1,2 (x,y,z) % not start pos
        #
        #     quat_b = achieved_goal[..., 3:]
        #     z_axis = np.array([1, 0.0087, 0, 0])
        #     if quat_b.shape[0] == 4:
        #         quat_diff_z_axis = rotations.quat_mul(z_axis, rotations.quat_conjugate(quat_b))
        #         slip_q = 2 * np.arccos(np.clip(quat_diff_z_axis[0], -1.0, 1.0))
        #         if abs(slip_p) > self.slip_pos_threshold or abs(slip_q) > self.slip_rot_threshold:
        #             penalty = - (abs(slip_p) + abs(slip_q))
        #             drop = True
        #         else:
        #             penalty = -20
        #             drop = False
        #     else:
        #         # print(quat_b.shape[0])  # this give number of tuples 256 from memory
        #         num_rows = quat_b.shape[0]
        #         z_axis = np.tile(z_axis, (num_rows, 1))
        #         quat_diff_z_axis = rotations.quat_mul(z_axis, rotations.quat_conjugate(quat_b))
        #         # print(num_rows)
        #         # print(z_axis.shape)
        #         penalty = {}
        #         drop = {}
        #         for i in range(num_rows):
        #             slip_q = 2 * np.arccos(np.clip(quat_diff_z_axis[i,0], -1.0, 1.0))
        #
        #             if np.abs(slip_p[i]) > self.slip_pos_threshold or np.abs(slip_q[i]) > self.slip_rot_threshold:
        #                 penalty[i] = - (abs(slip_p[i]) + abs(slip_q[i]))
        #                 drop[i] = True
        #             else:
        #                 penalty[i] = -20
        #                 drop[i] = False
        #
        #     # print((rotations.quat_conjugate(quat_b)).shape)
        #     # print(z_axis.shape)
        #
        #     return penalty, drop

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):
    def _get_achieved_goal(self):
        '''3 position element of object + 2 radius'''
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "joint:object")
        assert object_qpos.shape == (7,)

        if self.left_contact_idx != None and self.right_contact_idx != None:
            left_contact_point = self.data.site_xpos[self.left_contact_idx]
            right_contact_point = self.data.site_xpos[self.right_contact_idx]
            achieved_goal_radi = self.compute_goal_radi(left_contact_point, right_contact_point)
            # print("contact points:",left_contact_point,right_contact_point)
        else:
            achieved_goal_radi = [0,0]
            print("initialising")

        achieved_goal = np.concatenate((object_qpos, achieved_goal_radi))
        # print(achieved_goal)
        assert achieved_goal.shape == (9,)
        return achieved_goal

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)


    def _reset_sim(self):
        # print("reset")
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        # self.action_count = 0;
        self.switchFriction_count = 0;
        self.pick_up = False
        self.closing = False
        self.data.ctrl = 0
        self.count = 0
        self.last_motor_diff = 0
        self.torque_high_indicator = 0
        self.terminate_count = 0
        self.stuck_terminate = False
        self.successSlide = False
        self.success = False
        self.left_contact_idx = None;
        self.right_contact_idx = None;
        self.torque_ctrl = 1

        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        initial_qpos = self._utils.get_joint_qpos(
            self.model, self.data, "joint:object"
        ).copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        # 注意: 每次都需要升到固定高度,需要给z加一个offset
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "fixed":
                pass
            else:
                raise error.Error(
                    f'Unknown target_rotation option "{self.target_rotation}".'
                )

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != "fixed":
                # initial_pos += self.np_random.normal(size=3, scale=0.005)
                initial_pos = self._sample_coord(0)

        # finalise initial pose
        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self._utils.set_joint_qpos(self.model, self.data, "joint:object", initial_qpos)

        # def is_on_palm():
        #     self._mujoco.mj_forward(self.model, self.data)
        #     cube_middle_idx = self._model_names._site_name2id["object:center"]
        #     cube_middle_pos = self.data.site_xpos[cube_middle_idx]
        #     is_on_palm = cube_middle_pos[2] > 0.04
        #     return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(2))
            try:
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            except Exception:
                return False

        return True

    def _sample_coord(self, z):
        # sample region: a triangle
        # y_range = [-0.25, -0.315]
        # finger_length = [0.0505,0.1305] # this is the length of pad from motor
        # target_r_l = self.np_random.uniform(finger_length[0], finger_length[1])
        # target_r_r = self.np_random.uniform(finger_length[0], finger_length[1])

        # finger_length = np.linalg.norm(delta_pos, axis=-1)
        x_range = [-0.04, 0.04]
        x = self.np_random.uniform(x_range[0], x_range[1])
        y_range = [-0.24, 1.625*abs(x)-0.315]
        y = self.np_random.uniform(y_range[1],y_range[0])
        coord = [x, y-0.01, z]
        return coord


    def _sample_goal(self):

        # Select a goal for the object position.
        ''' this random was set to add offset to x,y,z, but now it will only add offset to x,y '''
        target_pos = None
        if self.target_position == "random":
            z = 0
            target_pos = self._sample_coord(z)
            target_pos = np.array(target_pos, dtype=np.float32)
        elif self.target_position in "fixed":
            target_pos = [0.02, -0.26, 0]
            target_pos = np.array(target_pos, dtype=np.float32)
        else:
            raise error.Error(
                f'Unknown target_position option "{self.target_position}".'
            )
        assert target_pos is not None
        assert target_pos.shape == (3,)
        # print("target_pos",target_pos)

        '''Select a goal for the object rotation.'''
        target_quat = None
        if self.target_rotation == "z":
            angle = self.np_random.uniform(-np.pi/4, np.pi/4)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in "fixed":
            angle = 0
            # target_quat = self.data.get_joint_qpos("object:joint")
            target_quat = self._utils.get_joint_qpos(
                self.model, self.data, "joint:target"
            ).copy()[3:]
        else:
            raise error.Error(
                f'Unknown target_rotation option "{self.target_rotation}".'
            )
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        # goal_radi = self.find_goal_radi(target_pos, angle)

        # convert goal coordinate to goal radisu
        '''3+4+2
        4 is needed to set the target position as joint
        '''
        goal_radi = np.zeros(2)
        goal = np.concatenate((target_pos,target_quat,goal_radi))
        assert goal.shape == (9,)

        return goal

    def _get_contact_point(self,goal):
        contact_coord = []
        for num in range(self.number_of_corners):
            contact_idx = self._model_names._site_name2id[f"target:corner{num + 1}"]
            contact_coord.append(self.data.site_xpos[contact_idx])
            # print("contact:", contact_idx, self.data.site_xpos[contact_idx])
        left_index, left_contact = max(enumerate(contact_coord), key=lambda coord: coord[1][0])
        right_index, right_contact = min(enumerate(contact_coord), key=lambda coord: coord[1][0])

        self.left_contact_idx = self._model_names._site_name2id[f"object:corner{left_index + 1}"]
        self.right_contact_idx = self._model_names._site_name2id[f"object:corner{right_index + 1}"]
        # print("left index of target and object:", left_index, self.left_contact_idx)

        left_contact[2] = 0.025
        left_contact_coord = np.concatenate((left_contact, [0, 0, 0, 0]))
        right_contact[2] = 0.025
        right_contact_coord = np.concatenate((right_contact, [0, 0, 0, 0]))

        goal_radi = self.compute_goal_radi(left_contact_coord[:3],right_contact_coord[:3])
        goal[7:] = goal_radi
        assert goal.shape == (9,)

        self._utils.set_joint_qpos(self.model, self.data, "site-checker", right_contact_coord)
        self._utils.set_joint_qvel(self.model, self.data, "site-checker", np.zeros(6))

        return goal

    def compute_goal_radi(self,a, b):
        '''
        a is the left contact point,
        b is the right contact point
        '''
        a[2] = 0.002
        b[2] = 0.002

        left_motor = [0.037012, -0.1845, 0.002]
        right_motor = [-0.037488, -0.1845, 0.002]

        assert a.shape == b.shape
        assert a.shape[-1] == 3

        radius_al = np.zeros_like(a[..., 0])
        radius_br = np.zeros_like(b[..., 0])

        delta_r_a_left_motor = a[..., :3] - left_motor  # pos of motor left
        radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)

        delta_r_b_right_motor = b[..., :3] - right_motor  # pos of motor right
        radius_br = np.linalg.norm(delta_r_b_right_motor, axis=-1)

        goal_radi = [radius_al,radius_br]

        return goal_radi

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (9,)

        self._utils.set_joint_qpos(self.model, self.data, "joint:target", goal[:7])
        self._utils.set_joint_qvel(self.model, self.data, "joint:target", np.zeros(6))

        # print("test complete")

        self._mujoco.mj_forward(self.model, self.data)


    def _get_obs(self):
        # what's expect with single joint: (array([], dtype=float64), array([], dtype=float64))
        # what's expected:
        # Position: 4 + 1 slide joints, 2 6DOF free joints, 1 + 4 + 2*7 = 19 element
        # each slide joint has position with 1 element, each free joints has position with 7 elements
        # for free joint, 3 elements are (x,y,z), 4 elements are (x,y,z,w) quaternion
        # Velocity: 4 + 1 slide joints, 2 DOF free joints, 1 + 4 + 2*6 = 17 element
        # Result:
        # pos = [0. 0.0.0.0. 0.-0.251365 0.1.0.0.0. 0.0.0.1.0.0.0.]
        # vel = [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        robot_qpos = self.data.qpos
        robot_qvel = self.data.qvel

        # achieved_goal = robot_qpos[4:11]
        object_qvel = robot_qvel[5:11]
        robot_qpos = robot_qpos[1:5]
        robot_qvel = robot_qvel[1:5]

        # simplify observation space

        '''object information: radius to two motor + current coordinate'''
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the current radius to two motor + current coordinate

        '''new observation
                position: 3 slide joints, 2 radius, pos of 6DOF free joint (3 elements)
                velocity: 3 slide joints, vel of 6 DOF free joint (6 elements)
                '''

        '''useful infor:
        robot_joint_pos: 3 element
        achieved-goal: 3 + 2
        '''
        robot_joint_pos = np.array([robot_qpos[0], robot_qpos[1], robot_qpos[3]])
        # print(np.array(robot_joint_pos))
        # achieved_goal = [achieved_goal[:3],achieved_goal[7:]]
        # assert achieved_goal.shape == (5,)
        assert robot_joint_pos.shape == (3,)

        # for observation: 3 + 2 + 3
        observation = np.concatenate(
            [
                robot_joint_pos,
                # robot_qvel,
                # object_qvel,
                achieved_goal
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def compute_terminated(self, achieved_goal, desired_goal, info):
        # exceed range
        radius_l, radius_r = self._compute_radi(achieved_goal[:7])
        # radius_l = achieved_goal[7]
        # radius_r = achieved_goal[8]
        # print(radius_l,radius_r)
        if self.stuck_terminate == True:
            print("terminate: stuck")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            return True
        elif (not self.terminate_r_limit[0] < radius_l < self.terminate_r_limit[1]) or (not self.terminate_r_limit[0] < radius_r < self.terminate_r_limit[1]):

            print("terminate: out of range", radius_l, radius_r)
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            return True
        elif self.success:
            print("success")
            self.data.ctrl[2] = self.data.qpos[2]
            self.data.ctrl[3] = self.data.qpos[4]
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            print("------------------------------------")
            return True
        else:
            """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
            return False