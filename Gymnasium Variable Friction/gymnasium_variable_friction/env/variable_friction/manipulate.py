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

from gymnasium_robotics.envs.variable_friction import MujocoHandEnv
from gymnasium_robotics.utils import rotations


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
            randomize_initial_position=True,
            randomize_initial_rotation=True,
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
            self.r_threshold = 0.015
            self.reward_type = reward_type
            self.slip_pos_threshold = slip_pos_threshold
            self.slip_rot_threshold = slip_rot_threshold
            self.switchFriction_count = 0;
            self.terminate_r_limit = [0.07,0.1355]
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
            d_radi, r_left, r_right = self._radi_error(achieved_goal,desired_goal)
            success = (d_radi < self.r_threshold).astype(np.float32)
            return success, d_radi, r_left, r_right

        def _radi_error(self, a, b):
            # left motor pos: 0.037012 -0.1845 0.002
            # right motor pos: -0.037488 -0.1845 0.002
            a[2] = 0.002
            b[2] = 0.002

            left_motor = [0.037012,-0.1845,0.002]
            right_motor = [-0.037488,-0.1845,0.002]

            assert a.shape == b.shape
            assert a.shape[-1] == 7

            radius_al = np.zeros_like(a[..., 0])
            radius_ar = np.zeros_like(a[..., 0])
            radius_bl = np.zeros_like(b[..., 0])
            radius_br = np.zeros_like(b[..., 0])


            delta_r_a_left_motor = a[..., :3] - left_motor # pos of motor left
            delta_r_a_right_motor = a[..., :3] - right_motor # pos of motor right
            radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)
            radius_ar = np.linalg.norm(delta_r_a_right_motor, axis=-1)

            delta_r_b_left_motor = b[..., :3] - left_motor  # pos of motor left
            delta_r_b_right_motor = b[..., :3] - right_motor  # pos of motor right
            radius_bl = np.linalg.norm(delta_r_b_left_motor, axis=-1)
            radius_br = np.linalg.norm(delta_r_b_right_motor, axis=-1)

            return abs(radius_bl-radius_al) + abs(radius_br-radius_ar),radius_al,radius_ar

        def _goal_distance(self, goal_a, goal_b):
            ''' This convert 7-dimensional goal to the pos difference and rotation difference'''
            ''' 
            This considers three dimensional space, x,y,z, whereas we only need consider x and y
            Check if this works well, otherwise, need to replace it with two dimensional space
            '''
            # left motor pos: 0.037012 -0.1845 0.002
            # right motor pos: -0.037488 -0.1845 0.002
            assert goal_a.shape == goal_b.shape
            assert goal_a.shape[-1] == 7

            d_pos = np.zeros_like(goal_a[..., 0])
            d_rot = np.zeros_like(goal_b[..., 0])

            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

            # print(goal_b, goal_a)
            # delta_pos_z = goal_a[..., 3] - goal_b[..., 3]
            # print("distance",delta_pos_z)
            # d_pos_z = np.linalg.norm(delta_pos_z)
            # print("distance", d_pos_z)

            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))  # convert w to actual angle θ
            d_rot = angle_diff

            assert d_pos.shape == d_rot.shape
            return d_pos, d_rot

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            self.reward_type = "dense"
            if self.reward_type == "sparse":
                '''success是 0, unsuccess是 1'''
                success, _, _ = self._is_success(achieved_goal, goal)
                print("sparse")
                return success.astype(np.float32) - 1.0
            else:
                ''' dense: distance and angle dependent
                the negative summation of the Euclidean distance to the block’s target x 10
                + the theta angle difference to the target orientation	
                '''
                # print("achieve:", achieved_goal)
                success, d_radi, r_left, r_right = self._is_success_radi(achieved_goal, goal)
                # achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
                # achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
                # d_achieve = achieved_pos + achieved_rot

                # success = (0.02 - d_radi) * 200 + 2: range is 6 ~ 2
                # closer to target, greater reward

                # success *= 2
                success *= 0
                if isinstance(d_radi, np.ndarray):
                    # print("x is an array (list)")
                    for i, radius in enumerate(np.array(d_radi)):
                        # if radius < 0.075:
                        #     success[i] = success[i] * 3
                        # if r_left[i] > 0.135 or r_right[i] > 0.135 or r_left[i] < 0.0535 or r_right[i] < 0.0535:
                        if (not self.terminate_r_limit[0] + 0.005 < r_left[i] < self.terminate_r_limit[1] - 0.005) or (
                                not self.terminate_r_limit[0] + 0.005 < r_right[i] < self.terminate_r_limit[1] - 0.005):
                            success[i] -= 10

                else:
                    # print("x is a single integer")
                    # if d_radi < 0.075:
                    #     success = success * 3
                    # if r_left > 0.135 or r_right > 0.135 or r_left < 0.0485 or r_right < 0.0485:
                    # [0.07,0.1355]
                    if (not self.terminate_r_limit[0]+0.005 < r_left < self.terminate_r_limit[1]-0.005) or (
                            not self.terminate_r_limit[0]+0.005 < r_right < self.terminate_r_limit[1]-0.005):
                        success -= 10

                # if self.switchFriction_count > 3:
                #     d_switch = -0.1
                # else:
                #     d_switch = 0

                # if np.any(achieved_goal[1] > -0.23) or np.any(achieved_goal[1] < -0.29):
                #     d_outofrange = -10
                #     # print("out of range")
                # else:
                #     d_outofrange = 0

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
                return - (20 * d_radi) + success

        # RobotEnv methods
        # ----------------------------

        def _is_success(self, achieved_goal, desired_goal):
            d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
            achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
            achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
            achieved_both = achieved_pos * achieved_rot
            # print("success:", achieved_goal,desired_goal)
            # print("reward:", d_pos, d_rot, d_pos_z)
            return achieved_both, d_pos, d_rot

        def _slip_indicator(self,achieved_goal):
            start_position = self.initial_qpos
            qpos = self._utils.get_joint_qpos(self.model, self.data, "object:joint").copy() # 7 element
            slip_p = qpos[2] - start_position[2] # index: 0,1,2 (x,y,z) % not start pos

            quat_b = achieved_goal[..., 3:]
            z_axis = np.array([1, 0.0087, 0, 0])
            if quat_b.shape[0] == 4:
                quat_diff_z_axis = rotations.quat_mul(z_axis, rotations.quat_conjugate(quat_b))
                slip_q = 2 * np.arccos(np.clip(quat_diff_z_axis[0], -1.0, 1.0))
                if abs(slip_p) > self.slip_pos_threshold or abs(slip_q) > self.slip_rot_threshold:
                    penalty = - (abs(slip_p) + abs(slip_q))
                    drop = True
                else:
                    penalty = -20
                    drop = False
            else:
                # print(quat_b.shape[0])  # this give number of tuples 256 from memory
                num_rows = quat_b.shape[0]
                z_axis = np.tile(z_axis, (num_rows, 1))
                quat_diff_z_axis = rotations.quat_mul(z_axis, rotations.quat_conjugate(quat_b))
                # print(num_rows)
                # print(z_axis.shape)
                penalty = {}
                drop = {}
                for i in range(num_rows):
                    slip_q = 2 * np.arccos(np.clip(quat_diff_z_axis[i,0], -1.0, 1.0))

                    if np.abs(slip_p[i]) > self.slip_pos_threshold or np.abs(slip_q[i]) > self.slip_rot_threshold:
                        penalty[i] = - (abs(slip_p[i]) + abs(slip_q[i]))
                        drop[i] = True
                    else:
                        penalty[i] = -20
                        drop[i] = False

            # print((rotations.quat_conjugate(quat_b)).shape)
            # print(z_axis.shape)

            return penalty, drop

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):
    def _get_achieved_goal(self):
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "joint:object")
        assert object_qpos.shape == (7,)
        return object_qpos

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

        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        initial_qpos = self._utils.get_joint_qpos(
            self.model, self.data, "object:joint"
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

        self._utils.set_joint_qpos(self.model, self.data, "object:joint", initial_qpos)

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
        '''
        this random was set to add offset to x,y,z, but now it will only add offset to x,y
        '''
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

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == "z":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in "fixed":
            # target_quat = self.data.get_joint_qpos("object:joint")
            target_quat = self._utils.get_joint_qpos(
                self.model, self.data, "object:joint"
            ).copy()[3:]
        else:
            raise error.Error(
                f'Unknown target_rotation option "{self.target_rotation}".'
            )
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)

        self._utils.set_joint_qpos(self.model, self.data, "target:joint", goal)
        self._utils.set_joint_qvel(self.model, self.data, "target:joint", np.zeros(6))

        self._mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        # what's expect with single joint: (array([], dtype=float64), array([], dtype=float64))
        # what's expected:
        # Position: 4 + 1 slide joints, 2 6DOF free joints, 1 + 4 + 2*7 = 19 element
        # each slide joint has position with 1 element, each free joints has position with 7 elements
        # for free joint, 3 elements are (x,y,z), 4 elements are (x,y,z,w) quaternion
        # Velocity: 4 slide joints, 2 DOF free joints, 1 + 4 + 2*6 = 17 element
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


        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation

        # for observation: 4 + 4 + 6 + 7 = 21
        observation = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                object_qvel,
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
        radius_l, radius_r = self._compute_radi(achieved_goal)
        # print(radius_l,radius_r)
        if self.stuck_terminate == True:
            # print("terminate: stuck")
            return True
        # elif radius_l < self.terminate_r_limit[0] and radius_l > self.terminate_r_limit[1]:
        #     print("terminate: left finger out of range")
        #     return True
        # elif radius_l < self.terminate_r_limit[0] and radius_l > self.terminate_r_limit[1]:
        #     print("terminate: left finger out of range")
        elif (not self.terminate_r_limit[0] < radius_l < self.terminate_r_limit[1]) or (not self.terminate_r_limit[0] < radius_r < self.terminate_r_limit[1]):
            # print("terminate: out of range")
            return True
        else:
            """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
            return False

    def _compute_radi(self, a):
        # left motor pos: 0.037012 -0.1845 0.002
        # right motor pos: -0.037488 -0.1845 0.002
        a[2] = 0.002

        left_motor = [0.037012, -0.1845, 0.002]
        right_motor = [-0.037488, -0.1845, 0.002]

        assert a.shape[-1] == 7

        radius_al = np.zeros_like(a[..., 0])
        radius_ar = np.zeros_like(a[..., 0])

        delta_r_a_left_motor = a[..., :3] - left_motor  # pos of motor left
        delta_r_a_right_motor = a[..., :3] - right_motor  # pos of motor right
        radius_al = np.linalg.norm(delta_r_a_left_motor, axis=-1)
        radius_ar = np.linalg.norm(delta_r_a_right_motor, axis=-1)

        return radius_al, radius_ar
