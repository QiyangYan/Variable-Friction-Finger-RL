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

from gymnasium_robotics.envs.shadow_dexterous_hand import MujocoHandEnv, MujocoPyHandEnv
from gymnasium_robotics.utils import rotations


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def get_base_manipulate_env(HandEnvClass: Union[MujocoHandEnv, MujocoPyHandEnv]):
    """Factory function that returns a BaseManipulateEnv class that inherits from MujocoPyHandEnv or MujocoHandEnv depending on the mujoco python bindings."""

    '''基于这个BaseManipulateEnv, 按需求进行添加'''
    class BaseManipulateEnv(HandEnvClass):
        def __init__(
            self,
            target_position,  # with a randomized target position (x,y,z), where z is fixed but with tolerance
            target_rotation,  # with a randomized target rotation around z, no rotation around x and y
            target_position_range,
            reward_type,  # more likely dense
            initial_qpos=None,
            randomize_initial_position=True,
            randomize_initial_rotation=True,
            distance_threshold=0.01,
            rotation_threshold=0.1,
            slip_pos_threshold = 0.01,  # this is the tolerance for displacement along z-axis due to slipping
            slip_rot_threshold = 0.1,  # this is the tolerance for rotation around z-axis due to slipping
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
            self.target_position_range = target_position_range
            # 这段代码用于将一组 “欧拉角表示的旋转” (“平行旋转”) 转换为 “四元数表示”
            self.parallel_quats = [
                rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
            ]
            self.randomize_initial_rotation = randomize_initial_rotation
            self.randomize_initial_position = randomize_initial_position
            self.distance_threshold = distance_threshold
            self.rotation_threshold = rotation_threshold
            self.reward_type = reward_type
            self.slip_pos_threshold = slip_pos_threshold
            self.slip_rot_threshold = slip_rot_threshold
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

        def _goal_distance(self, goal_a, goal_b):
            ''' This convert 7-dimensional goal to the pos difference and rotation difference'''
            ''' 
            This considers three dimensional space, x,y,z, whereas we only need consider x and y
            Check if this works well, otherwise, need to replace it with two dimensional space
            '''
            assert goal_a.shape == goal_b.shape
            assert goal_a.shape[-1] == 7


            d_pos = np.zeros_like(goal_a[..., 0])
            d_rot = np.zeros_like(goal_b[..., 0])

            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))  # convert w to actual angle θ
            d_rot = angle_diff

            assert d_pos.shape == d_rot.shape
            return d_pos, d_rot

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, action, info):
            if self.reward_type == "sparse":
                '''success是 0, unsuccess是 1'''
                success, _, _ = self._is_success(achieved_goal, goal).astype(np.float32)
                return success - 1.0
            else:
                ''' dense: distance and angle dependent
                the negative summation of the Euclidean distance to the block’s target x 10
                + the theta angle difference to the target orientation	
                '''
                success_bool, d_pos, d_rot = self._is_success(achieved_goal, goal)
                success = success_bool.astype(np.float32)
                d_achieve = success * 5

                '''Add more reward term'''
                # slip penalty
                d_slip, drop = self._slip_indicator(achieved_goal)  # d_slip is negative value

                # contact reward @Sean

                # same action reward
                if self.last_friction == self.current_friction:
                    d_action = 0.1
                else:
                    d_action = -0.1

                # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
                # dominated by `d_rot` (in radians).
                return -(10.0 * d_pos + d_rot) + d_achieve + d_slip + d_action  # d_slip is negative value

        # RobotEnv methods
        # ----------------------------

        def _is_success(self, achieved_goal, desired_goal):
            d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
            achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
            achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
            achieved_both = achieved_pos * achieved_rot
            return achieved_both, d_pos, d_rot

        def _slip_indicator(self,achieved_goal):
            start_position = self.initial_qpos
            qpos = self._utils.get_joint_qpos(self.model, self.data, "object:joint").copy() # 7 element
            slip_p = qpos[2] - start_position[2] # index: 0,1,2 (x,y,z)

            z_axis = np.array([0, 0, 1])
            quat_b = achieved_goal[..., 3:]
            quat_diff_z_axis = rotations.quat_mul(z_axis, rotations.quat_conjugate(quat_b))
            slip_q = 2 * np.arccos(np.clip(quat_diff_z_axis[0], -1.0, 1.0))

            if abs(slip_p) > self.slip_pos_threshold or abs(slip_q) > self.slip_rot_threshold:
                penalty = - ( abs(slip_p) + abs(slip_q) )
            else:
                penalty = -20
                drop = False
            return penalty, drop

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):
    def _get_achieved_goal(self):
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object:joint")
        assert object_qpos.shape == (7,)
        return object_qpos

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
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
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "parallel":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[
                    self.np_random.integers(len(self.parallel_quats))
                ]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ["xyz", "ignore"]:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1.0, 1.0, size=3)
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
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self._utils.set_joint_qpos(self.model, self.data, "object:joint", initial_qpos)

        def is_on_palm():
            self._mujoco.mj_forward(self.model, self.data)
            cube_middle_idx = self._model_names._site_name2id["object:center"]
            cube_middle_pos = self.data.site_xpos[cube_middle_idx]
            is_on_palm = cube_middle_pos[2] > 0.04
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            except Exception:
                return False

        return is_on_palm()

    def _sample_goal(self):

        # Select a goal for the object position.
        '''
        this random was set to add offset to x,y,z, but now it will only add offset to x,y
        '''
        target_pos = None
        if self.target_position == "random":
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(
                self.target_position_range[:, 0], self.target_position_range[:, 1]
                # sample in defined position range
            )
            assert offset.shape == (3,)
            offset = np.concatenate(offset[1:2], 0) # remove the z-offset
            target_pos = (
                self._utils.get_joint_qpos(self.model, self.data, "object:joint")[:3]
                + offset
            )
        elif self.target_position in ["ignore", "fixed"]:
            target_pos = self._utils.get_joint_qpos(
                self.model, self.data, "object:joint"
            )[:3]
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
        elif self.target_rotation == "parallel":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[
                self.np_random.integers(len(self.parallel_quats))
            ]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == "xyz":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1.0, 1.0, size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ["ignore", "fixed"]:
            target_quat = self.data.get_joint_qpos("object:joint")
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
        if self.target_position == "ignore":
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15

        self._utils.set_joint_qpos(self.model, self.data, "target:joint", goal)
        self._utils.set_joint_qvel(self.model, self.data, "target:joint", np.zeros(6))

        if "object_hidden" in self._model_names.geom_names:
            hidden_id = self._model_names.geom_name2id["object_hidden"]
            self.model.geom_rgba[hidden_id, 3] = 1.0
        self._mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation

        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal]
        )
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


# class MujocoPyManipulateEnv(get_base_manipulate_env(MujocoPyHandEnv)):
#     def _get_achieved_goal(self):
#         # Object position and rotation.
#         object_qpos = self.sim.data.get_joint_qpos("object:joint")
#
#         assert object_qpos.shape == (7,)
#         return object_qpos
#
#     def _env_setup(self, initial_qpos):
#         for name, value in initial_qpos.items():
#             self.sim.data.set_joint_qpos(name, value)
#         self.sim.forward()
#
#     def _reset_sim(self):
#         self.sim.set_state(self.initial_state)
#         self.sim.forward()
#
#         initial_qpos = self.sim.data.get_joint_qpos("object:joint").copy()
#
#         initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
#         assert initial_qpos.shape == (7,)
#         assert initial_pos.shape == (3,)
#         assert initial_quat.shape == (4,)
#         initial_qpos = None
#
#         # Randomization initial rotation.
#         if self.randomize_initial_rotation:
#             if self.target_rotation == "z":
#                 angle = self.np_random.uniform(-np.pi, np.pi)
#                 axis = np.array([0.0, 0.0, 1.0])
#                 offset_quat = quat_from_angle_and_axis(angle, axis)
#                 initial_quat = rotations.quat_mul(initial_quat, offset_quat)
#             elif self.target_rotation == "parallel":
#                 angle = self.np_random.uniform(-np.pi, np.pi)
#                 axis = np.array([0.0, 0.0, 1.0])
#                 z_quat = quat_from_angle_and_axis(angle, axis)
#                 parallel_quat = self.parallel_quats[
#                     self.np_random.integers(len(self.parallel_quats))
#                 ]
#                 offset_quat = rotations.quat_mul(z_quat, parallel_quat)
#                 initial_quat = rotations.quat_mul(initial_quat, offset_quat)
#             elif self.target_rotation in ["xyz", "ignore"]:
#                 angle = self.np_random.uniform(-np.pi, np.pi)
#                 axis = self.np_random.uniform(-1.0, 1.0, size=3)
#                 offset_quat = quat_from_angle_and_axis(angle, axis)
#                 initial_quat = rotations.quat_mul(initial_quat, offset_quat)
#             elif self.target_rotation == "fixed":
#                 pass
#             else:
#                 raise error.Error(
#                     f'Unknown target_rotation option "{self.target_rotation}".'
#                 )
#
#         # Randomize initial position.
#         if self.randomize_initial_position:
#             if self.target_position != "fixed":
#                 initial_pos += self.np_random.normal(size=3, scale=0.005)
#
#         initial_quat /= np.linalg.norm(initial_quat)
#         initial_qpos = np.concatenate([initial_pos, initial_quat])
#
#         self.sim.data.set_joint_qpos("object:joint", initial_qpos)
#
#         def is_on_palm():
#             self.sim.forward()
#             cube_middle_idx = self.sim.model.site_name2id("object:center")
#             cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
#
#             is_on_palm = cube_middle_pos[2] > 0.04
#             return is_on_palm
#
#         # Run the simulation for a bunch of timesteps to let everything settle in.
#         for _ in range(10):
#             self._set_action(np.zeros(20))
#             try:
#                 self.sim.step()
#             except self._mujoco_py.MujocoException:
#                 return False
#
#         return is_on_palm()
#
#     def _sample_goal(self):
#         # Select a goal for the object position.
#         target_pos = None
#         if self.target_position == "random":
#             assert self.target_position_range.shape == (3, 2)
#             offset = self.np_random.uniform(
#                 self.target_position_range[:, 0], self.target_position_range[:, 1]
#             )
#             assert offset.shape == (3,)
#             target_pos = self.sim.data.get_joint_qpos("object:joint")[:3] + offset
#
#         elif self.target_position in ["ignore", "fixed"]:
#             target_pos = self.sim.data.get_joint_qpos("object:joint")[:3]
#         else:
#             raise error.Error(
#                 f'Unknown target_position option "{self.target_position}".'
#             )
#         assert target_pos is not None
#         assert target_pos.shape == (3,)
#
#         # Select a goal for the object rotation.
#         target_quat = None
#         if self.target_rotation == "z":
#             angle = self.np_random.uniform(-np.pi, np.pi)
#             axis = np.array([0.0, 0.0, 1.0])
#             target_quat = quat_from_angle_and_axis(angle, axis)
#         elif self.target_rotation == "parallel":
#             angle = self.np_random.uniform(-np.pi, np.pi)
#             axis = np.array([0.0, 0.0, 1.0])
#             target_quat = quat_from_angle_and_axis(angle, axis)
#             parallel_quat = self.parallel_quats[
#                 self.np_random.integers(len(self.parallel_quats))
#             ]
#             target_quat = rotations.quat_mul(target_quat, parallel_quat)
#         elif self.target_rotation == "xyz":
#             angle = self.np_random.uniform(-np.pi, np.pi)
#             axis = self.np_random.uniform(-1.0, 1.0, size=3)
#             target_quat = quat_from_angle_and_axis(angle, axis)
#         elif self.target_rotation in ["ignore", "fixed"]:
#             target_quat = self.sim.data.get_joint_qpos("object:joint")
#         else:
#             raise error.Error(
#                 f'Unknown target_rotation option "{self.target_rotation}".'
#             )
#         assert target_quat is not None
#         assert target_quat.shape == (4,)
#
#         target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
#         goal = np.concatenate([target_pos, target_quat])
#         return goal
#
#     def _render_callback(self):
#         # Assign current state to target object but offset a bit so that the actual object
#         # is not obscured.
#         goal = self.goal.copy()
#         assert goal.shape == (7,)
#         if self.target_position == "ignore":
#             # Move the object to the side since we do not care about it's position.
#             goal[0] += 0.15
#         self.sim.data.set_joint_qpos("target:joint", goal)
#         self.sim.data.set_joint_qvel("target:joint", np.zeros(6))
#
#         if "object_hidden" in self.sim.model.geom_names:
#             hidden_id = self.sim.model.geom_name2id("object_hidden")
#             self.sim.model.geom_rgba[hidden_id, 3] = 1.0
#         self.sim.forward()
#
#     def _get_obs(self):
#         robot_qpos, robot_qvel = self._utils.robot_get_obs(self.sim)
#         object_qvel = self.sim.data.get_joint_qvel("object:joint")
#
#         achieved_goal = (
#             self._get_achieved_goal().ravel()
#         )  # this contains the object position + rotation
#         observation = np.concatenate(
#             [robot_qpos, robot_qvel, object_qvel, achieved_goal]
#         )
#         return {
#             "observation": observation.copy(),
#             "achieved_goal": achieved_goal.copy(),
#             "desired_goal": self.goal.ravel().copy(),
#         }
