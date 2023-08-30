import gymnasium as gym
import time
import mujoco
import os

import numpy as np

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
env = gym.make("VariableFriction-v1", render_mode="human")
env.reset()
action = np.zeros(env.action_space.shape)
while 1:
    env.step(action)
#     print(env.observation_space)
#     observation, info = env.reset(seed=42)
#     for _ in range(1000):
#         action = env.action_space.sample()  # this is where you would insert your policy
#         observation, reward, terminated, truncated, info = env.step(action)
#
#         if terminated or truncated:
#             observation, info = env.reset()
#     env.close()

    # print(env.observation_space)

    # env = gym.make("HandManipulateBlock-v1")
    # print(env.observation_space)
    # print(env.observation_space["observation"])

    # The following always has to hold:
    # assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
    # assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
    # assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# MANIPULATE_BLOCK_XML = os.path.join("/Users/qiyangyan/Desktop/IHM_finger", "IHM.xml")
# fullpath = MANIPULATE_BLOCK_XML
# _mujoco = mujoco
# model = _mujoco.MjModel.from_xml_path(fullpath)
# data = _mujoco.MjData(model)
# print(data.qpos)

# env = gym.make("VariableFriction-v0",render_mode="human")
# env.reset()
# action = [0,0,0]
#
#
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)
# action = [0,0,1]
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)
# action = [1,0,1]
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)
# action = [0,0,0]
# next_env_dict, _, _, _, _ = env.step(action)
# next_state = next_env_dict["observation"]
# friction_state_L = next_state[1]
# friction_state_R = next_state[3]
# print(friction_state_L,friction_state_R)

# start_time = time.time()
# for _ in range(20):
#     next_env_dict, _, _, _, _ = env.step(action)
#     print("time:", time.time() - start_time, "L:", friction_state_L, "R", friction_state_R)
#
# print("next action")
# action = [0,0,1]
# count = 0
# while friction_state_L != 0 and friction_state_R != 0.34:
#     print("time:", time.time() - start_time, "count:", count, "L:", friction_state_L, "R", friction_state_R)
#     next_env_dict, _, _, _, _ = env.step(action)
#     next_state = next_env_dict["observation"]
#     friction_state_L = next_state[1]
#     friction_state_R = next_state[3]
#     count += 1

print("finish")

# 35 reaches approaximate right