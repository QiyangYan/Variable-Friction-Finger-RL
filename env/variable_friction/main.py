import gymnasium as gym
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = gym.make("VariableFriction-v0", render_mode="human")
    print(env.observation_space)
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

    # print(env.observation_space)

    # env = gym.make("HandManipulateBlock-v1")
    # print(env.observation_space)
    # print(env.observation_space["observation"])

    # The following always has to hold:
    # assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
    # assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
    # assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
