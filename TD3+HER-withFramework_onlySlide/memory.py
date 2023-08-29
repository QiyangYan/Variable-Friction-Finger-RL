import numpy as np
from copy import deepcopy as dc
import random


class Memory:
    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):

        ep_indices = np.random.randint(0, len(self.memory), batch_size)
        # time_indices = np.random.randint(0, len(self.memory[0]["next_state"]), batch_size)
        time_indices = []
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

        for episode in ep_indices:
            timestep = np.random.randint(0, len(self.memory[episode]["next_state"]))
            time_indices.append(timestep)
            states.append(dc(self.memory[episode]["state"][timestep]))
            actions.append(dc(self.memory[episode]["action"][timestep]))
            desired_goals.append(dc(self.memory[episode]["desired_goal"][timestep]))
            next_achieved_goals.append(dc(self.memory[episode]["next_achieved_goal"][timestep]))
            next_states.append(dc(self.memory[episode]["next_state"][timestep]))

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)
        time_indices = np.array(time_indices)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        episode_length = [len(self.memory[episode]["next_state"]) for episode in range(len(self.memory))]
        future_offset = [np.random.uniform() * (episode_length[episode] - timestep) for episode, timestep in
                         zip(ep_indices, time_indices)]
        future_offset = np.array(future_offset).astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(dc(self.memory[episode]["achieved_goal"][f_offset]))
        future_ag = np.vstack(future_ag)
        desired_goals[her_indices] = future_ag

        rewards = np.expand_dims(self.env.compute_reward(next_achieved_goals, desired_goals, None, actions), 1)

        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batch, stuck_episode):
        # stuck_episode is [episode, action_steps] with action steps of 1,2,3
        # get 100 number of transition randomly from two episode
        episode_length = (len(batch[0]["next_state"]),len(batch[1]["next_state"]))
        # print(episode_length)
        size = 100
        if np.all(stuck_episode[:, 0] == 0) or np.any(stuck_episode[:, 0] < 0):
            # if episode = 0
            ep_indices = np.random.randint(0, len(batch), size)
        else:
            # if episode = 1,2
            total_steps = np.sum(stuck_episode[:, 1])
            probabilities = [action_step / total_steps for action_step in stuck_episode[:, 1]]
            ep_indices = np.random.choice(range(len(batch)), size, p=probabilities)

        # store these transitions into new array
        states = []
        desired_goals = []
        time_indices = []
        for episode in ep_indices:
            timestep = np.random.randint(0, episode_length[episode])
            time_indices.append(timestep)
            states.append(dc(batch[episode]["state"][timestep]))
            desired_goals.append(dc(batch[episode]["desired_goal"][timestep]))

        # 把array里的多个episode整合成一个
        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)
        time_indices = np.array(time_indices)

        # 用来生成100个随机的index, 随机地从融合的array里抽取100个值
        her_indices = np.where(np.random.uniform(size=size) < self.future_p)
        # np.random.uniform(size=size)生成包含随机均匀分布的数值的 “数组”
        # 返回一个包含满足随机数小于概率阈值的元素索引的元组
        future_offset = [np.random.uniform() * (episode_length[episode] - timestep) for episode, timestep in
                         zip(ep_indices, time_indices)]
        # future_offset = np.random.uniform(size=size) * (len(batch[0]["next_state"]) - time_indices)
        future_offset = np.array(future_offset).astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []  # future achieved goal
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(dc(batch[episode]["achieved_goal"][f_offset]))
        future_ag = np.vstack(future_ag)

        # replace desired goal with future achieved goal
        desired_goals[her_indices] = future_ag

        return self.clip_obs(states), self.clip_obs(desired_goals)
