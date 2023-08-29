import gymnasium as gym
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from play import Play
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch
import gymnasium.utils.seeding
from gymnasium_robotics.utils import rotations
import math

ENV_NAME = "VariableFriction-v1"
INTRO = False
Train = False
Play_FLAG = True
MAX_EPOCHS = 200
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 2
memory_size = 7e+5 // 50
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4
update_freq = 2
# centre_to_corner = 0.015 / math.sqrt(2)

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

# print("state_shape", state_shape, "n_actions", n_actions, "n_goals", n_goals)
# print(action_bounds)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'

env = gym.make(ENV_NAME)
# env.seed(MPI.COMM_WORLD.Get_rank())
gymnasium.utils.seeding.np_random(MPI.COMM_WORLD.Get_rank())
random.seed(MPI.COMM_WORLD.Get_rank())
np.random.seed(MPI.COMM_WORLD.Get_rank())
torch.manual_seed(MPI.COMM_WORLD.Get_rank())
agent = Agent(n_states=state_shape,
              n_actions=n_actions-1,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions-1,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(env))
agent.load_weights("/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_1_stuck.pth")

def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(MAX_EPISODES):
        env_dictionary = env_.reset()[0]
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        while np.linalg.norm(ag - g) <= 0.05:
            env_dictionary = env_.reset()[0]
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0

        '''get action'''
        with torch.no_grad():
            complete_action = agent_.choose_action(s, g)
        action_array, action = action_preprocess(complete_action)

        '''pick up block'''
        pick_up()

        '''Initialise'''
        action_steps = 0
        time_steps = 0
        friction_change_complete = False
        stuck_indicator = 0
        slow_indicator = 0
        adjust_torque(env_, lower=False)

        '''Episode start'''
        while action_steps != 6:
            time_steps += 1
            '''Complete friction change for the move'''
            if not friction_change_complete:
                friction_change_complete = change_friction(action, env_)
                action[6] = 0
                # print("Complete friction change")

            '''Start movement'''
            observation_new, r, _, _, info_ = env_.step(action)
            s_ = observation_new['observation']
            ag_ = observation_new['achieved_goal']
            ep_r += r

            next_vel = (abs(s_[8]) + abs(s_[9]) * 10000)
            current_vel = (abs(s[8]) + abs(s[9]) * 10000)
            # print(next_vel-current_vel)
            if current_vel < next_vel and current_vel < 20:
                stuck_indicator += 1
                adjust_torque(env_,lower=True)
            elif current_vel > next_vel and current_vel < 5:
                slow_indicator += 1

            s = s_.copy()

            if env_.action_complete() or stuck_indicator == 10 or ag_[1] > -0.242 or slow_indicator == 10:
                action_steps += 1
                adjust_torque(env_,lower=False)
                if action_steps == 1 or action_steps == 5:
                    action_steps += 1

                if stuck_indicator == 10 or ag_[1] > -0.242:
                    total_success_rate.append(info_['is_success'])
                    break
                elif action_steps == 6:
                    total_success_rate.append(info_['is_success'])
                    break

                action = action_array[action_steps] - action_array[action_steps - 1]
                action[6] = action_array[action_steps][6]
                friction_change_complete = False
                stuck_indicator = 0

        '''episode end'''
        # print(per_success_rate)
        # print(total_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)

    total_success_rate = np.array(total_success_rate, dtype=object)
    # 计算每个episode最后一步的成功率的平均值（局部成功率）
    local_success_rate = np.mean(total_success_rate)
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r[-1], ep_r

def change_friction(action, env):
    _, _, _, _, _ = env.step(action)
    # step = 0
    while not env.action_complete():
        # step += 1
        # if step % 1000 == 0:
        #     print("changing friction for action:", action)
        _, _, _, _, _ = env.step(action)
    return True

def pick_up():
    '''Pick Up'''
    pick_up = [0, 0, 0, 0, 0, 0, 2]
    # print("start picking")
    _, _, _, _, _ = env.step(pick_up)
    while not env.action_complete():
        _, _, _, _, _ = env.step(pick_up)
        # let self.pick_up = true
    for _ in range(50):
        _, _, _, _, _ = env.step(pick_up)
    # print("pick up complete --------")

def adjust_torque(env, lower):
    env.adjust_torque(lower)

def action_preprocess(complete_action):
    action_array = np.zeros((6, 7))
    # print("Action for the episode:", complete_action)
    action_array[0][0] = complete_action[0]
    action_array[0][6] = np.sign(complete_action[0])
    for i in range(5):
        action_array[i + 1] = action_array[i]
        action_array[i + 1][i + 1] = complete_action[i + 1]
        if i == 0 or i == 4:
            action_array[i + 1][6] = 2
        else:
            action_array[i + 1][6] = np.sign(complete_action[i + 1])
    return action_array, action_array[0]

def modify_action(action_dict,state,action_steps):
    angle = state[0]
    action = (angle * 2 / 1.68) - 1
    for action_step in action_dict["action"]:
        action_step[action_steps] = action
    return action_dict


if Train:
    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES):
            start_time_cycle = time.time()
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            stuck_episode = np.zeros((MAX_EPISODES,2))
            for episode in range(MAX_EPISODES):
                episode_start = time.time()
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                action_dict = {
                    "state": [],
                    "action": [],
                    "achieved_goal": [],
                    "desired_goal": []}
                env_dict = env.reset()[0]
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict = env.reset()[0]
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]

                '''get action'''
                complete_action = agent.choose_action(state, desired_goal)
                # print(complete_action)
                action = []
                action_array = np.zeros((6, 7))
                action_array, action = action_preprocess(complete_action)
                # print(action_array)

                '''pick up block'''
                pick_up()

                '''Initialise'''
                action_steps = 2
                time_steps = 0
                friction_change_complete = False
                stuck_indicator = 0
                slow_indicator = 0
                adjust_torque(env, lower=False)

                while action_steps != 6:
                    time_steps += 1
                    # print("action_steps", action_steps)
                    '''Complete friction change for the move'''
                    if not friction_change_complete:
                        friction_change_complete = change_friction(action,env)
                        action[6] = 0
                        # print("Complete friction change")

                    '''Start movement'''
                    next_env_dict, reward, _, _, info = env.step(action)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]
                    # print((abs(next_state[8])+abs(next_state[9])*10000))

                    action_dict["state"].append(state.copy())
                    action_dict["action"].append(action_array[action_steps][:6].copy())
                    action_dict["achieved_goal"].append(achieved_goal.copy())
                    action_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                    '''check if velocity has any error'''
                    next_vel = (abs(next_state[8]) + abs(next_state[9]) * 10000)
                    current_vel = (abs(state[8]) + abs(state[9]) * 10000)
                    # print(next_vel-current_vel)
                    if current_vel < next_vel and current_vel < 20:
                        stuck_indicator += 1
                        if stuck_indicator % 5 == 0:
                            adjust_torque(env, lower=True)
                    elif next_vel < current_vel < 5:
                        slow_indicator += 1

                    current_time = time.time()
                    if env.action_complete() or stuck_indicator == 10 or achieved_goal[1] > -0.242 or slow_indicator == 10 or current_time-episode_start > 5:
                        # time.sleep(2)
                        action_steps += 1
                        adjust_torque(env, lower=False)
                        if action_steps == 1 or action_steps == 5:
                            action_steps += 1

                        if action_steps == 6:
                            stuck_episode[episode] = [0, 6]
                            for key, value in action_dict.items():
                                if key in episode_dict:
                                    episode_dict[key].extend(value)
                            break
                        elif reward != -0.1:
                            stuck_episode[episode] = [-episode-1, 6]
                            action_dict = modify_action(action_dict, state, action_steps - 1)
                            print("success")
                            for key, value in action_dict.items():
                                if key in episode_dict:
                                    episode_dict[key].extend(value)
                            break

                        '''考虑: 当stuck的时候把整个action stage的action都替换成结束点的action'''
                        if stuck_indicator == 10 or achieved_goal[1] > -0.242 or current_time-episode_start > 5:
                            stuck_episode[episode] = [episode+1, action_steps-1]  # action 2,3,4 -> 1,2,3方便后面根据step数量计算采样率
                            '''改action可能不合适, 因为不同action的qvel不同, 所以observation其实是有区别的,不能直接换observation'''
                            action_dict = modify_action(action_dict, state, action_steps-1)  # 这个从0~5因为是作为index
                            '''modify desired goal'''
                            for step in action_dict["desired_goal"]:
                                step = action_dict["achieved_goal"][-1]
                            # break

                        for key, value in action_dict.items():
                            if key in episode_dict:
                                episode_dict[key].extend(value)

                        action = action_array[action_steps] - action_array[action_steps-1]
                        action[6] = action_array[action_steps][6]
                        friction_change_complete = False
                        stuck_indicator = 0

                # states, actions, rewards, next_states, goals
                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                # episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            ram = psutil.virtual_memory()
            if cycle % 10 == 0:
                print(f"Cycle:{cycle}| "
                  f"Duration:{time.time() - start_time_cycle:.3f}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")

            # print(stuck_episode)
            agent.store(mb, stuck_episode)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss / num_updates
            agent.update_networks()

        ram = psutil.virtual_memory()
        success_rate, running_reward, episode_reward = eval_agent(env, agent)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        if MPI.COMM_WORLD.Get_rank() == 0:
            t_success_rate.append(success_rate)
            print(f"Epoch:{epoch}| ")
            # print(running_reward)
            # print(f"Running_reward:{running_reward.item():.3f}| ")
            print(f"EP_reward:{episode_reward:.3f}| ")
            print(f"Memory_length:{len(agent.memory)}| ")
            print(f"Duration:{time.time() - start_time:.3f}| ")
            print(f"Actor_Loss:{actor_loss:.3f}| ")
            print(f"Critic_Loss:{critic_loss:.3f}| ")
            print(f"Success rate:{success_rate:.3f}| ")
            print(f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            agent.save_weights(ENV_NAME)

    if MPI.COMM_WORLD.Get_rank() == 0:

        with SummaryWriter("logs") as writer:
            for i, success_rate in enumerate(t_success_rate):
                writer.add_scalar("Success_rate", success_rate, i)

        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate)
        plt.title(f"Success rate: {ENV_NAME}")
        plt.savefig(f"success_rate_{ENV_NAME}.png")
        plt.show()

elif Play_FLAG:
    # player = Play(env, agent, ENV_NAME, max_episode=100)
    # player.evaluate()
    address_list = [
        "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_1.pth"
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_1_stuck.pth"
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_1_goodPerformanceFrom8.26Night.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_20_epoch.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_20_epoch_8.26.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_30_epoch.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_75_epoch.pth"
    ]
    for address in address_list:
        agent.load_weights(address)
        max_episode = 200
        total_success_rate = 0
        for index in range(max_episode):
            if index % 20 == 0:
                print(index)
            success_rate, _, _ = eval_agent(env, agent)
            total_success_rate += success_rate
        print(f"Success rate:{total_success_rate / 100:.3f}| ")


