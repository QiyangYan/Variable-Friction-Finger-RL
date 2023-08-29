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

ENV_NAME = "VariableFriction-v1"
INTRO = False
Train = True
Play_FLAG = False
MAX_EPOCHS = 400
MAX_CYCLES = 20
num_updates = 18
MAX_EPISODES = 2
memory_size = 7e+5 // 50
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4
update_freq = 2

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

print("state_shape", state_shape, "n_actions", n_actions, "n_goals", n_goals)
# print(action_bounds)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'


def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        per_success_rate = []
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
        '''episode start'''
        for t in range(50):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, _, _, info_ = env_.step(a)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r

        '''episode end'''
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)

    total_success_rate = np.array(total_success_rate)
    # 计算每个episode最后一步的成功率的平均值（局部成功率）
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r


env = gym.make(ENV_NAME,render_mode="human")
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
# agent.load_weights("/Users/qiyangyan/Desktop/IHM_finger/TD3+HER/Pre-trained models/VariableFriction-v0_learned.pth")

def change_friction(action, env):
    _, _, _, _, _ = env.step(action)
    while not env.action_complete():
        _, _, _, _, _ = env.step(action)
    return True


if Train:
    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        count = 0
        for cycle in range(0, MAX_CYCLES):
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                env_dict = env.reset()[0]
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict = env.reset()[0]
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]

                action_array = np.zeros((6, 7))
                action = []
                complete_action = agent.choose_action(state, desired_goal)
                print("Action for the episode:", complete_action)
                action_array[0][0] = complete_action[0]
                action_array[0][6] = np.sign(complete_action[0])
                for i in range(5):
                    action_array[i+1] = action_array[i]
                    action_array[i+1][i+1] = complete_action[i+1]
                    if i == 0 or i == 4:
                        action_array[i + 1][6] = 2
                    else:
                        action_array[i + 1][6] = np.sign(complete_action[i+1])
                # print(action_array)

                '''Pick Up'''
                pick_up = [0, 0, 0, 0, 0, 0, 2]
                print("start picking")
                _, _, _, _, _ = env.step(pick_up)
                while not env.action_complete():
                    _, _, _, _, _ = env.step(pick_up)
                    # let self.pick_up = true
                for _ in range(50):
                    _, _, _, _, _ = env.step(pick_up)
                print("pick up complete --------")

                '''Trajectory generated from reinforcement learning'''
                action_steps = 0
                action = action_array[action_steps]
                friction_change_complete = False
                # print(action_array)
                while action_steps != 6:
                    '''Complete friction change for the move'''
                    if not friction_change_complete:
                        friction_change_complete = change_friction(action,env)
                        action[6] = 0
                        # print("Complete friction change")

                    '''Start movement'''
                    next_env_dict, reward, terminated, done, info = env.step(action)

                    if env.action_complete():
                        action_steps += 1
                        if action_steps == 6:
                            break
                        action = action_array[action_steps] - action_array[action_steps-1]
                        # print(action.shape)
                        # print(action_array[action_steps].shape)
                        action[6] = action_array[action_steps][6]
                        friction_change_complete = False

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action_array[action_steps][:6].copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            agent.store(mb)
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
            print(f"Epoch:{epoch}| "
                  f"Running_reward:{running_reward[-1]:.3f}| "
                  f"EP_reward:{episode_reward:.3f}| "
                  f"Memory_length:{len(agent.memory)}| "
                  f"Duration:{time.time() - start_time:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Success rate:{success_rate:.3f}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
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
    player = Play(env, agent, ENV_NAME, max_episode=100)
    player.evaluate()

