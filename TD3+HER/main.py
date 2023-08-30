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

ENV_NAME = "VariableFriction-v0"
INTRO = False
Train = False
Play_FLAG = True
MAX_EPOCHS = 50
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

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'

env = gym.make(ENV_NAME,render_mode="human")
# env.seed(MPI.COMM_WORLD.Get_rank())
gymnasium.utils.seeding.np_random(MPI.COMM_WORLD.Get_rank())
random.seed(MPI.COMM_WORLD.Get_rank())
np.random.seed(MPI.COMM_WORLD.Get_rank())
torch.manual_seed(MPI.COMM_WORLD.Get_rank())
agent = Agent(n_states=state_shape,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(env))
agent.load_weights("/Users/qiyangyan/Desktop/TD3+HER/Pre-trained models/VariableFriction-v0_1.pth")

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

        pick_up()

        last_f = 0
        for t in range(50):
            with torch.no_grad():
                action = agent_.choose_action(s, g, train_mode=False)

            # print("friction changing")
            if action[1] != last_f:
                friction_action1 = [2, 0]
                friction_action2 = [2, action[1]]  # 2 here is the friction change indicator
                complete = change_friction(np.array(friction_action1), env)
                if complete == False:
                    # print("terminated during friction changing")
                    break
                complete = change_friction(np.array(friction_action2), env)
                if complete == False:
                    # print("terminated during friction changing")
                    break
                last_f = action[1].copy()
            else:
                last_f = last_f
            # print("friction change complete")

            observation_new, r, terminated, _, info_ = env_.step(action)
            if terminated:
                # print(terminated)
                break

            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r

        total_success_rate.append(per_success_rate[-1])
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    # print(total_success_rate)
    total_success_rate = np.array(total_success_rate,dtype=object)
    local_success_rate = np.mean(total_success_rate)
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r[-1], ep_r


def change_friction(action, env):
    # print("--------------changing friction")
    _, _, _, _, _ = env.step(action)
    # step = 0
    while not env.action_complete():
        # step += 1
        # if step % 1000 == 0:
        #     print("changing friction for action:", action)
        _, _, terminated, _, _ = env.step(action)
        if terminated == True:
            return False # false meaning not complete
    _, _, _, _, _ = env.step(action)
    # print("completed change friction")
    return True


def pick_up():
    '''Pick Up'''
    pick_up = [1.05, 2]
    # print("start picking")
    _, _, _, _, _ = env.step(np.array(pick_up))
    while not env.action_complete():
        _, _, _, _, _ = env.step(np.array(pick_up))
        # let self.pick_up = true
    for _ in range(50):
        _, _, _, _, _ = env.step(np.array(pick_up))
    # print("pick up complete --------")


def adjust_torque(env, lower):
    env.adjust_torque(lower)


def action_preprocess(action):
    friction_state = action[1]
    '''sliding and rotation'''
    # if friction_state >= 0.33:
    #     friction_state = 1
    # elif friction_state <= -0.33:
    #     friction_state = -1
    # elif -0.33 < friction_state < 0.33:
    #     friction_state = 0
    # else:
    #     print(friction_state)
    #     assert(friction_state,2) # 随便写个东西来报错
    '''only sliding'''
    if friction_state >= 0:
        friction_state = 1
    elif friction_state < 0:
        friction_state = -1
    else:
        print(friction_state)
        assert friction_state==2 # 随便写个东西来报错
    action[1] = friction_state
    return action


if Train:
    success_list = []
    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES):
            if cycle % 10 == 0:
                print("cycle:", cycle)
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            count = 0
            for episode in range(MAX_EPISODES):
                # print(episode)
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
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.03:
                    env_dict = env.reset()[0]
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]

                '''picking'''
                pick_up()

                last_f = 0
                # print("-------------start-----------")
                for t in range(50):
                    action = agent.choose_action(state, desired_goal)
                    action = action_preprocess(action)  # convert continuous friction to discrete
                    '''change friction first'''
                    # print("friction changing")
                    if action[1] != last_f:
                        friction_action1 = [2,0]
                        friction_action2 = [2, action[1]]  # 2 here is the friction change indicator
                        complete = change_friction(np.array(friction_action1), env)
                        if complete == False:
                            # print("terminated during friction changing")
                            break
                        complete = change_friction(np.array(friction_action2), env)
                        if complete == False:
                            # print("terminated during friction changing")
                            break
                        last_f = action[1].copy()
                    else:
                        last_f = last_f
                    # print("friction change complete")
                    # time.sleep(2)

                    '''take step'''
                    # print(action)
                    next_env_dict, reward, terminated, truncated, info = env.step(action)
                    # print("terminated:", terminated, "truncated", truncated)
                    if terminated:
                        # print(terminated)
                        break

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                    # time.sleep(2)

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            # print("agent store")
            agent.store(mb)
            for n_update in range(num_updates):
                # print("train")
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
            print("--------------------------")
            print("Epoch: ", epoch)
            # print(running_reward)
            # print(f"Running_reward:{running_reward.item():.3f}| ")
            print("EP_reward: ",episode_reward)
            print("Memory_length: ",len(agent.memory))
            print("Duration: ",time.time() - start_time)
            print(f"Actor_Loss: ",actor_loss)
            print(f"Critic_Loss: ",critic_loss)
            print(f"Success rate: ",success_rate) # success here checks the end position of object for the episode
            print(f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            print("--------------------------")

            success_list.append(success_rate)
            np.save('success_list.npy', np.array(success_list))

            # print(epoch,len(agent.memory), time.time() - start_time, actor_loss, critic_loss)
            # print(episode_reward)
            # print(success_rate)
            # print(f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")

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
    address_list = [
        # "/Users/qiyangyan/Desktop/TD3+HER/Pre-trained models/VariableFriction-v0_after46epoch_smallerSuccessRegion.pth"
        # "/Users/qiyangyan/Desktop/TD3+HER/Pre-trained models/VariableFriction-v0_after15epoch_niceProgress.pth"
        "/Users/qiyangyan/Desktop/TD3+HER/Pre-trained models/VariableFriction-v0_1.pth"
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_1_stuck.pth"
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_1_goodPerformanceFrom8.26Night.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_20_epoch.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_20_epoch_8.26.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_30_epoch.pth",
        # "/Users/qiyangyan/Desktop/IHM_finger/TD3+HER-withFramework_onlySlide/Pre-trained models/VariableFriction-v1_trained_75_epoch.pth"
    ]

    for address in address_list:
        agent.load_weights(address)
        max_episode = 100
        total_success_rate = 0
        for index in range(max_episode):
            # if index % 20 == 0:
            print(index)
            success_rate, _, _ = eval_agent(env, agent)
            print(success_rate)
            total_success_rate += success_rate
        print(f"Success rate:{total_success_rate / 100:.3f}| ")


    # player = Play(env, agent, ENV_NAME, max_episode=100)
    # player.evaluate()
