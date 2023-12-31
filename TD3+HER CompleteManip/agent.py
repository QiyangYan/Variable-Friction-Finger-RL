import copy

import torch
from torch import from_numpy, device
import numpy as np
from models import Actor, Critic
from memory import Memory
from torch.optim import Adam
from mpi4py import MPI
from normalizer import Normalizer
import torch.nn.functional as F


class Agent:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.98):
        self.device = device("cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env

        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.q1 = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.q2 = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        # self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals)
        # self.q1 = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals)
        # self.q2 = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals)

        self.sync_networks(self.actor)
        self.sync_networks(self.q1)
        self.sync_networks(self.q2)
        self.actor_target = copy.deepcopy(self.actor)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.tau = tau
        self.gamma = gamma

        self.capacity = capacity
        self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.q1_optim = Adam(self.q1.parameters(), self.critic_lr)
        self.q2_optim = Adam(self.q2.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

        self.policy_noise = 0.2
        self.noise_clip = 0.5

    def to_cpu(self):
        self.actor_target.to('cpu')
        self.q1_target.to('cpu')
        self.q2_target.to('cpu')
        self.q1.to('cpu')
        self.q2.to('cpu')

    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(self.device)
            # x = from_numpy(x).float().to('cpu')
            action = self.actor(x)[0].cpu().data.numpy()

        if train_mode:
            action += 0.2 * np.random.randn(self.n_actions)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=self.n_actions)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action

    def store(self, mini_batch):
        for batch in mini_batch:
            self.memory.add(batch)
        self._update_normalizer(mini_batch)

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    # def to_gpu(self):
    #     self.actor_target.to(device('mps'))
    #     self.q1_target.to(device('mps'))
    #     self.q2_target.to(device('mps'))
    #     self.q1.to(device('mps'))
    #     self.q2.to(device('mps'))

    def train(self):
        states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)
        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_inputs = torch.Tensor(next_inputs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        # print(self.device)

        # inputs = torch.Tensor(inputs).to('mps')
        # rewards = torch.Tensor(rewards).to('mps')
        # next_inputs = torch.Tensor(next_inputs).to('mps')
        # actions = torch.Tensor(actions).to('mps')

        self.actor_target.to(self.device)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_inputs)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(self.action_bounds[0], self.action_bounds[1])

            target_q1 = self.q1_target.forward(next_inputs, next_action)
            target_q2 = self.q2_target.forward(next_inputs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_returns = rewards + self.gamma * target_q.detach()
            target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)

        eval_q1 = self.q1(inputs, actions)
        eval_q2 = self.q2(inputs, actions)
        q1_loss = F.mse_loss(eval_q1, target_returns)
        q2_loss = F.mse_loss(eval_q2, target_returns)
        critic_loss = q1_loss + q2_loss

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.sync_grads(self.q1)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.sync_grads(self.q2)
        self.q2_optim.step()

        a = self.actor(inputs)
        actor_loss = -self.q1(inputs, a).mean()
        actor_loss += a.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.sync_grads(self.actor)
        self.actor_optim.step()

        return actor_loss.item(), critic_loss.item()

    def save_weights(self,env):
        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "state_normalizer_mean": self.state_normalizer.mean,
                    "state_normalizer_std": self.state_normalizer.std,
                    "goal_normalizer_mean": self.goal_normalizer.mean,
                    "goal_normalizer_std": self.goal_normalizer.std}, f"./Pre-trained models/{env}_1.pth")

    def load_weights(self,path):

        checkpoint = torch.load(path)
        actor_state_dict = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state_dict)
        state_normalizer_mean = checkpoint["state_normalizer_mean"]
        self.state_normalizer.mean = state_normalizer_mean
        state_normalizer_std = checkpoint["state_normalizer_std"]
        self.state_normalizer.std = state_normalizer_std
        goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
        self.goal_normalizer.mean = goal_normalizer_mean
        goal_normalizer_std = checkpoint["goal_normalizer_std"]
        self.goal_normalizer.std = goal_normalizer_std

    def set_to_eval_mode(self):
        self.actor.eval()
        # self.critic.eval()

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.q1, self.q1_target, self.tau)
        self.soft_update_networks(self.q2, self.q2_target, self.tau)


    def _update_normalizer(self, mini_batch):
        states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    @staticmethod
    def sync_networks(network):
        comm = MPI.COMM_WORLD
        flat_params = _get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        _set_flat_params_or_grads(network, flat_params, mode='params')

    @staticmethod
    def sync_grads(network):
        flat_grads = _get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grads = np.zeros_like(flat_grads)
        comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
        _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
