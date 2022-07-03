from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from ounoise import OUNoise
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        ma_cfg: Any = None,
        agent_cfg: Any = None,
    ):
        """Initialize an Agent object
        Parameters
        ----------
        state_size : int
            size of state space
        action_size : int
            size of action space
        n_agents : int
            number of agents
        tau : float, optional
            soft update of target parameters, by default 1e-3
        agent_cfg : Any, optional
            config for agent, by default None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = agent_cfg.seed
        self.n_agents = ma_cfg.n_agents

        self.batch_size = ma_cfg.batch_size

        self.update_every = agent_cfg.update_every
        self.learn_iterations = agent_cfg.learn_iterations

        # soft update
        self.tau = agent_cfg.tau
        self.gamma = agent_cfg.gamma

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size,
            action_size,
            fc_1=agent_cfg.actor.fc_1,
            fc_2=agent_cfg.actor.fc_2,
            seed=self.seed,
        ).to(device)

        self.actor_target = Actor(
            state_size,
            action_size,
            fc_1=agent_cfg.actor.fc_1,
            fc_2=agent_cfg.actor.fc_2,
            seed=self.seed,
        ).to(device)

        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=agent_cfg.actor.lr
        )

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size,
            action_size,
            fc_1=agent_cfg.critic.fc_1,
            fc_2=agent_cfg.critic.fc_2,
            # dropout=agent_cfg.critic.dropout,
            seed=self.seed,
        ).to(device)

        self.critic_target = Critic(
            state_size,
            action_size,
            fc_1=agent_cfg.critic.fc_1,
            fc_2=agent_cfg.critic.fc_2,
            # dropout=agent_cfg.critic.dropout,
            seed=self.seed,
        ).to(device)

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=agent_cfg.critic.lr,
            weight_decay=agent_cfg.critic.weight_decay,
        )

        # Noise process
        self.noise = OUNoise(
            (self.n_agents, self.action_size),
            self.seed,
            mu=agent_cfg.oun.mu,
            theta=agent_cfg.oun.theta,
            sigma=agent_cfg.oun.sigma,
        )

        self.memory = ReplayBuffer(ma_cfg.buffer_size, self.batch_size, self.seed)

        self.t_step = 0

        # initialize local and target to be same
        self.soft_update(self.actor_target, self.actor_local, 1)
        self.soft_update(self.critic_target, self.critic_local, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experiences from each agent in replay memory, and use random sample from buffer to learn."""

        for i in range(self.n_agents):
            self.memory.add(
                states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i]
            )

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            for _ in range(self.learn_iterations):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""

        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.n_agents, self.action_size))

        self.actor_local.eval()

        # get current action per present policy
        with torch.no_grad():
            for i in range(self.n_agents):
                actions[i, :] = self.actor_local(states[i]).cpu().data.numpy()

        self.actor_local.train()

        # include noise
        if add_noise:
            actions += self.noise.sample()

        # clip actions
        actions = np.clip(actions, -1, 1)

        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Parameters
        ----------
        experiences : Tuple[torch.Tensor]
            tuple of (s, a, r, s', done) tuples
        gamma : float
            discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # clip gradients before step
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)

        # negative such that we maximize
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Parameters
        ----------
        local_model : PyTorch model
            weights will be copied from
        target_model : PyTorch model
            weights will be copied to
        tau : float
            interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
