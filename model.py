import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_1: int = 256,
        fc_2: int = 256,
        seed: int = 42,
    ):
        """Create and initialize model layers
        Parameters
        ----------
        state_size : int
            size of the state vector
        action_size : int
            size of the action vector
        fc_1 : int, optional
            size of the first hidden layer, by default 256
        fc_2 : int, optional
            size of the second hidden layer, by default 256
        seed : int
            random seed, by default 42
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_1 = nn.Linear(state_size, fc_1)
        self.bn1 = nn.BatchNorm1d(fc_1)

        self.fc_2 = nn.Linear(fc_1, fc_2)
        self.bn2 = nn.BatchNorm1d(fc_2)

        # output
        self.fc_out = nn.Linear(fc_2, action_size)

        # initialize layers
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""

        if states.dim() == 1:
            states = states.unsqueeze(0)
        x = F.relu(self.bn1(self.fc_1(states)))
        x = F.relu(self.bn2(self.fc_2(x)))
        x = self.fc_out(x)
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_1: int = 256,
        fc_2: int = 256,
        seed: int = 42,
    ):
        """Create and initialize model layers
        Parameters
        ----------
        state_size : int
            size of the state vector
        action_size : int
            size of the action vector
        fc_1 : int, optional
            number of nodes in the first dense layer, by default 256
        fc_2 : int, optional
            number of nodes in the second dense layer, by default 256
        seed : int, optional
            random seed, by default 42
        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(fc_1)
        self.bn2 = nn.BatchNorm1d(fc_2)

        # input and hidden
        self.fc_1 = nn.Linear(state_size, fc_1)
        self.fc_2 = nn.Linear(fc_1 + action_size, fc_2)

        # output
        self.fc_out = nn.Linear(fc_2, 1)

        # initialize layers
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        if states.dim() == 1:
            states = states.unsqueeze(0)

        x = F.relu(self.bn1(self.fc_1(states)))
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.bn2(self.fc_2(x)))
        out = self.fc_out(x)
        return out
