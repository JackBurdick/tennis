import copy
import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process.
    e.g. https://arxiv.org/pdf/1509.02971.pdf

    https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = 1e-2
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.array([np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state
