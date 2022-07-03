[//]: # (Image References)

[training_plot]: ./assets/training_plot.png "training_plot"
[smoothed_training_plot]: ./assets/smoothed_training_plot.png "training_plot"


# Tennis (Unity tennis environment)

## Output

```log
E:   100,               Avg Score: 0.01230,             Last Score:  0.00000
E:   200,               Avg Score: 0.04400,             Last Score:  0.10000
E:   300,               Avg Score: 0.07150,             Last Score:  0.10000
E:   400,               Avg Score: 0.10750,             Last Score:  0.10000
E:   500,               Avg Score: 0.12230,             Last Score:  0.10000
E:   600,               Avg Score: 0.12710,             Last Score:  0.10000
E:   700,               Avg Score: 0.40070,             Last Score:  0.60000
E:   720,               Avg Score: 0.51300,             Last Score:  2.60000
Env solved in 720 episodes!     Avg Score: 0.51
params saved: ./params/best_params_0_...
env closed
```


## Context

The goal environment information can be found in the main [readme.md](./README.md)


## Description

Below is an overview of:
 - loss plots
 - model
 - agent
 - config settings

### Loss Plots

The raw loss plot (below) is difficult to interpret, but shows a slow, then
rapid increase in performance

![Reward over time][training_plot]

The smoothed version confirm the slow, then rapidly increasing rise in
performance, followed by a dip

![Smoothed reward over time][smoothed_training_plot]

### Model (Actor/Critic)

The models aren't terribly exciting, largely standard DNNs.

#### Actor

The core actor model consists of a number of linear layers of pre-specified
units (determined by a quick optuna run) in this case `[256, 384]`. Deeper
models (e.g. with more layers), subjectively performed worse in my quick
experiments, though I find it hard to believe this would hold true with more
testing.

```python
self.fc_1 = nn.Linear(state_size, fc_1)
self.bn1 = nn.BatchNorm1d(fc_1)

self.fc_2 = nn.Linear(fc_1, fc_2)
self.bn2 = nn.BatchNorm1d(fc_2)

# output
self.fc_out = nn.Linear(fc_2, action_size)
```

Which are subsequently called in the following order:

```python
def forward(self, state):
    if states.dim() == 1:
        states = states.unsqueeze(0)
    x = F.relu(self.bn1(self.fc_1(states)))
    x = F.relu(self.bn2(self.fc_2(x)))
    x = self.fc_out(x)
    return torch.tanh(x)
```


#### Critic

The critic model was similar to the actor. With layers of size: `fc_1: 384 fc_2:
64` (which were determined by optuna)

```python
self.bn1 = nn.BatchNorm1d(fc_1)
self.bn2 = nn.BatchNorm1d(fc_2)

# input and hidden
self.fc_1 = nn.Linear(state_size, fc_1)
self.fc_2 = nn.Linear(fc_1 + action_size, fc_2)

# output
self.fc_out = nn.Linear(fc_2, 1)
```

The difference from the actor model, is that in this case we have multiple
inputs (state and action) and so we concatenated a projection of the state to
the action before using the rest of the architecture. This design was a hold
over from a prior exercise and seemed to work better than other design choices
(without a ton of experimentation)

```python
def forward(self, states, actions):
    """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

    if states.dim() == 1:
        states = states.unsqueeze(0)

    x = F.relu(self.bn1(self.fc_1(states)))
    x = torch.cat((x, actions), dim=1)
    x = F.relu(self.bn2(self.fc_2(x)))
    out = self.fc_out(x)
    return out
```

### Agent

The main algorithm was based on `DDPG` ([Deep deterministic Policy
Gradient](https://arxiv.org/abs/1509.02971v6)). Some overview blogs can be found
[here](https://saashanair.com/blog/blog-posts/deep-deterministic-policy-gradient-ddpg-how-does-the-algorithm-work),
[here](https://keras.io/examples/rl/ddpg_pendulum/), and
[here](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b).

The agent consists of a couple components worth mentioning:
 - Experience Replay (not prioritized)
 - Local and Target network (and corresponding soft update)
 - OUNoise


#### Experience Replay Buffer

A standard `ReplayBuffer` class described in [replay_buffer.py](./replay_buffer.py) was used.


#### Local and Target network (and corresponding soft update)

Select best action with one set of params and evaluate that action with a
different set of parameters, this is best described here: [Deep Reinforcement
Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).

Update procedure in pytorch:
```python
# tau: interpolation parameter

# iterate params and update target params with tau 
# regulated combination of local and target
for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    target_param.data.copy_(
        tau * local_param.data + (1.0 - tau) * target_param.data
    )
```

The target networks are initialized to tbe the same as the local networks

```python
# initialize local and target to be same
self.soft_update(self.actor_target, self.actor_local, 1)
self.soft_update(self.critic_target, self.critic_local, 1)
```


#### OUNoise

Different than my other implementations, I used the implementation
[here](https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab)
to add the `dt` param

```python
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

```

Noise was then added every time

```python
with torch.no_grad():
    for i in range(self.n_agents):
        actions[i, :] = self.actor_local(states[i]).cpu().data.numpy()

    self.actor_local.train()

    # include noise
    if add_noise:
        actions += self.noise.sample()
```

### Config

```yaml
trainer:
  _target_: trainer.Trainer
  n_episodes: 1800
  max_t: 2000
  run_headless: true
  print_setup: false
  print_every: 100
  target_score: 0.5
  window_size: 100
  scores_path: scores.pkl
  ma_cfg:
    n_agents: 2
    batch_size: 512
    buffer_size: 100000
  agent_cfg:
    actor:
      lr: 0.0007587945476302646
      fc_1: 256
      fc_2: 384
    critic:
      lr: 0.00024039506830258236
      weight_decay: 0.0
      fc_1: 384
      fc_2: 64
    learn_iterations: 15
    update_every: 10
    gamma: 0.99
    tau: 0.001
    oun:
      mu: 0.0
      theta: 0.15
      sigma: 0.2
    seed: 42
```

## Future Work

Below are a few ideas related to improving performance

### Improving the Agent

Other Algorithms
> There are other algorithms that could be attempted. but MADDPG interests me most
> - Multi-Agent Actor-Critic for Mixed Cooperative-CompetitiveEnvironments
>   [MADDPG](https://arxiv.org/abs/1706.02275). Some additional information can
>   be found [here](https://www.youtube.com/watch?v=KMt2eCHO9io),
>   [here](https://towardsdatascience.com/openais-multi-agent-deep-deterministic-policy-gradients-maddpg-9d2dad34c82)

"Prioritized" Replay Buffer?
> I'd like to think about which episodes are being saved to the replay buffer,
> rather than any/all episodes. Some episodes are likely more useful than
> others. [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) looks
> like a promising place to start

More advanced models / Stability?
> I feel like all my standard DL intuitions aren't super useful in RL. I don't
> understand this yet. For example, using "better" models doesn't seem to lead
> to "better" results. But I still need more experience here.


## Implementation

Hydra and optuna were used to "optimize" some of the training params. However,
not many runs were actually performed and could be done more effectively in the
future.

## Challenges

Notice anything strange here?

```python
def forward(self, state):
    if states.dim() == 1:
        states = states.unsqueeze(0)
    x = F.relu(self.bn1(self.fc_1(states)))
    x = F.relu(self.bn2(self.fc_2(x)))
    x = F.relu(self.fc_out(x))
    return torch.tanh(x)
```

Look at the activations:

```python
def forward(self, state):
    if states.dim() == 1:
        states = states.unsqueeze(0)
    x = F.relu(self.bn1(self.fc_1(states)))
    x = F.relu(self.bn2(self.fc_2(x)))

    # Here!
    x = F.relu(self.fc_out(x)) # <<<< relu!!

    return torch.tanh(x)
```

The correct implementation should be:

```python
def forward(self, state):
    if states.dim() == 1:
        states = states.unsqueeze(0)
    x = F.relu(self.bn1(self.fc_1(states)))
    x = F.relu(self.bn2(self.fc_2(x)))
    x = self.fc_out(x)
    return torch.tanh(x)
```

This mistake drove me a bit mad. This likely would have been caught much sooner
if I were training the agents in non-headless mode.