[//]: # (Image References)

[trained_tennis]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Tennis

### Introduction

[Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment

![Trained Agent][trained_tennis]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Environment Information

Information adapted from [here](https://github.com/deepanshut041/Reinforcement-Learning/tree/master/mlagents/02_tennis) which may not be 100% accurate for the slightly altered version of the env used here.

- Set-up: Two-player game where agents control rackets to hit a ball over the
  net.
- Goal: The agents must hit the ball so that the opponent cannot hit a valid return.
- Agents: The environment contains two agent with same Behavior Parameters.After training you can set the `Behavior Type` to `Heuristic Only` on one of the Agent's Behavior Parameters to play against your trained model.
- Agent Reward Function (independent):
  - +1.0 To the agent that wins the point. An agent wins a point by preventing
   the opponent from hitting a valid return.
  - -1.0 To the agent who loses the point.
- Behavior Parameters:
  - Vector Observation space: 9 variables corresponding to position, velocity
    and orientation of ball and racket.
  - Vector Action space: (Continuous) Size of 3, corresponding to movement
    toward net or away from net, jumping and rotation.
  - Visual Observations: None
- Float Properties: Three
  - gravity: Magnitude of gravity
    - Default: 9.81
    - Recommended Minimum: 6
    - Recommended Maximum: 20
  - scale: Specifies the scale of the ball in the 3 dimensions (equal across the three dimensions)
    - Default: .5
    - Recommended Minimum: 0.2
    - Recommended Maximum: 5

### Goal

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Getting Started (Dependencies)

- Unity Ml Agents
- PyTorch
- numpy
- matplotlib
- hydra (`pip install hydra-core --upgrade`)

Optional
- Optuna (`pip install optuna`)
- Optuna sweeper (`pip install hydra-optuna-sweeper --upgrade`)
    - More [information](https://hydra.cc/docs/plugins/optuna_sweeper/)


### Environment Instructions

To set up the environment please see [unity_instructions](./unity_instructions.md)

The main entry is `trainer.py` and any config can be changed by either modifying `/conf/best.yaml` or `/conf/config.yaml` (or creating a new config) and specifying the location in trainer:

```python
@hydra.main(version_base=None, config_path="conf", config_name="best")
def train(cfg: DictConfig) -> None:
    """Train the agent"""
    trainer = hydra.utils.instantiate(cfg.trainer)
    _ = trainer.train()
    trainer.cleanup()
    return np.mean(trainer.scores_window)
```


### Contents

- Instructions
    - [unity_instructions.md](./unity_instructions.md)
        - Information about the unity environment
- Configs
    - [config.yaml](./conf/config.yaml)
        - main configuration file
    - [best.yaml](./conf/best.yaml)
        - configuration file containing 'best' params (over only 10 runs)
- Model+Training Code
    - [trainer.py](./trainer.py)
        - main entry and convenience wrapper to execute training of an agent in the environment
    - [agent_ma.py](./agent_ma.py)
        - The convenience wrapper that uses the model (specified below) to learn and interact with the environment
    - [model.py](./model.py)
        - The model used to predict actions from the environment
    - [ounoise.py](./ounoise.py)
        - Ornstein-Uhlenbeck process
     [replay_buffer.py](./replay_buffer.py)
        - Replay Buffer
- Best params
    - [/params/best_params_0_checkpoint_actor.pth](./params/best_params_0_checkpoint_actor.pth)
        - model weights saved from a trained actor
    - [/params/best_params_0_checkpoint_critic.pth](./params/best_params_0_checkpoint_critic.pth)
        - model weights saved from a trained critic
- Final Scores
    - [scores.pkl](./scores.pkl)
        - log of scores over time during training run
- Notebooks
    - [visualize.ipynb](./visualize.ipynb)
        - visualize hyperparam information and scores
    - [explore.ipynb](./explore.ipynb)
        - explore the environment, show use of python api for random movements
- Report
    - [report.md](./report.md)
        - write up of the model/agent