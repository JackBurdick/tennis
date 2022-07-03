import pickle
from collections import deque

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from unityagents import UnityEnvironment

from agent_ma import Agent


class Trainer:
    def __init__(
        self,
        n_episodes=10000,
        max_t=1000,
        target_score=0.5,
        run_headless=True,
        print_setup=True,
        print_every=100,
        window_size=100,
        scores_path="scores.pkl",
        ma_cfg=None,
        agent_cfg=None,
    ):
        """Create an environment and train an agent

        len(env_info.agents) == 2
        action_size = brain.vector_action_space_size == 2
        state = env_info.vector_observations[0] -->
              [ 0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.          0.          0.
                0.          0.          0.          0.         -6.65278625 -1.5
                -0.          0.          6.83172083  6.         -0.          0.        ]
        len(state) == 24

        Parameters
        ----------
        n_episodes : int, optional
            maximum number of training episodes, by default 10000
        max_t : int, optional
            maximum number of timesteps per episode, by default 1000
        target_score : int, optional
            score threshold to terminate training, by default 0.5
        run_headless : bool, optional
            run the environment in headless mode, by default True
        print_setup : bool, optional
            print the environment setup, by default True
        print_every : int, optional
            print the score every n episodes, by default 100
        window_size : int, optional
            window size for the score, by default 100
        scores_path : str, optional
            path to save the scores, by default "scores.pkl"
        ma_cfg : DictConfig, optional
            multi agent configuration, by default None
        agent_cfg : DictConfig, optional
            agent configuration, by default None
        """

        self.n_episodes = n_episodes
        self.max_t = max_t

        self.target_score = target_score
        self.window_size = window_size

        self.scores = []
        self.scores_window = deque(maxlen=self.window_size)
        self.print_every = print_every

        self.save_path_fmt_score = "./params/best_params_{}_"
        self.save_scores_path = scores_path

        # environment
        if run_headless:
            self.env = UnityEnvironment(
                file_name="/home/jackburdick/dev/tennis/Tennis_Linux_NoVis/Tennis.x86_64"
            )
        else:
            raise FileNotFoundError("file not presently included")

        # get the default brain
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[brain_name]
        self.action_size = brain.vector_action_space_size
        _ex_states = env_info.vector_observations
        self.state_size = _ex_states.shape[1]  # from first agent
        self.n_agents = len(env_info.agents)
        if print_setup:
            print(f"state_size: {self.state_size}")
            print(f"action_size: {self.action_size}")
            print(f"example states:\n {_ex_states[0]}")
            print(f"num_agents: {self.n_agents}")

        # create agent
        self.multi_agent = Agent(
            state_size=self.state_size,
            action_size=self.action_size,
            ma_cfg=ma_cfg,
            agent_cfg=agent_cfg,
        )

    def _check_done_save_params(self, e):
        """Check if the environment is solved and save the parameters"""
        if np.mean(self.scores_window) >= self.target_score:
            print(
                f"\nEnv solved in {e:d} episodes!\tAvg Score: {np.mean(self.scores_window):.2f}"
            )
            save_path = self.save_path_fmt_score.format(
                int(np.mean(self.scores_window))
            )
            torch.save(
                self.multi_agent.actor_local.state_dict(),
                f"{save_path}checkpoint_actor.pth",
            )
            torch.save(
                self.multi_agent.critic_local.state_dict(),
                f"{save_path}checkpoint_critic.pth",
            )
            print(f"params saved: {save_path}....")
            # preserve score log
            with open(self.save_scores_path, "wb") as f:
                pickle.dump(self.scores, f)
            return True
        return False

    def _terminal_monitor(self, e):
        """Print the score to the terminal"""
        print(
            f"\rE: {e:5},\t\tAvg Score: {np.mean(self.scores_window) :.5f},\t\tLast Score: {self.scores_window[-1] : .5f}",
            end="",
        )
        if e % self.print_every == 0:
            print(f"\rE: {e:5},\t\tAvg Score: {np.mean(self.scores_window) :.5f}")

    def _unpack_env_info(self, env_info):
        cur_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return cur_states, rewards, dones

    def _update_score(self, scores):
        self.scores_window.append(scores)
        self.scores.append(scores)

    def cleanup(self):
        # close environment
        self.env.close()
        print("env closed")

    def train(self):
        """train the agent and return the scores"""
        brain_name = self.env.brain_names[0]
        # run episode
        for e in range(1, self.n_episodes + 1):
            scores = np.zeros(self.n_agents)
            self.multi_agent.reset()

            env_info = self.env.reset(train_mode=True)[brain_name]

            # set initial states
            states, _, _ = self._unpack_env_info(env_info)
            for t in range(self.max_t):

                # use states to determine action
                actions = self.multi_agent.act(states)

                # send the action to the environment
                env_info = self.env.step(actions)[brain_name]

                # unpack reward and next states
                next_states, rewards, dones = self._unpack_env_info(env_info)

                # record step information to agent, possibly learn
                self.multi_agent.step(states, actions, rewards, next_states, dones)

                # update state
                states = next_states

                scores += rewards  # TODO: #np.array(rewards)
                if any(dones):
                    break

            # update score, display in terminal
            e_score = np.max(scores)
            self._update_score(e_score)
            self._terminal_monitor(e)

            # Maybe save params
            if self._check_done_save_params(e):
                break

        return self.scores


@hydra.main(version_base=None, config_path="conf", config_name="best")
def train(cfg: DictConfig) -> None:
    """Train the agent"""
    trainer = hydra.utils.instantiate(cfg.trainer)
    _ = trainer.train()
    trainer.cleanup()
    return np.mean(trainer.scores_window)


if __name__ == "__main__":
    train()
