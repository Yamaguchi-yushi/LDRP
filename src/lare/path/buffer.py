"""Episode buffer for LaRe-Path.

Stores per-step latent factors for each agent for one episode at a time, then
appends the completed episode (padded to a fixed seq length) to a fixed-size
ring buffer along with R_path (the episode-level training target).
"""

from collections import deque

import numpy as np


class PathEpisodeBuffer:
    def __init__(self, capacity, seq_length, n_agents, factor_dim):
        self.capacity = int(capacity)
        self.seq_length = int(seq_length)
        self.n_agents = int(n_agents)
        self.factor_dim = int(factor_dim)
        self.episodes = deque(maxlen=self.capacity)
        self._reset_current()

    def _reset_current(self):
        self._cur_factors = np.zeros((self.n_agents, self.seq_length, self.factor_dim), dtype=np.float32)
        self._cur_len = 0
        self._cur_return = 0.0

    def add_step(self, factors_per_agent, env_reward_sum):
        """factors_per_agent: array (n_agents, factor_dim).

        env_reward_sum: scalar - sum of per-agent rewards at this step.
        """
        if self._cur_len >= self.seq_length:
            return
        self._cur_factors[:, self._cur_len, :] = factors_per_agent
        self._cur_len += 1
        self._cur_return += float(env_reward_sum)

    def end_episode(self):
        if self._cur_len == 0:
            self._reset_current()
            return
        self.episodes.append({
            "factors": self._cur_factors.copy(),
            "length": int(self._cur_len),
            "return": float(self._cur_return),
        })
        self._reset_current()

    def __len__(self):
        return len(self.episodes)

    def sample_batch(self, batch_size, rng=None):
        if len(self.episodes) == 0:
            return None
        rng = rng if rng is not None else np.random
        n = min(batch_size, len(self.episodes))
        idxs = rng.choice(len(self.episodes), size=n, replace=False)
        factors = np.stack([self.episodes[i]["factors"] for i in idxs], axis=0)
        lengths = np.array([self.episodes[i]["length"] for i in idxs], dtype=np.int64)
        returns = np.array([self.episodes[i]["return"] for i in idxs], dtype=np.float32)
        return factors, lengths, returns
