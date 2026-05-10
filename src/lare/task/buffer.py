"""Episode buffer for LaRe-Task.

Each episode contains a *variable* number of assignment decisions K_i. We store
the per-episode factor matrices (K_i, factor_dim) and the episode-level training
target R_task (= task completion count). Sampling pads up to the batch's max-K.
"""

from collections import deque

import numpy as np


class TaskEpisodeBuffer:
    def __init__(self, capacity, factor_dim):
        self.capacity = int(capacity)
        self.factor_dim = int(factor_dim)
        self.episodes = deque(maxlen=self.capacity)
        self._reset_current()

    def _reset_current(self):
        self._cur_factors = []  # list of (factor_dim,) arrays
        self._cur_return = 0.0  # filled at end_episode

    def add_decision(self, factor_vec):
        """factor_vec: 1D numpy array of length factor_dim."""
        self._cur_factors.append(np.asarray(factor_vec, dtype=np.float32))

    def end_episode(self, r_task):
        if len(self._cur_factors) == 0:
            self._reset_current()
            return False
        factors = np.stack(self._cur_factors, axis=0).astype(np.float32)
        self.episodes.append({
            "factors": factors,                  # (K, factor_dim)
            "k": int(factors.shape[0]),
            "return": float(r_task),
        })
        self._reset_current()
        return True

    def __len__(self):
        return len(self.episodes)

    def sample_batch(self, batch_size, rng=None):
        if len(self.episodes) == 0:
            return None
        rng = rng if rng is not None else np.random
        n = min(batch_size, len(self.episodes))
        idxs = rng.choice(len(self.episodes), size=n, replace=False)
        episodes = [self.episodes[i] for i in idxs]
        max_k = max(ep["k"] for ep in episodes)

        factors = np.zeros((n, max_k, self.factor_dim), dtype=np.float32)
        ks = np.zeros((n,), dtype=np.int64)
        returns = np.zeros((n,), dtype=np.float32)
        for i, ep in enumerate(episodes):
            k = ep["k"]
            factors[i, :k, :] = ep["factors"]
            ks[i] = k
            returns[i] = ep["return"]
        return factors, ks, returns
