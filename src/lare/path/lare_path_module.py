"""LaRe-Path main module: ties encoder, decoder, optional transformer, and buffer.

Usage from the env:
  module = LaRePathModule(env, config)
  on each env.step():
      factors, proxy_rewards = module.step(prev_onehot_position, current_colliding_pairs)
      module.record_step(factors, env_reward_sum=sum(ri_array))
      ...
      if done: module.end_episode()  # may trigger a decoder update

When the decoder has not been trained yet, `proxy_rewards` is None and the env
should fall back to the original reward.

Three operating modes:
  1. use_lare_path=False                                 -> baseline (no LaRe code path)
  2. use_lare_path=True, frozen=False                    -> train decoder online; proxy after first update
  3. use_lare_path=True, frozen=True (pretrained loaded) -> use loaded decoder, no training
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .encoder import (
    FACTOR_NUMBER,
    build_lare_obs_for_agent,
    compute_graph_diameter,
    evaluation_func,
    precompute_edge_info,
)
from .decoder import PathRewardDecoder
from .transformer import TimeAgentTransformer


@dataclass
class LaRePathConfig:
    factor_dim: int = FACTOR_NUMBER
    decoder_hidden_dim: int = 64
    decoder_n_layers: int = 3
    use_transformer: bool = False
    transformer_heads: int = 4
    transformer_depth: int = 2
    transformer_seq_length: int = 100
    buffer_capacity: int = 512
    seq_length: int = 100
    min_buffer: int = 64
    update_freq: int = 32
    batch_size: int = 32
    learning_rate: float = 5e-4
    train_epochs: int = 1
    use_lare_training: bool = True
    # Pretrained-model mode. When frozen=True the decoder is used in eval-only mode
    # and end_episode() will NOT trigger optimizer updates.
    frozen: bool = False
    # Auto-save the decoder after each successful update (training mode only).
    # Either a fixed string path, or a zero-arg callable returning a path
    # (callable form lets the caller embed mutable state like step counts).
    autosave_path: Optional[object] = None


class LaRePathModule:
    def __init__(self, env, config: Optional[LaRePathConfig] = None):
        self.env = env
        self.cfg = config if config is not None else LaRePathConfig()

        self.edge_info_cache = precompute_edge_info(env)
        self.graph_diameter = compute_graph_diameter(env)

        self.factor_dim = self.cfg.factor_dim
        self.n_agents = env.agent_num
        self.seq_length = self.cfg.seq_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decoder = PathRewardDecoder(
            factor_dim=self.factor_dim,
            hidden_dim=self.cfg.decoder_hidden_dim,
            n_layers=self.cfg.decoder_n_layers,
        ).to(self.device)

        self.use_transformer = bool(self.cfg.use_transformer)
        if self.use_transformer:
            self.transformer = TimeAgentTransformer(
                emb=self.factor_dim,
                heads=self.cfg.transformer_heads,
                depth=self.cfg.transformer_depth,
                seq_length=self.cfg.transformer_seq_length,
                n_agents=self.n_agents,
                agent=True,
            ).to(self.device)
            params = list(self.decoder.parameters()) + list(self.transformer.parameters())
        else:
            self.transformer = None
            params = list(self.decoder.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss(reduction="mean")

        from .buffer import PathEpisodeBuffer
        self.buffer = PathEpisodeBuffer(
            capacity=self.cfg.buffer_capacity,
            seq_length=self.seq_length,
            n_agents=self.n_agents,
            factor_dim=self.factor_dim,
        )

        self.is_trained = False
        self.is_pretrained = False
        self.episode_count = 0
        self.update_count = 0
        self.last_loss = None
        self.use_lare_training = bool(self.cfg.use_lare_training)
        self.frozen = bool(self.cfg.frozen)

    def compute_factors(self, prev_onehot_position, current_colliding_pairs):
        """Compute the (n_agents, factor_dim) factor matrix for the current step.

        prev_onehot_position: (n_agents, n_nodes) array of pre-step positions.
        current_colliding_pairs: list of agent-id pairs that collided this step (or None).
        """
        rows = []
        for i in range(self.n_agents):
            obs_row = build_lare_obs_for_agent(
                self.env, i, self.edge_info_cache, self.graph_diameter,
                prev_onehot_position, current_colliding_pairs,
            )
            rows.append(obs_row)
        batch_obs = np.stack(rows, axis=0)
        factor_list = evaluation_func(batch_obs)
        factor_arr = np.concatenate(factor_list, axis=-1).astype(np.float32)
        return factor_arr

    def proxy_rewards(self, factors_per_agent):
        """Decoder forward for one step. Returns shape (n_agents,) numpy array.

        Returns None if the decoder has not been trained yet.
        """
        if not self.is_trained:
            return None
        with torch.no_grad():
            self.decoder.eval()
            z = torch.from_numpy(factors_per_agent).float().to(self.device)
            r_hat = self.decoder(z).squeeze(-1)
        return r_hat.detach().cpu().numpy()

    def record_step(self, factors_per_agent, env_reward_sum):
        self.buffer.add_step(factors_per_agent, env_reward_sum)

    def end_episode(self):
        self.buffer.end_episode()
        self.episode_count += 1

        # Frozen / pretrained mode: never update the decoder.
        if self.frozen:
            return

        if (
            len(self.buffer) >= self.cfg.min_buffer
            and self.episode_count % max(1, self.cfg.update_freq) == 0
        ):
            self._update()

    def _update(self):
        if self.frozen:
            return
        sample = self.buffer.sample_batch(self.cfg.batch_size)
        if sample is None:
            return
        factors_np, lengths_np, returns_np = sample

        factors = torch.from_numpy(factors_np).to(self.device)
        lengths = torch.from_numpy(lengths_np).to(self.device)
        returns = torch.from_numpy(returns_np).to(self.device)

        b, n_a, t, _ = factors.shape
        time_idx = torch.arange(t, device=self.device)[None, :]
        mask = (time_idx < lengths[:, None]).float()
        mask = mask.unsqueeze(1).expand(b, n_a, t)

        for _ in range(max(1, self.cfg.train_epochs)):
            self.decoder.train()
            if self.transformer is not None:
                self.transformer.train()
                z = self.transformer(factors).unsqueeze(-1)
                r_hat = self.decoder(z.squeeze(-1).unsqueeze(-1).expand(b, n_a, t, self.factor_dim))
            else:
                r_hat = self.decoder(factors)
            r_hat = r_hat.squeeze(-1)
            r_hat_masked = r_hat * mask
            pred_return = r_hat_masked.sum(dim=[1, 2])

            loss = self.loss_fn(pred_return, returns)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.is_trained = True
        self.update_count += 1
        self.last_loss = float(loss.detach().cpu().item())

        if self.cfg.autosave_path:
            try:
                target = self.cfg.autosave_path
                if callable(target):
                    target = target()
                self.save_model(target)
            except Exception as e:
                print(f"[LaRe-Path] autosave failed: {e}")

    def save_model(self, path):
        """Persist decoder (and optional transformer) weights + minimal metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "decoder_state_dict": self.decoder.state_dict(),
            "transformer_state_dict": (
                self.transformer.state_dict() if self.transformer is not None else None
            ),
            "factor_dim": self.factor_dim,
            "n_agents": self.n_agents,
            "use_transformer": self.use_transformer,
            "update_count": self.update_count,
            "last_loss": self.last_loss,
        }
        torch.save(payload, path)

    def load_model(self, path, freeze=True):
        """Load decoder weights from disk. With freeze=True (default) further
        training is disabled and the proxy reward is used immediately."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LaRe-Path model file not found: {path}")
        payload = torch.load(path, map_location=self.device)

        saved_factor_dim = int(payload.get("factor_dim", self.factor_dim))
        if saved_factor_dim != self.factor_dim:
            raise ValueError(
                f"factor_dim mismatch: model={saved_factor_dim}, current={self.factor_dim}"
            )

        self.decoder.load_state_dict(payload["decoder_state_dict"])
        self.decoder.eval()
        if self.transformer is not None and payload.get("transformer_state_dict") is not None:
            self.transformer.load_state_dict(payload["transformer_state_dict"])
            self.transformer.eval()

        self.is_trained = True
        self.is_pretrained = True
        self.frozen = bool(freeze)
        self.update_count = int(payload.get("update_count", 0))
        self.last_loss = payload.get("last_loss", None)
