"""AREL Time_Agent_Transformer (causal across time, attention across agents).

Adapted from Safe-TSL-DBCT (MARL4DRP/drp_env/reward_model/arel/transformers.py).
Used to assign step-level credit before passing into fψ_path.
"""

import torch
from torch import nn

from src.lare.shared.attention import TransformerBlock, TransformerBlockAgent


class TimeAgentTransformer(nn.Module):
    def __init__(self, emb, heads, depth, seq_length, n_agents,
                 agent=True, dropout=0.0, comp=True, comp_emb_cap=100):
        super().__init__()
        self.comp = comp
        self.n_agents = n_agents
        self.input_emb = emb

        if comp:
            self.comp_emb = min(emb, comp_emb_cap)
            self.compress_input = nn.Linear(emb, self.comp_emb)
            inner_emb = self.comp_emb
        else:
            inner_emb = emb

        self.pos_embedding = nn.Embedding(num_embeddings=seq_length, embedding_dim=inner_emb)

        blocks = []
        for _ in range(depth):
            blocks.append(TransformerBlock(emb=inner_emb, heads=heads,
                                           seq_length=seq_length, mask=True, dropout=dropout))
            if agent:
                blocks.append(TransformerBlockAgent(emb=inner_emb, heads=heads,
                                                    seq_length=seq_length, n_agents=n_agents,
                                                    mask=False, dropout=dropout))
        self.tblocks = nn.Sequential(*blocks)
        self.toreward = nn.Linear(inner_emb, 1)

    def forward(self, x):
        """
        x: (batch, n_agents, t, emb)
        Returns step-wise scalar (batch, n_agents, t).
        """
        b, n_a, t, e = x.size()
        device = x.device

        if self.comp:
            x = self.compress_input(x)
            inner_emb = self.comp_emb
        else:
            inner_emb = self.input_emb

        positions = self.pos_embedding(torch.arange(t, device=device))[None, :, :].expand(b * n_a, t, inner_emb)
        x = x.view(b * n_a, t, inner_emb) + positions
        x = self.tblocks(x)
        x = self.toreward(x).view(b, n_a, t)
        return x
