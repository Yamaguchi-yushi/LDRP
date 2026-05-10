"""Shared attention building blocks for LaRe.

Adapted from Safe-TSL-DBCT (MARL4DRP/drp_env/reward_model/arel/modules.py).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def _mask_upper_triangle(matrices, maskval=float("-inf"), mask_diagonal=False):
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


def _device_of(x):
    return "cuda" if x.is_cuda else "cpu"


class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        if self.mask:
            _mask_upper_triangle(dot)
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fed = self.ff(x)
        x = self.norm2(fed + x)
        x = self.do(x)
        return x


class TransformerBlockAgent(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, n_agents, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.n_a = n_agents
        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        _, t, e = x.size()
        x = x.view(-1, self.n_a, t, e).transpose(1, 2).contiguous().view(-1, self.n_a, e)
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fed = self.ff(x)
        x = self.norm2(fed + x)
        x = self.do(x)
        x = x.view(-1, t, self.n_a, e).transpose(1, 2).contiguous().view(-1, t, e)
        return x
