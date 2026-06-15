from __future__ import annotations

import math

import torch
from torch import nn


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        n_nodes = x.size(0)
        projected = self.proj(x).view(n_nodes, self.heads, self.out_dim)
        src_index, dst_index = edge_index
        src_h = projected[src_index]
        dst_h = projected[dst_index]

        scores = (src_h * self.attn_src).sum(-1) + (dst_h * self.attn_dst).sum(-1)
        scores = self.leaky_relu(scores)
        if edge_weight is not None:
            scores = scores + edge_weight.unsqueeze(-1).log1p()

        alpha = torch.zeros_like(scores)
        for node_id in range(n_nodes):
            mask = dst_index == node_id
            if mask.any():
                alpha[mask] = torch.softmax(scores[mask], dim=0)
        alpha = self.dropout(alpha)

        out = torch.zeros((n_nodes, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        weighted = src_h * alpha.unsqueeze(-1)
        out.index_add_(0, dst_index, weighted)
        return out.reshape(n_nodes, self.heads * self.out_dim)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = max(16, in_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((0, 0), device=x.device)
        scores = self.scorer(x).squeeze(-1)
        batch_size = int(batch.max().item()) + 1
        pooled = []
        for graph_id in range(batch_size):
            mask = batch == graph_id
            weights = torch.softmax(scores[mask], dim=0)
            pooled.append((weights.unsqueeze(-1) * x[mask]).sum(dim=0))
        return torch.stack(pooled, dim=0)
