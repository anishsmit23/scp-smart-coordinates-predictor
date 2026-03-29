"""Social pooling module for multi-agent trajectory prediction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SocialPooling(nn.Module):
	"""Pool nearby agent features into a social context vector.

	Input:
		- hidden_states: Tensor of shape ``(N, hidden_dim)``
		- neighbor_indices: list of neighbors for each agent
		- positions: Tensor of shape ``(N, 2)`` for grid pooling

	Output:
		- social_features: Tensor of shape ``(N, social_dim)``
	"""

	def __init__(
		self,
		hidden_dim: int = 128,
		social_dim: int = 32,
		pooling_type: str = "mean",
		grid_size: int = 8,
		neighbor_radius: float = 2.0,
	) -> None:
		"""Initialize social pooling layers."""
		super().__init__()
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be a positive integer")
		if social_dim <= 0:
			raise ValueError("social_dim must be a positive integer")
		if grid_size <= 0:
			raise ValueError("grid_size must be a positive integer")
		if neighbor_radius <= 0:
			raise ValueError("neighbor_radius must be a positive value")

		pooling_type = pooling_type.lower().strip()
		if pooling_type not in {"mean", "grid"}:
			raise ValueError("pooling_type must be either 'mean' or 'grid'")

		self.hidden_dim = hidden_dim
		self.social_dim = social_dim
		self.pooling_type = pooling_type
		self.grid_size = grid_size
		self.neighbor_radius = float(neighbor_radius)

		self.neighbor_proj = nn.Linear(hidden_dim, social_dim)
		self.proj = nn.Linear(social_dim, social_dim)
		self.activation = nn.ReLU()
		if self.pooling_type == "grid":
			self.grid_proj = nn.Linear(grid_size * grid_size * social_dim, social_dim)

	def _build_adjacency(
		self,
		neighbor_indices: Sequence[Sequence[int]],
		num_agents: int,
		device: torch.device,
	) -> torch.Tensor:
		"""Convert neighbor list into dense adjacency matrix."""
		if num_agents == 0:
			return torch.zeros((0, 0), dtype=torch.float32, device=device)

		lengths = torch.tensor(
			[len(nbrs) for nbrs in neighbor_indices[:num_agents]],
			dtype=torch.long,
			device=device,
		)
		if lengths.numel() < num_agents:
			pad = torch.zeros(num_agents - lengths.numel(), dtype=torch.long, device=device)
			lengths = torch.cat([lengths, pad], dim=0)

		total_edges = int(lengths.sum().item())
		if total_edges == 0:
			return torch.zeros((num_agents, num_agents), dtype=torch.float32, device=device)

		dst_index = torch.arange(num_agents, device=device, dtype=torch.long).repeat_interleave(lengths)
		src_flat = torch.tensor(
			[src for nbrs in neighbor_indices[:num_agents] for src in nbrs],
			dtype=torch.long,
			device=device,
		)
		valid = (src_flat >= 0) & (src_flat < num_agents)
		if not torch.any(valid):
			return torch.zeros((num_agents, num_agents), dtype=torch.float32, device=device)

		dst_valid = dst_index[valid]
		src_valid = src_flat[valid]
		values = torch.ones(dst_valid.size(0), dtype=torch.float32, device=device)
		adj = torch.sparse_coo_tensor(
			indices=torch.stack([dst_valid, src_valid], dim=0),
			values=values,
			size=(num_agents, num_agents),
			device=device,
		)
		return adj.to_dense()

	def _grid_pool(
		self,
		hidden_states: torch.Tensor,
		neighbor_indices: Sequence[Sequence[int]],
		positions: torch.Tensor,
	) -> torch.Tensor:
		"""Pool neighbor features into an occupancy-style social grid."""
		num_agents = hidden_states.size(0)
		if num_agents == 0:
			return torch.zeros((0, self.social_dim), dtype=hidden_states.dtype, device=hidden_states.device)

		pooled = torch.zeros(
			(num_agents, self.grid_size, self.grid_size, self.social_dim),
			dtype=hidden_states.dtype,
			device=hidden_states.device,
		)
		projected_hidden = self.neighbor_proj(hidden_states)

		cell_width = (2.0 * self.neighbor_radius) / float(self.grid_size)
		for agent_idx in range(num_agents):
			for neighbor_idx in neighbor_indices[agent_idx] if agent_idx < len(neighbor_indices) else []:
				if neighbor_idx < 0 or neighbor_idx >= num_agents:
					continue

				rel = positions[neighbor_idx] - positions[agent_idx]
				rx = float(rel[0].item())
				ry = float(rel[1].item())
				if abs(rx) > self.neighbor_radius or abs(ry) > self.neighbor_radius:
					continue

				gx = int((rx + self.neighbor_radius) / cell_width)
				gy = int((ry + self.neighbor_radius) / cell_width)
				gx = max(0, min(self.grid_size - 1, gx))
				gy = max(0, min(self.grid_size - 1, gy))
				pooled[agent_idx, gy, gx, :] += projected_hidden[neighbor_idx]

		pooled = pooled.reshape(num_agents, self.grid_size * self.grid_size * self.social_dim)
		return self.activation(self.grid_proj(pooled))

	def forward(
		self,
		hidden_states: torch.Tensor,
		neighbor_indices: Sequence[Sequence[int]],
		positions: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Compute social context using mean or grid pooling."""
		if hidden_states.ndim != 2:
			raise ValueError("hidden_states must have shape (N, hidden_dim)")
		if hidden_states.size(1) != self.hidden_dim:
			raise ValueError(
				f"hidden_states second dimension must be {self.hidden_dim}, got {hidden_states.size(1)}"
			)

		num_agents = hidden_states.size(0)
		if self.pooling_type == "grid":
			if positions is None:
				raise ValueError("positions must be provided when pooling_type='grid'")
			if positions.ndim != 2 or positions.size(1) != 2 or positions.size(0) != num_agents:
				raise ValueError("positions must have shape (N, 2) matching hidden_states")
			return self._grid_pool(hidden_states, neighbor_indices, positions)

		projected_hidden = self.neighbor_proj(hidden_states)
		pooled = torch.zeros((num_agents, self.social_dim), dtype=hidden_states.dtype, device=hidden_states.device)

		if num_agents == 0:
			return self.activation(self.proj(pooled))

		adjacency = self._build_adjacency(
			neighbor_indices=neighbor_indices,
			num_agents=num_agents,
			device=hidden_states.device,
		)
		if adjacency.numel() > 0:
			counts = adjacency.sum(dim=1, keepdim=True)
			pooled_sum = adjacency @ projected_hidden
			pooled = torch.where(counts > 0, pooled_sum / counts.clamp_min(1.0), pooled)

		social_features = self.activation(self.proj(pooled))
		return social_features

