"""Preprocessing utilities for nuScenes trajectory prediction."""

from __future__ import annotations

from typing import List

import torch


def _validate_trajectory(trajectory: torch.Tensor) -> None:
	"""Validate trajectory tensor shape ``(T, 2)``."""
	if not torch.is_tensor(trajectory):
		raise TypeError("trajectory must be a torch.Tensor")
	if trajectory.ndim != 2 or trajectory.shape[1] != 2:
		raise ValueError("trajectory must have shape (T, 2)")


def _validate_positions(positions: torch.Tensor) -> None:
	"""Validate positions tensor shape ``(N, 2)``."""
	if not torch.is_tensor(positions):
		raise TypeError("positions must be a torch.Tensor")
	if positions.ndim != 2 or positions.shape[1] != 2:
		raise ValueError("positions must have shape (N, 2)")


def compute_velocity(trajectory: torch.Tensor) -> torch.Tensor:
	"""Compute per-step velocity from positions.

	Args:
		trajectory: Position tensor with shape ``(T, 2)``.

	Returns:
		Velocity tensor with shape ``(T, 2)`` where:
		``v[t] = pos[t] - pos[t-1]`` and ``v[0] = (0, 0)``.
	"""
	_validate_trajectory(trajectory)

	velocity = torch.zeros_like(trajectory)
	if trajectory.shape[0] > 1:
		velocity[1:] = trajectory[1:] - trajectory[:-1]
	return velocity


def compute_direction(velocity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Normalize velocity vectors into unit directions.

	Args:
		velocity: Velocity tensor with shape ``(T, 2)``.
		eps: Small constant to avoid division by zero.

	Returns:
		Unit direction tensor with shape ``(T, 2)``.
	"""
	_validate_trajectory(velocity)

	norms = torch.norm(velocity, dim=1, keepdim=True)
	safe_denominator = torch.clamp(norms, min=eps)
	return velocity / safe_denominator


def find_neighbors(positions: torch.Tensor, radius: float = 2.0) -> List[List[int]]:
	"""Find neighboring agents within a Euclidean radius.

	Args:
		positions: Agent position tensor with shape ``(N, 2)``.
		radius: Neighborhood radius in meters.

	Returns:
		A list where each element contains neighbor indices for one agent.
	"""
	_validate_positions(positions)
	if radius < 0:
		raise ValueError("radius must be non-negative")

	num_agents = positions.shape[0]
	if num_agents == 0:
		return []

	distances = torch.cdist(positions, positions, p=2)
	within_radius = distances <= radius

	self_mask = torch.eye(num_agents, dtype=torch.bool, device=positions.device)
	within_radius = within_radius & (~self_mask)

	neighbors: List[List[int]] = []
	for i in range(num_agents):
		idxs = torch.nonzero(within_radius[i], as_tuple=False).flatten()
		neighbors.append(idxs.tolist())

	return neighbors


def normalize_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
	"""Shift trajectory so the last point becomes the origin.

	Args:
		trajectory: Position tensor with shape ``(T, 2)``.

	Returns:
		Shifted trajectory tensor with shape ``(T, 2)``.
	"""
	_validate_trajectory(trajectory)
	if trajectory.shape[0] == 0:
		return trajectory.clone()

	anchor = trajectory[-1].unsqueeze(0)
	return trajectory - anchor

