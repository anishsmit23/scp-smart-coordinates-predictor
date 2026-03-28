"""Custom collate function utilities for trajectory prediction."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List

import torch

from utilities.preprocessing import find_neighbors


DEFAULT_NEIGHBOR_RADIUS = 2.0


def _validate_sample(sample: Dict[str, Any]) -> None:
	"""Validate one dataset sample required by trajectory collate function."""
	required_keys = {"past", "future", "agent_id"}
	if not required_keys.issubset(sample.keys()):
		missing = required_keys.difference(sample.keys())
		raise KeyError(f"Sample is missing required keys: {sorted(missing)}")

	past = sample["past"]
	future = sample["future"]
	if not torch.is_tensor(past) or past.ndim != 2 or past.shape[1] < 2:
		raise ValueError("sample['past'] must be a tensor with shape (steps, features>=2)")
	if not torch.is_tensor(future) or future.ndim != 2 or future.shape[1] != 2:
		raise ValueError("sample['future'] must be a tensor with shape (steps, 2)")


def trajectory_collate_fn(
	batch: List[Dict[str, Any]],
	neighbor_radius: float = DEFAULT_NEIGHBOR_RADIUS,
) -> Dict[str, Any]:
	"""Collate trajectory samples into a structured batch.

	Input batch sample format:
		{
			"past": Tensor (past_steps, F),
			"future": Tensor (future_steps, 2),
			"agent_id": int,
		}

	Output format:
		{
			"past": Tensor (batch, past_steps, F),
			"future": Tensor (batch, future_steps, 2),
			"neighbor_indices": list,
		}
	"""
	if len(batch) == 0:
		raise ValueError("Batch is empty")

	for sample in batch:
		_validate_sample(sample)

	past = torch.stack([sample["past"].float() for sample in batch], dim=0)
	future = torch.stack([sample["future"].float() for sample in batch], dim=0)

	current_positions = past[:, -1, :2]
	neighbor_indices = find_neighbors(current_positions, radius=float(neighbor_radius))

	return {
		"past": past,
		"future": future,
		"neighbor_indices": neighbor_indices,
	}


def build_collate_fn(neighbor_radius: float = DEFAULT_NEIGHBOR_RADIUS):
	"""Build a collate function bound to a specific neighbor radius."""
	return partial(trajectory_collate_fn, neighbor_radius=float(neighbor_radius))

