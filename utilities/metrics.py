"""Metrics for trajectory prediction."""

from __future__ import annotations

from typing import Tuple

import torch


def _to_batched_trajectory(x: torch.Tensor, name: str) -> torch.Tensor:
	"""Ensure trajectory tensor has shape (B, T, 2)."""
	if not torch.is_tensor(x):
		raise TypeError(f"{name} must be a torch.Tensor")

	if x.ndim == 2:
		if x.size(-1) != 2:
			raise ValueError(f"{name} must have shape (T, 2) or (B, T, 2)")
		return x.unsqueeze(0)

	if x.ndim == 3 and x.size(-1) == 2:
		return x

	raise ValueError(f"{name} must have shape (T, 2) or (B, T, 2)")


def compute_ADE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	"""Compute Average Displacement Error (ADE).

	Args:
		pred: Predicted trajectory tensor with shape (T, 2) or (B, T, 2).
		gt: Ground-truth trajectory tensor with shape (T, 2) or (B, T, 2).

	Returns:
		Scalar ADE tensor averaged across batch and time.
	"""
	pred_b = _to_batched_trajectory(pred, "pred")
	gt_b = _to_batched_trajectory(gt, "gt")

	if pred_b.shape != gt_b.shape:
		raise ValueError(f"pred and gt must have matching shapes, got {pred_b.shape} and {gt_b.shape}")

	disp = torch.norm(pred_b - gt_b, dim=-1)
	return disp.mean()


def compute_FDE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	"""Compute Final Displacement Error (FDE).

	Args:
		pred: Predicted trajectory tensor with shape (T, 2) or (B, T, 2).
		gt: Ground-truth trajectory tensor with shape (T, 2) or (B, T, 2).

	Returns:
		Scalar FDE tensor averaged across batch.
	"""
	pred_b = _to_batched_trajectory(pred, "pred")
	gt_b = _to_batched_trajectory(gt, "gt")

	if pred_b.shape != gt_b.shape:
		raise ValueError(f"pred and gt must have matching shapes, got {pred_b.shape} and {gt_b.shape}")

	final_disp = torch.norm(pred_b[:, -1, :] - gt_b[:, -1, :], dim=-1)
	return final_disp.mean()


def compute_best_of_k(predictions: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Compute best-of-k ADE/FDE by selecting lowest-ADE mode per sample.

	Args:
		predictions: Predicted trajectories with shape (B, K, T, 2).
		gt: Ground-truth trajectories with shape (T, 2) or (B, T, 2).

	Returns:
		Tuple ``(best_ade, best_fde)`` as scalar tensors.
	"""
	if not torch.is_tensor(predictions):
		raise TypeError("predictions must be a torch.Tensor")
	if predictions.ndim != 4 or predictions.size(-1) != 2:
		raise ValueError("predictions must have shape (B, K, T, 2)")

	gt_b = _to_batched_trajectory(gt, "gt")

	batch_size, _, steps, _ = predictions.shape
	if gt_b.size(0) != batch_size or gt_b.size(1) != steps:
		raise ValueError(
			"predictions and gt must have matching batch/time dimensions, "
			f"got predictions {predictions.shape} and gt {gt_b.shape}"
		)

	gt_expanded = gt_b.unsqueeze(1)
	disp = torch.norm(predictions - gt_expanded, dim=-1)  # Shape: (B, K, T)

	ade_per_mode = disp.mean(dim=-1)  # Shape: (B, K)
	best_mode = torch.argmin(ade_per_mode, dim=1)  # Shape: (B,)

	best_ade = ade_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()

	fde_per_mode = disp[:, :, -1]  # Shape: (B, K)
	best_fde = fde_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()

	return best_ade, best_fde

