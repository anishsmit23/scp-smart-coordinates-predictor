"""Loss functions for trajectory prediction."""

from __future__ import annotations

import torch


def _validate_single_modal_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
	"""Validate tensors with expected shape ``(B, T, 2)``."""
	if pred.ndim != 3 or pred.size(-1) != 2:
		raise ValueError("pred must have shape (batch, steps, 2)")
	if target.ndim != 3 or target.size(-1) != 2:
		raise ValueError("target must have shape (batch, steps, 2)")
	if pred.shape != target.shape:
		raise ValueError(f"pred and target must have matching shapes, got {pred.shape} and {target.shape}")


def ade_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Compute Average Displacement Error (ADE).

	Args:
		pred: Predicted trajectory tensor of shape ``(B, T, 2)``.
		target: Ground-truth trajectory tensor of shape ``(B, T, 2)``.

	Returns:
		Scalar loss tensor (mean ADE across batch and timesteps).
	"""
	_validate_single_modal_inputs(pred, target)
	displacement = torch.norm(pred - target, dim=-1)
	return displacement.mean()


def fde_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Compute Final Displacement Error (FDE).

	Args:
		pred: Predicted trajectory tensor of shape ``(B, T, 2)``.
		target: Ground-truth trajectory tensor of shape ``(B, T, 2)``.

	Returns:
		Scalar loss tensor (mean final-step displacement across batch).
	"""
	_validate_single_modal_inputs(pred, target)
	final_disp = torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1)
	return final_disp.mean()


def multimodal_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	lambda_reg: float = 0.02,
	lambda_div: float = 0.08,
	lambda_fde: float = 0.5,
	lambda_smooth: float = 0.03,
	lambda_first: float = 0.4,
	lambda_path: float = 0.25,
	lambda_heading: float = 0.2,
	select_with_fde: float = 0.3,
	early_step_bias: float = 0.6,
	hard_mining_alpha: float = 0.0,
) -> torch.Tensor:
	"""Compute multimodal trajectory loss with stability and mode-diversity terms.

	Args:
		pred: Predicted trajectories of shape ``(B, M, T, 2)``.
		target: Ground-truth trajectories of shape ``(B, T, 2)``.

	Returns:
		Scalar loss tensor combining:
		- best-mode weighted ADE
		- best-mode FDE
		- first-step alignment
		- path-shape consistency
		- smoothness and regularization terms
	"""
	if pred.ndim != 4 or pred.size(-1) != 2:
		raise ValueError("pred must have shape (batch, modes, steps, 2)")
	if target.ndim != 3 or target.size(-1) != 2:
		raise ValueError("target must have shape (batch, steps, 2)")
	if pred.size(0) != target.size(0) or pred.size(2) != target.size(1):
		raise ValueError(
			"Batch and step dimensions must match between pred and target, "
			f"got pred {pred.shape} and target {target.shape}"
		)

	disp = torch.norm(pred - target.unsqueeze(1), dim=-1)  # Shape: (B, M, T)
	steps = pred.size(2)
	time_weights = torch.linspace(
		1.0 + float(early_step_bias),
		1.0,
		steps,
		dtype=pred.dtype,
		device=pred.device,
	)
	time_weights = time_weights / time_weights.sum()
	weighted_ade_per_mode = (disp * time_weights.view(1, 1, -1)).sum(dim=-1)  # Shape: (B, M)

	fde_per_mode = torch.norm(pred[:, :, -1, :] - target[:, -1, :].unsqueeze(1), dim=-1)

	selection_score = weighted_ade_per_mode + float(select_with_fde) * fde_per_mode
	best_mode = selection_score.min(dim=1)[1]
	best_weighted_ade = weighted_ade_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()
	best_fde = fde_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()
	first_step_per_mode = disp[:, :, 0]
	best_first = first_step_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()

	if steps >= 2:
		pred_deltas = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		target_deltas = target[:, 1:, :] - target[:, :-1, :]
		path_per_mode = torch.norm(pred_deltas - target_deltas.unsqueeze(1), dim=-1).mean(dim=-1)
		best_path = path_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()
	else:
		best_path = torch.zeros((), dtype=pred.dtype, device=pred.device)

	# Align first-step motion direction with target to reduce heading mismatch.
	best_pred_first = pred[torch.arange(pred.size(0), device=pred.device), best_mode, 0, :]
	target_first = target[:, 0, :]
	cos_sim = torch.nn.functional.cosine_similarity(best_pred_first, target_first, dim=-1, eps=1e-6)
	heading_loss = (1.0 - cos_sim).mean()

	mode_magnitude = torch.norm(
		pred,
		dim=-1,
	).mean()

	# Smoothness term reduces jitter.
	if pred.size(2) >= 3:
		vel = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		acc = vel[:, :, 1:, :] - vel[:, :, :-1, :]
		smoothness_loss = torch.norm(acc, dim=-1).mean()
	else:
		smoothness_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)

	if pred.size(1) >= 2:
		num_modes = pred.size(1)
		pairs = torch.combinations(torch.arange(num_modes, device=pred.device), r=2)
		pair_distances = []
		for pair in pairs:
			i = int(pair[0].item())
			j = int(pair[1].item())
			# Mean distance per sample over time for this mode pair.
			pair_dist = torch.norm(pred[:, i] - pred[:, j], dim=-1).mean(dim=-1)
			pair_distances.append(pair_dist)

		pair_distances_stacked = torch.stack(pair_distances, dim=1)  # (B, P)
		diversity_per_sample = torch.exp(-pair_distances_stacked.mean(dim=1))
	else:
		diversity_per_sample = torch.zeros(pred.size(0), dtype=pred.dtype, device=pred.device)

	regularization_loss = lambda_reg * mode_magnitude

	# Build per-sample objective so we can emphasize harder samples.
	best_weighted_ade_per_sample = weighted_ade_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1)
	best_fde_per_sample = fde_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1)
	best_first_per_sample = first_step_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1)
	if steps >= 2:
		best_path_per_sample = path_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1)
	else:
		best_path_per_sample = torch.zeros(pred.size(0), dtype=pred.dtype, device=pred.device)

	heading_cos = torch.nn.functional.cosine_similarity(best_pred_first, target_first, dim=-1, eps=1e-6)
	heading_per_sample = (1.0 - heading_cos)

	core_per_sample = (
		best_weighted_ade_per_sample
		+ lambda_fde * best_fde_per_sample
		+ lambda_first * best_first_per_sample
		+ lambda_path * best_path_per_sample
		+ lambda_heading * heading_per_sample
		+ lambda_div * diversity_per_sample
	)

	# Keep global stabilizers as means.
	base_loss = core_per_sample + regularization_loss + lambda_smooth * smoothness_loss

	if hard_mining_alpha > 0.0:
		hardness = best_fde_per_sample.detach()
		norm = hardness.mean().clamp_min(1e-6)
		weights = 1.0 + hard_mining_alpha * (hardness / norm)
		return (base_loss * weights).sum() / weights.sum().clamp_min(1e-6)

	return base_loss.mean()

