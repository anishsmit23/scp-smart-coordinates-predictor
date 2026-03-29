"""Multi-modal LSTM decoder for trajectory prediction."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TrajectoryDecoder(nn.Module):
	"""Decode goal-conditioned context into multiple future trajectory modes.

	Output shape is ``(batch, num_modes, future_steps, 2)``.
	"""

	def __init__(
		self,
		hidden_size: int = 128,
		future_steps: int = 6,
		num_modes: int = 3,
	) -> None:
		"""Initialize decoder.

		Args:
			hidden_size: Hidden size for LSTM decoder.
			future_steps: Number of future steps to predict.
			num_modes: Number of trajectory modes.
		"""
		super().__init__()
		if hidden_size <= 0:
			raise ValueError("hidden_size must be a positive integer")
		if future_steps <= 0:
			raise ValueError("future_steps must be a positive integer")
		if num_modes <= 0:
			raise ValueError("num_modes must be a positive integer")

		self.hidden_size = hidden_size
		self.future_steps = future_steps
		self.num_modes = num_modes

		self.input_proj = nn.Linear(2, hidden_size)
		self.init_hidden = nn.Linear(hidden_size, hidden_size)
		self.decoder_cell = nn.LSTMCell(hidden_size, hidden_size)
		self.output_layer = nn.Linear(hidden_size, 2)
		self.mode_embedding = nn.Embedding(num_modes, hidden_size)

	def _validate_context(self, context: torch.Tensor) -> torch.Tensor:
		"""Validate and standardize context to shape ``(B, M, H)``."""
		if context.ndim == 2:
			if context.size(-1) != self.hidden_size:
				raise ValueError(
					f"Expected context feature dim {self.hidden_size}, got {context.size(-1)}"
				)
			return context.unsqueeze(1).expand(-1, self.num_modes, -1)

		if context.ndim == 3:
			if context.size(1) != self.num_modes:
				raise ValueError(
					f"Expected context num_modes={self.num_modes}, got {context.size(1)}"
				)
			if context.size(2) != self.hidden_size:
				raise ValueError(
					f"Expected context hidden dim {self.hidden_size}, got {context.size(2)}"
				)
			return context

		raise ValueError("context must have shape (batch, hidden_size) or (batch, num_modes, hidden_size)")

	def _prepare_teacher_targets(
		self,
		target_trajectory: Optional[torch.Tensor],
		batch_size: int,
		device: torch.device,
		dtype: torch.dtype,
	) -> Optional[torch.Tensor]:
		"""Prepare teacher-forcing targets as shape ``(B, M, T, 2)``."""
		if target_trajectory is None:
			return None

		target = target_trajectory.to(device=device, dtype=dtype)

		if target.ndim == 3:
			if target.shape != (batch_size, self.future_steps, 2):
				raise ValueError(
					"target_trajectory with ndim=3 must have shape "
					f"(batch, {self.future_steps}, 2)"
				)
			target = target.unsqueeze(1).expand(-1, self.num_modes, -1, -1)
			return target

		if target.ndim == 4:
			if target.shape != (batch_size, self.num_modes, self.future_steps, 2):
				raise ValueError(
					"target_trajectory with ndim=4 must have shape "
					f"(batch, {self.num_modes}, {self.future_steps}, 2)"
				)
			return target

		raise ValueError(
			"target_trajectory must have shape "
			f"(batch, {self.future_steps}, 2) or (batch, {self.num_modes}, {self.future_steps}, 2)"
		)

	def forward(
		self,
		context: torch.Tensor,
		target_trajectory: Optional[torch.Tensor] = None,
		teacher_forcing_ratio: float = 0.0,
	) -> torch.Tensor:
		"""Decode trajectories from goal-conditioned context.

		Args:
			context: Goal-conditioned context, shape ``(B, H)`` or ``(B, M, H)``.
			target_trajectory: Optional teacher targets with shape ``(B, T, 2)``
				or ``(B, M, T, 2)``.
			teacher_forcing_ratio: Probability of feeding ground-truth input at
				each decoding step.

		Returns:
			Predicted trajectories of shape ``(B, M, T, 2)``.
		"""
		if not 0.0 <= teacher_forcing_ratio <= 1.0:
			raise ValueError("teacher_forcing_ratio must be in [0, 1]")

		context = self._validate_context(context)
		batch_size, _, _ = context.shape

		target = self._prepare_teacher_targets(
			target_trajectory=target_trajectory,
			batch_size=batch_size,
			device=context.device,
			dtype=context.dtype,
		)

		# Turn target points into step-by-step deltas.
		if target is not None:
			target_deltas = torch.zeros_like(target)
			target_deltas[:, :, 0, :] = target[:, :, 0, :]
			if self.future_steps > 1:
				target_deltas[:, :, 1:, :] = target[:, :, 1:, :] - target[:, :, :-1, :]
		else:
			target_deltas = None

		# Add mode embeddings so modes stay different.
		mode_ids = torch.arange(self.num_modes, device=context.device, dtype=torch.long)
		mode_embed = self.mode_embedding(mode_ids).unsqueeze(0).expand(batch_size, -1, -1)
		context = context + mode_embed

		# Merge modes into batch for faster decoding.
		flat_context = context.reshape(batch_size * self.num_modes, self.hidden_size)
		hidden = self.init_hidden(flat_context)
		cell = torch.zeros_like(hidden)

		decoder_input_xy = torch.zeros(
			batch_size * self.num_modes,
			2,
			device=context.device,
			dtype=context.dtype,
		)

		step_outputs = []
		for step in range(self.future_steps):
			embedded = self.input_proj(decoder_input_xy)
			hidden, cell = self.decoder_cell(embedded, (hidden, cell))
			step_pred = self.output_layer(hidden)
			step_outputs.append(step_pred)

			use_teacher = (
				target_deltas is not None
				and teacher_forcing_ratio > 0.0
				and torch.rand(1, device=context.device).item() < teacher_forcing_ratio
			)

			if use_teacher:
				teacher_step = target_deltas[:, :, step, :].reshape(batch_size * self.num_modes, 2)
				decoder_input_xy = teacher_step
			else:
				decoder_input_xy = step_pred

		stacked = torch.stack(step_outputs, dim=1)
		positions = torch.cumsum(stacked, dim=1)
		return positions.view(batch_size, self.num_modes, self.future_steps, 2)

