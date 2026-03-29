"""Goal-conditioned future endpoint predictor for trajectory forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class GoalPredictor(nn.Module):
	"""Predict multiple future goal endpoints from encoded features.

	Input:
		encoded features of shape (batch, hidden_dim)

	Output:
		goals tensor of shape (batch, 3, 2)
	"""

	def __init__(self, hidden_dim: int = 128, num_goals: int = 3) -> None:
		"""Initialize the goal predictor network.

		Args:
			hidden_dim: Feature dimension of the encoded input.
			num_goals: Number of endpoint hypotheses to predict.
		"""
		super().__init__()
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be a positive integer")
		if num_goals <= 0:
			raise ValueError("num_goals must be a positive integer")

		self.hidden_dim = hidden_dim
		self.num_goals = num_goals

		self.mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, num_goals * 2),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Predict goal endpoints.

		Args:
			x: Encoded features with shape (batch, hidden_dim).

		Returns:
			Goal predictions with shape (batch, 3, 2) when using default
			num_goals=3.
		"""
		if x.ndim != 2:
			raise ValueError("Input x must have shape (batch, hidden_dim)")
		if x.size(1) != self.hidden_dim:
			raise ValueError(
				f"Expected input feature dimension {self.hidden_dim}, got {x.size(1)}"
			)

		batch_size = x.size(0)
		goals = self.mlp(x).view(batch_size, self.num_goals, 2)
		return goals

