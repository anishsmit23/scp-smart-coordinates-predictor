"""LSTM encoder for trajectory sequences."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
	"""Encode observed trajectory into hidden representations.

	Input shape:
		``(batch, seq_len, 2)``

	Returns:
		- sequence features: ``(batch, seq_len, hidden_dim)``
		- final hidden state: ``(batch, hidden_dim)``
	"""

	def __init__(self, input_dim: int = 10, hidden_dim: int = 96, num_layers: int = 2) -> None:
		super().__init__()
		if input_dim <= 0:
			raise ValueError("input_dim must be a positive integer")
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be a positive integer")
		if num_layers <= 0:
			raise ValueError("num_layers must be a positive integer")

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.encoder = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
		)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Encode input trajectory.

		Args:
			x: Tensor of shape ``(batch, seq_len, input_dim)``.

		Returns:
			Tuple of ``(sequence_features, final_hidden)``.
		"""
		if x.ndim != 3:
			raise ValueError("Input x must have shape (batch, seq_len, input_dim)")
		if x.size(-1) != self.input_dim:
			raise ValueError(
				f"Expected input_dim={self.input_dim}, got feature dimension {x.size(-1)}"
			)

		# Makes LSTM run a bit faster.
		self.encoder.flatten_parameters()
		sequence_features, (hidden_n, _) = self.encoder(x)
		final_hidden = hidden_n[-1]
		return sequence_features, final_hidden

