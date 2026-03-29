"""Sinusoidal positional encoding module for transformer models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
	"""Add sinusoidal positional information to sequence embeddings.

	Expected input shape:
		``(batch, seq_len, hidden_dim)``
	"""

	def __init__(self, hidden_dim: int, max_len: int = 5000) -> None:
		"""Initialize sinusoidal positional encoding table.

		Args:
			hidden_dim: Feature dimension of transformer embeddings.
			max_len: Maximum sequence length supported.
		"""
		super().__init__()
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be a positive integer")
		if max_len <= 0:
			raise ValueError("max_len must be a positive integer")

		position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, hidden_dim, 2, dtype=torch.float32)
			* (-math.log(10000.0) / hidden_dim)
		)

		pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		# Keep shape for easy batch broadcast.
		self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply positional encoding.

		Args:
			x: Input tensor with shape ``(batch, seq_len, hidden_dim)``.

		Returns:
			Positionally encoded tensor with the same shape as input.
		"""
		if x.ndim != 3:
			raise ValueError("Input must have shape (batch, seq_len, hidden_dim)")
		if x.size(-1) != self.pe.size(-1):
			raise ValueError(
				f"hidden_dim mismatch: expected {self.pe.size(-1)}, got {x.size(-1)}"
			)

		seq_len = x.size(1)
		if seq_len > self.pe.size(1):
			raise ValueError(
				f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}"
			)

		return x + self.pe[:, :seq_len, :]

