"""Transformer encoder module for trajectory prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding for sequence features."""

	def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
		super().__init__()
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be a positive integer")
		if max_len <= 0:
			raise ValueError("max_len must be a positive integer")

		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, hidden_dim, 2, dtype=torch.float32)
			* (-math.log(10000.0) / hidden_dim)
		)

		pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Add positional encoding to sequence tensor.

		Args:
			x: Tensor of shape ``(batch, seq_len, hidden_dim)``.

		Returns:
			Tensor of shape ``(batch, seq_len, hidden_dim)``.
		"""
		if x.ndim != 3:
			raise ValueError("Input x must have shape (batch, seq_len, hidden_dim)")

		seq_len = x.size(1)
		if seq_len > self.pe.size(1):
			raise ValueError(
				f"Sequence length {seq_len} exceeds max positional length {self.pe.size(1)}"
			)

		x = x + self.pe[:, :seq_len, :]
		return self.dropout(x)


class TrajectoryTransformer(nn.Module):
	"""Transformer encoder for context-aware trajectory features."""

	def __init__(
		self,
		hidden_dim: int = 128,
		num_heads: int = 4,
		num_layers: int = 2,
		dropout: float = 0.1,
	) -> None:
		super().__init__()

		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be a positive integer")
		if num_heads <= 0:
			raise ValueError("num_heads must be a positive integer")
		if num_layers <= 0:
			raise ValueError("num_layers must be a positive integer")
		if hidden_dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")

		self.hidden_dim = hidden_dim
		self.positional_encoding = PositionalEncoding(hidden_dim=hidden_dim, dropout=dropout)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_dim,
			nhead=num_heads,
			dropout=dropout,
			batch_first=True,
			activation="relu",
		)
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer=encoder_layer,
			num_layers=num_layers,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Transform encoded trajectory sequence.

		Args:
			x: Input tensor with shape ``(batch, seq_len, hidden_dim)``.

		Returns:
			Context-aware features with shape ``(batch, seq_len, hidden_dim)``.
		"""
		if x.ndim != 3:
			raise ValueError("Input x must have shape (batch, seq_len, hidden_dim)")
		if x.size(-1) != self.hidden_dim:
			raise ValueError(
				f"Expected hidden_dim={self.hidden_dim}, but got input feature dim {x.size(-1)}"
			)

		x = self.positional_encoding(x)
		return self.transformer_encoder(x)
	