"""Model builder for complete trajectory prediction network."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from configurations.config_loader import load_config
from models.decoder import TrajectoryDecoder
from models.encoder import LSTMEncoder
from models.future_predictor import GoalPredictor
from models.social_pool import SocialPooling
from models.transformer import TrajectoryTransformer
from utilities.preprocessing import find_neighbors


class TrajectoryPredictionModel(nn.Module):
	"""Complete trajectory prediction model assembled from project modules.

	Pipeline:
		past -> encoder -> social pooling -> transformer -> goal predictor
		-> decoder -> predictions
	"""

	def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None) -> None:
		"""Initialize model from config dictionary or YAML config path."""
		super().__init__()
		self.config = self._resolve_config(config=config, config_path=config_path)
		performance_cfg = self.config.get("performance", {})
		cpu_threads = int(performance_cfg.get("cpu_threads", 4))
		if cpu_threads > 0:
			torch.set_num_threads(cpu_threads)
		cudnn_benchmark = bool(performance_cfg.get("cudnn_benchmark", True))
		torch.backends.cudnn.benchmark = cudnn_benchmark

		model_cfg = self.config.get("model", {})
		social_cfg = self.config.get("social_pooling", {})
		dataset_cfg = self.config.get("dataset", {})

		self.input_dim = int(model_cfg.get("input_dim", 10))
		self.hidden_dim = int(model_cfg.get("hidden_dim", 96))
		self.lstm_layers = int(model_cfg.get("lstm_layers", 2))
		self.transformer_heads = int(model_cfg.get("transformer_heads", 2))
		self.transformer_layers = int(model_cfg.get("transformer_layers", 1))
		self.dropout = float(model_cfg.get("dropout", 0.1))
		self.num_modes = int(model_cfg.get("num_modes", 4))
		self.future_steps = int(dataset_cfg.get("future_steps", 3))
		self.neighbor_radius = float(social_cfg.get("neighbor_radius", 2.0))
		self.social_dim = int(social_cfg.get("social_dim", max(1, 128 - self.hidden_dim)))
		self.pooling_type = str(social_cfg.get("pooling_type", "mean"))
		self.grid_size = int(social_cfg.get("grid_size", 8))
		self.combined_feature_dim = int(model_cfg.get("combined_feature_dim", 128))
		if self.hidden_dim + self.social_dim != self.combined_feature_dim:
			self.social_dim = max(1, self.combined_feature_dim - self.hidden_dim)

		self.encoder = LSTMEncoder(
			input_dim=self.input_dim,
			hidden_dim=self.hidden_dim,
			num_layers=self.lstm_layers,
		)
		self.social_pool = SocialPooling(
			hidden_dim=self.hidden_dim,
			social_dim=self.social_dim,
			pooling_type=self.pooling_type,
			grid_size=self.grid_size,
			neighbor_radius=self.neighbor_radius,
		)
		self.temporal_social_proj = nn.Linear(self.hidden_dim + self.social_dim, self.hidden_dim)
		self.temporal_social_activation = nn.ReLU()
		self.transformer = TrajectoryTransformer(
			hidden_dim=self.hidden_dim,
			num_heads=self.transformer_heads,
			num_layers=self.transformer_layers,
			dropout=self.dropout,
		)
		self.goal_predictor = GoalPredictor(hidden_dim=self.hidden_dim, num_goals=self.num_modes)
		self.goal_condition = nn.Linear(2, self.hidden_dim)
		self.decoder = TrajectoryDecoder(
			hidden_size=self.hidden_dim,
			future_steps=self.future_steps,
			num_modes=self.num_modes,
		)

	@staticmethod
	def _resolve_config(
		config: Optional[Dict[str, Any]],
		config_path: Optional[str],
	) -> Dict[str, Any]:
		"""Resolve config from provided dictionary/path/default file."""
		if config is not None:
			if not isinstance(config, dict):
				raise TypeError("config must be a dictionary when provided")
			return config

		if config_path is not None:
			return load_config(config_path)

		default_path = Path("configurations/config.yaml")
		if default_path.exists():
			return load_config(str(default_path))

		return {}

	def forward(
		self,
		past: torch.Tensor,
		neighbor_indices: Optional[list[list[int]]] = None,
		target_trajectory: Optional[torch.Tensor] = None,
		teacher_forcing_ratio: float = 0.0,
	) -> torch.Tensor:
		"""Run forward prediction pipeline.

		Args:
			past: Past trajectory tensor with shape ``(batch, past_steps, 2)``.
			neighbor_indices: Optional precomputed neighbor index list.
			target_trajectory: Optional future trajectory for teacher forcing.
			teacher_forcing_ratio: Teacher forcing probability passed to decoder.

		Returns:
			Predicted trajectories with shape
			``(batch, num_modes, future_steps, 2)``.
		"""
		if past.ndim != 3 or past.shape[-1] != self.input_dim:
			raise ValueError(
				f"past must have shape (batch, past_steps, {self.input_dim})"
			)

		encoded_seq, encoded_last = self.encoder(past)
		current_positions = past[:, -1, :2]

		if neighbor_indices is None:
			neighbor_indices = find_neighbors(current_positions, radius=self.neighbor_radius)

		social_features = self.social_pool(
			hidden_states=encoded_last,
			neighbor_indices=neighbor_indices,
			positions=current_positions,
		)
		social_seq = social_features.unsqueeze(1).expand(-1, encoded_seq.size(1), -1)
		combined_seq = torch.cat([encoded_seq, social_seq], dim=-1)
		transformer_input = self.temporal_social_activation(self.temporal_social_proj(combined_seq))

		transformed = self.transformer(transformer_input)
		context = transformed[:, -1, :]

		goals = self.goal_predictor(context)
		conditioned_context = context.unsqueeze(1) + self.goal_condition(goals)

		predictions = self.decoder(
			conditioned_context,
			target_trajectory=target_trajectory,
			teacher_forcing_ratio=teacher_forcing_ratio,
		)
		return predictions

