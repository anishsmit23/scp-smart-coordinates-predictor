"""Checkpoint management utilities for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
	"""Manage model checkpoint save/load operations."""

	def __init__(self, checkpoint_dir: str = "checkpoints", filename: str = "latest.pth") -> None:
		"""Initialize checkpoint manager.

		Args:
			checkpoint_dir: Directory where checkpoints are stored.
			filename: Default checkpoint file name.
		"""
		self.checkpoint_dir = Path(checkpoint_dir)
		self.filename = filename if filename.endswith(".pth") else f"{filename}.pth"

		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	@property
	def checkpoint_path(self) -> Path:
		"""Return full path to default checkpoint file."""
		return self.checkpoint_dir / self.filename

	def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> Path:
		"""Save model and optimizer states to a .pth checkpoint.

		Stored keys:
			- model_state_dict
			- optimizer_state_dict
			- epoch

		Args:
			model: PyTorch model.
			optimizer: PyTorch optimizer.
			epoch: Current epoch index.

		Returns:
			Path to saved checkpoint.
		"""
		if epoch < 0:
			raise ValueError("epoch must be non-negative")

		state = {
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"epoch": int(epoch),
		}

		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		path = self.checkpoint_path
		torch.save(state, path)
		return path

	def load(
		self,
		model: torch.nn.Module,
		optimizer: Optional[torch.optim.Optimizer] = None,
	) -> int:
		"""Load checkpoint into model/optimizer.

		Args:
			model: PyTorch model to restore weights into.
			optimizer: Optional optimizer to restore state into.

		Returns:
			Loaded epoch value.

		Raises:
			FileNotFoundError: If checkpoint file does not exist.
			KeyError: If expected keys are missing in checkpoint.
		"""
		path = self.checkpoint_path
		if not path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {path}")

		checkpoint = torch.load(path, map_location="cpu")

		if "model_state_dict" in checkpoint:
			state_dict = checkpoint["model_state_dict"]
		else:
			# Support old split checkpoint format.
			legacy_components = (
				"encoder",
				"social_pool",
				"transformer",
				"goal_predictor",
				"goal_condition",
				"decoder",
			)
			if not all(component in checkpoint for component in legacy_components):
				raise KeyError("Missing 'model_state_dict' in checkpoint")

			state_dict = {}
			for component in legacy_components:
				component_state = checkpoint[component]
				for key, value in component_state.items():
					state_dict[f"{component}.{key}"] = value

		model.load_state_dict(state_dict)

		if optimizer is not None:
			if "optimizer_state_dict" not in checkpoint:
				raise KeyError("Missing 'optimizer_state_dict' in checkpoint")
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

		return int(checkpoint.get("epoch", 0))

