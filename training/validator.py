"""Validation loop utilities for trajectory prediction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utilities.metrics import compute_ADE, compute_FDE, compute_best_of_k


class Validator:
	"""Evaluate trajectory prediction model on a validation dataloader."""

	def __init__(
		self,
		model: torch.nn.Module,
		val_loader: DataLoader,
		device: Optional[torch.device] = None,
		show_progress: bool = True,
		max_batches: int = 0,
		metric_scale: float = 1.0,
	) -> None:
		"""Initialize validator.

		Args:
			model: Trajectory prediction model.
			val_loader: Validation dataloader.
			device: Device for evaluation. If ``None``, auto-selects CUDA/CPU.
			show_progress: Whether to show tqdm progress bar.
			max_batches: Optional cap on validation batches; ``0`` means no cap.
		"""
		self.model = model
		self.val_loader = val_loader
		self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.show_progress = show_progress
		self.max_batches = int(max_batches)
		self.metric_scale = float(metric_scale)

	@torch.no_grad()
	def validate(self) -> Dict[str, float]:
		"""Run validation loop and compute ADE/FDE.

		Returns:
			Dictionary with aggregated validation metrics:
			- ``ADE``
			- ``FDE``
		"""
		self.model.eval()

		total_ade = 0.0
		total_fde = 0.0
		total_count = 0

		iterator = self.val_loader
		if self.show_progress:
			iterator = tqdm(self.val_loader, desc="Validation", unit="batch")

		for batch_idx, batch in enumerate(iterator, start=1):
			if self.max_batches > 0 and batch_idx > self.max_batches:
				break
			if "past" not in batch or "future" not in batch:
				raise KeyError("Validation batch must contain 'past' and 'future'")

			past = batch["past"].to(self.device)
			future = batch["future"].to(self.device)
			batch_size = int(past.size(0))

			neighbor_indices = batch.get("neighbor_indices")
			if neighbor_indices is not None:
				predictions = self.model(past, neighbor_indices=neighbor_indices)
			else:
				predictions = self.model(past)

			if predictions.ndim == 4:
				ade_value, fde_value = compute_best_of_k(predictions, future)
			elif predictions.ndim == 3:
				ade_value = compute_ADE(predictions, future)
				fde_value = compute_FDE(predictions, future)
			else:
				raise ValueError(
					"Model predictions must have shape (B, T, 2) or (B, K, T, 2), "
					f"got {predictions.shape}"
				)

			total_ade += float(ade_value.item()) * batch_size
			total_fde += float(fde_value.item()) * batch_size
			total_count += batch_size

		if total_count == 0:
			return {"ADE": 0.0, "FDE": 0.0}

		metrics = {
			"ADE": (total_ade / total_count) * self.metric_scale,
			"FDE": (total_fde / total_count) * self.metric_scale,
		}
		return metrics

