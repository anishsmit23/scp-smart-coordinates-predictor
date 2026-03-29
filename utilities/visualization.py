"""Visualization utilities for trajectory prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(array: np.ndarray | "object") -> np.ndarray:
	"""Convert supported inputs to numpy array."""
	if isinstance(array, np.ndarray):
		return array

	try:
		import torch

		if torch.is_tensor(array):
			return array.detach().cpu().numpy()
	except Exception:
		pass

	return np.asarray(array)


def _validate_xy(name: str, arr: np.ndarray) -> None:
	"""Validate trajectory array shape ``(T, 2)``."""
	if arr.ndim != 2 or arr.shape[1] != 2:
		raise ValueError(f"{name} must have shape (steps, 2), got {arr.shape}")


def plot_trajectory(
	past,
	future,
	predicted,
	title: str = "Trajectory Visualization",
	save_path: Optional[str] = None,
	show: bool = True,
) -> None:
	"""Plot past, future, and predicted trajectories using matplotlib.

	Colors:
	- Past: Blue
	- Future: Green
	- Predicted: Red

	Args:
		past: Past trajectory with shape ``(T_past, 2)``.
		future: Ground truth future trajectory with shape ``(T_future, 2)``.
		predicted: Predicted trajectory with shape ``(T_future, 2)`` or
			multimodal predictions with shape ``(M, T_future, 2)``.
		title: Plot title.
		save_path: Optional output image path.
		show: Whether to display the figure window.
	"""
	past_np = _to_numpy(past)
	future_np = _to_numpy(future)
	pred_np = _to_numpy(predicted)

	_validate_xy("past", past_np)
	_validate_xy("future", future_np)

	if pred_np.ndim == 2:
		_validate_xy("predicted", pred_np)
	elif pred_np.ndim == 3 and pred_np.shape[-1] == 2:
		pass
	else:
		raise ValueError(
			"predicted must have shape (steps, 2) or (num_modes, steps, 2), "
			f"got {pred_np.shape}"
		)

	fig, ax = plt.subplots(figsize=(8, 6))

	ax.plot(past_np[:, 0], past_np[:, 1], color="blue", marker="o", linewidth=2, label="Past")
	ax.plot(future_np[:, 0], future_np[:, 1], color="green", marker="o", linewidth=2, label="Future")

	if pred_np.ndim == 2:
		ax.plot(
			pred_np[:, 0],
			pred_np[:, 1],
			color="red",
			marker="x",
			linestyle="--",
			linewidth=2,
			label="Predicted",
		)
	else:
		for mode_idx in range(pred_np.shape[0]):
			label = "Predicted" if mode_idx == 0 else None
			ax.plot(
				pred_np[mode_idx, :, 0],
				pred_np[mode_idx, :, 1],
				color="red",
				marker="x",
				linestyle="--",
				linewidth=1.8,
				alpha=0.8,
				label=label,
			)

	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.grid(True, alpha=0.3)
	ax.axis("equal")
	ax.legend()

	if save_path:
		out_path = Path(save_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=150, bbox_inches="tight")

	# Do not open GUI when saving files.
	if show and not save_path:
		plt.show()

	plt.close(fig)

