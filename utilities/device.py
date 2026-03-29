"""Device management utilities."""

from __future__ import annotations

from typing import Any, Dict

import torch


def get_device(config: Dict[str, Any]) -> torch.device:
	"""Select computation device based on config and availability.

	Selection logic:
	- Use CUDA when requested and available.
	- Otherwise fall back to CPU.

	Args:
		config: Configuration dictionary with optional ``device`` section.

	Returns:
		Selected ``torch.device``.
	"""
	device_cfg = config.get("device", {}) if isinstance(config, dict) else {}
	use_gpu = bool(device_cfg.get("use_gpu", True))
	gpu_id = int(device_cfg.get("gpu_id", 0))

	if use_gpu and torch.cuda.is_available():
		if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
			gpu_id = 0
		device = torch.device(f"cuda:{gpu_id}")
	else:
		device = torch.device("cpu")

	print(f"Selected device: {device}")
	return device

