"""Learning-rate scheduler builder utilities."""

from __future__ import annotations

from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def _resolve_scheduler_config(config: Dict[str, Any]) -> Dict[str, Any]:
	"""Resolve scheduler sub-config from either full or direct config mapping."""
	if not isinstance(config, dict):
		raise TypeError("config must be a dictionary")

	scheduler_cfg = config.get("scheduler")
	if isinstance(scheduler_cfg, dict):
		return scheduler_cfg
	return config


def build_scheduler(optimizer: Optimizer, config: Dict[str, Any]):
	"""Build and return a learning-rate scheduler from configuration.

	Supported scheduler types:
	- StepLR
	- CosineAnnealingLR

	Args:
		optimizer: Initialized optimizer.
		config: Full project config or scheduler-specific config.

	Returns:
		A PyTorch scheduler object.

	Raises:
		ValueError: If scheduler type is unsupported.
	"""
	scheduler_cfg = _resolve_scheduler_config(config)
	scheduler_type = str(scheduler_cfg.get("type", "StepLR")).strip().lower()

	if scheduler_type == "steplr":
		step_size = int(scheduler_cfg.get("step_size", 10))
		gamma = float(scheduler_cfg.get("gamma", 0.1))
		return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

	if scheduler_type == "cosineannealinglr":
		t_max = int(scheduler_cfg.get("t_max", scheduler_cfg.get("step_size", 10)))
		eta_min = float(scheduler_cfg.get("eta_min", 0.0))
		return CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=eta_min)

	raise ValueError(
		"Unsupported scheduler type: "
		f"{scheduler_cfg.get('type')}. Supported: StepLR, CosineAnnealingLR"
	)

