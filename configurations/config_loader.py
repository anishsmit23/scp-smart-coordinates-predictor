"""Configuration loader utilities for YAML-based project settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class ConfigLoader:
	"""Load and access YAML configuration with nested key support.

	Nested keys can be accessed using dot notation, for example:
	``dataset.batch_size``.
	"""

	def __init__(self, config_path: str) -> None:
		"""Initialize loader with path to YAML configuration file."""
		self.config_path = Path(config_path)
		self._config: Dict[str, Any] = {}

	def load(self) -> Dict[str, Any]:
		"""Load configuration from YAML file.

		Returns:
			Configuration dictionary.

		Raises:
			FileNotFoundError: If the configuration file does not exist.
			ValueError: If YAML content is invalid or not a mapping.
		"""
		if not self.config_path.exists():
			raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

		try:
			with self.config_path.open("r", encoding="utf-8") as handle:
				loaded = yaml.safe_load(handle)
		except yaml.YAMLError as exc:
			raise ValueError(f"Failed to parse YAML file: {self.config_path}") from exc

		if loaded is None:
			self._config = {}
			return self._config

		if not isinstance(loaded, dict):
			raise ValueError("Configuration root must be a dictionary/mapping")

		self._config = loaded
		self._resolve_time_windows()
		return self._config

	def _resolve_time_windows(self) -> None:
		"""Resolve dataset step counts from optional time-window config.

		Supported dataset keys:
		- ``past_seconds`` and ``future_seconds``
		- ``step_seconds`` (preferred) or ``target_hz``

		If present, these values become the source of truth and overwrite
		``past_steps``/``future_steps`` with derived integer counts.
		"""
		dataset_cfg = self._config.get("dataset")
		if not isinstance(dataset_cfg, dict):
			return

		past_seconds = dataset_cfg.get("past_seconds")
		future_seconds = dataset_cfg.get("future_seconds")
		if past_seconds is None and future_seconds is None:
			return

		step_seconds_raw = dataset_cfg.get("step_seconds")
		target_hz_raw = dataset_cfg.get("target_hz")

		step_seconds: Optional[float] = None
		if step_seconds_raw is not None:
			try:
				step_seconds = float(step_seconds_raw)
			except (TypeError, ValueError):
				step_seconds = None

		if (step_seconds is None or step_seconds <= 0) and target_hz_raw is not None:
			try:
				target_hz = float(target_hz_raw)
				if target_hz > 0:
					step_seconds = 1.0 / target_hz
			except (TypeError, ValueError):
				step_seconds = None

		if step_seconds is None or step_seconds <= 0:
			# Use nuScenes default step time.
			step_seconds = 0.5

		if past_seconds is not None:
			past_sec = float(past_seconds)
			if past_sec <= 0:
				raise ValueError("dataset.past_seconds must be positive")
			dataset_cfg["past_steps"] = max(1, int(round(past_sec / step_seconds)))

		if future_seconds is not None:
			future_sec = float(future_seconds)
			if future_sec <= 0:
				raise ValueError("dataset.future_seconds must be positive")
			dataset_cfg["future_steps"] = max(1, int(round(future_sec / step_seconds)))

		dataset_cfg["step_seconds"] = step_seconds

	def get(self, key: str, default: Any = None) -> Any:
		"""Get config value by key with graceful fallback.

		Supports dot-notation lookup for nested keys.

		Args:
			key: Config key (for example ``training.epochs``).
			default: Default value returned when key is missing.

		Returns:
			The resolved config value, or ``default`` if not found.
		"""
		if not key:
			return default

		current: Any = self._config
		for part in key.split("."):
			if not isinstance(current, dict) or part not in current:
				return default
			current = current[part]
		return current

	def as_dict(self) -> Dict[str, Any]:
		"""Return full configuration dictionary."""
		return self._config

	def apply_overrides(self, overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
		"""Apply command-line style overrides to loaded config.

		Override format:
			- ``section.key=value``
			- ``top_level=value``

		Values are parsed into bool/int/float when possible.

		Args:
			overrides: Iterable of override strings.

		Returns:
			Updated configuration dictionary.
		"""
		if not overrides:
			return self._config

		for item in overrides:
			if "=" not in item:
				continue

			key, raw_value = item.split("=", 1)
			key = key.strip()
			if not key:
				continue

			value = self._parse_scalar(raw_value.strip())
			self._set_nested_value(key, value)

		return self._config

	def _set_nested_value(self, dotted_key: str, value: Any) -> None:
		"""Set a nested config value using dot notation."""
		parts = dotted_key.split(".")
		current = self._config

		for part in parts[:-1]:
			next_node = current.get(part)
			if not isinstance(next_node, dict):
				next_node = {}
				current[part] = next_node
			current = next_node

		current[parts[-1]] = value

	@staticmethod
	def _parse_scalar(raw_value: str) -> Any:
		"""Parse scalar string into bool/int/float/None/str."""
		lowered = raw_value.lower()
		if lowered == "true":
			return True
		if lowered == "false":
			return False
		if lowered == "none" or lowered == "null":
			return None

		try:
			if any(ch in raw_value for ch in (".", "e", "E")):
				return float(raw_value)
			return int(raw_value)
		except ValueError:
			return raw_value

	def __getitem__(self, key: str) -> Any:
		"""Provide dictionary-style access with KeyError on missing keys."""
		sentinel = object()
		value = self.get(key, default=sentinel)
		if value is sentinel:
			raise KeyError(key)
		return value

	def __contains__(self, key: str) -> bool:
		"""Check key existence with nested dot notation support."""
		sentinel = object()
		return self.get(key, default=sentinel) is not sentinel


def load_config(path: str) -> Dict[str, Any]:
	"""Load a YAML configuration file and return it as dictionary.

	Args:
		path: YAML file path.

	Returns:
		Configuration dictionary.

	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError: If the YAML file cannot be parsed.
	"""
	loader = ConfigLoader(path)
	return loader.load()

