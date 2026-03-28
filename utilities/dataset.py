"""PyTorch dataset for trajectory prediction on nuScenes.

This module provides ``NuScenesTrajectoryDataset`` that loads the official
nuScenes data via nuscenes-devkit, extracts agent trajectories for supported
classes, and exposes normalized past/future trajectory windows.
"""

from __future__ import annotations

from collections import defaultdict
import os
import pickle
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset
from tqdm import tqdm


class NuScenesTrajectoryDataset(Dataset):
	"""Trajectory dataset from nuScenes annotations.

	The dataset extracts agent trajectories from nuScenes sample annotations,
	supports pedestrians/bicycles by default, and returns normalized windows:
	- past: ``(past_steps, 10)`` with
	  ``[x, y, vx, vy, ax, ay, speed, heading_sin, heading_cos, yaw_rate]``
	- future: ``(future_steps, 2)``

	For nuScenes 2 Hz data (0.5s step), common defaults are:
	- ``past_steps=4`` (4 observations)
	- ``future_steps=6`` (3 seconds horizon)
	"""
	NORMALIZATION_SCALE = 10.0
	DEFAULT_TARGET_HZ: float = 2.0

	_ALLOWED_CATEGORY_PREFIXES: Tuple[str, ...] = (
		"human.pedestrian",
		"vehicle.bicycle",
	)
	MIN_DISPLACEMENT: float = 0.2

	def __init__(
		self,
		dataroot: str,
		version: str,
		past_steps: int = 2,
		future_steps: int = 3,
		supported_category_prefixes: Sequence[str] | None = None,
		min_displacement: float = 0.2,
		target_hz: float | None = DEFAULT_TARGET_HZ,
	) -> None:
		"""Initialize the nuScenes trajectory dataset.

		Args:
			dataroot: Path to nuScenes data root.
			version: nuScenes version (for example, ``v1.0-trainval``).
			past_steps: Number of past frames per sample.
			future_steps: Number of future frames per sample.
			target_hz: Optional uniform resampling rate for trajectories.
		"""
		super().__init__()

		if past_steps <= 0 or future_steps <= 0:
			raise ValueError("past_steps and future_steps must be positive integers.")

		self.dataroot = dataroot
		self.version = version
		self.past_steps = past_steps
		self.future_steps = future_steps
		self.window_size = self.past_steps + self.future_steps
		self.cache_file = os.path.join(dataroot, "processed_cache.pkl")
		self.min_displacement = float(min_displacement)
		if self.min_displacement < 0:
			raise ValueError("min_displacement must be non-negative")

		self.target_hz = float(target_hz) if target_hz is not None else None
		if self.target_hz is not None and self.target_hz <= 0:
			raise ValueError("target_hz must be positive when provided")

		self.allowed_category_prefixes = tuple(
			supported_category_prefixes
			if supported_category_prefixes is not None
			else self._ALLOWED_CATEGORY_PREFIXES
		)
		# Keep only pedestrian and bicycle classes.
		self.allowed_category_prefixes = tuple(
			prefix for prefix in self.allowed_category_prefixes if prefix in self._ALLOWED_CATEGORY_PREFIXES
		)
		if not self.allowed_category_prefixes:
			self.allowed_category_prefixes = self._ALLOWED_CATEGORY_PREFIXES

		# Step time from timestamps or cache.
		self.source_step_seconds = 0.5
		self.effective_step_seconds = 1.0 / self.target_hz if self.target_hz is not None else self.source_step_seconds

		self.samples = self._load_or_build_processed_samples()
		self.past_window_seconds = self.past_steps * self.effective_step_seconds
		self.future_horizon_seconds = self.future_steps * self.effective_step_seconds
		print(
			"Timing summary: "
			f"source_dt={self.source_step_seconds:.3f}s, "
			f"effective_dt={self.effective_step_seconds:.3f}s, "
			f"past_window={self.past_window_seconds:.3f}s, "
			f"future_horizon={self.future_horizon_seconds:.3f}s"
		)
		self._total_windows = len(self.samples)

	def _load_nuscenes(self) -> NuScenes:
		"""Load nuScenes using the official nuscenes-devkit."""
		version_root = Path(self.dataroot) / self.version
		if not version_root.exists():
			raise FileNotFoundError(
				f"nuScenes table folder not found: {version_root}. "
				"Check dataset.dataroot and dataset.version in configuration."
			)
		return NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

	@staticmethod
	def compute_velocity(trajectory: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
		"""Compute per-step velocity for one trajectory tensor."""
		if not torch.is_tensor(trajectory) or trajectory.ndim != 2 or trajectory.size(1) != 2:
			raise ValueError("trajectory must be a tensor of shape (T, 2)")
		if delta_t <= 0:
			raise ValueError("delta_t must be positive")

		velocity = torch.zeros_like(trajectory)
		if trajectory.size(0) > 1:
			velocity[1:] = (trajectory[1:] - trajectory[:-1]) / float(delta_t)
		velocity[0] = torch.zeros(2, dtype=trajectory.dtype, device=trajectory.device)
		return velocity

	@staticmethod
	def build_motion_features(past_xy: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
		"""Build motion feature channels from normalized past xy.

		Args:
			past_xy: Tensor with shape ``(N, T, 2)``.
			delta_t: Step duration in seconds.

		Returns:
			Tensor with shape ``(N, T, 10)`` containing:
			``[x, y, vx, vy, ax, ay, speed, heading_sin, heading_cos, yaw_rate]``.
		"""
		if not torch.is_tensor(past_xy) or past_xy.ndim != 3 or past_xy.size(-1) != 2:
			raise ValueError("past_xy must be a tensor of shape (N, T, 2)")
		if delta_t <= 0:
			raise ValueError("delta_t must be positive")

		dt = float(delta_t)
		vel = torch.zeros_like(past_xy)
		if past_xy.size(1) > 1:
			vel[:, 1:, :] = (past_xy[:, 1:, :] - past_xy[:, :-1, :]) / dt

		acc = torch.zeros_like(vel)
		if vel.size(1) > 1:
			acc[:, 1:, :] = (vel[:, 1:, :] - vel[:, :-1, :]) / dt

		speed = torch.norm(vel, dim=-1)
		heading = torch.atan2(vel[..., 1], vel[..., 0])
		heading_sin = torch.sin(heading)
		heading_cos = torch.cos(heading)

		yaw_rate = torch.zeros_like(heading)
		if heading.size(1) > 1:
			delta_heading = heading[:, 1:] - heading[:, :-1]
			delta_heading = torch.remainder(delta_heading + torch.pi, 2.0 * torch.pi) - torch.pi
			yaw_rate[:, 1:] = delta_heading / dt

		return torch.cat(
			[
				past_xy,
				vel,
				acc,
				speed.unsqueeze(-1),
				heading_sin.unsqueeze(-1),
				heading_cos.unsqueeze(-1),
				yaw_rate.unsqueeze(-1),
			],
			dim=-1,
		)

	@staticmethod
	def normalize_trajectory(past: torch.Tensor, future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Normalize past and future with shared origin and fixed scale."""
		norm_past, norm_future, _origin = NuScenesTrajectoryDataset.normalize_trajectory_with_origin(
			past,
			future,
		)
		return norm_past, norm_future

	@staticmethod
	def normalize_trajectory_with_origin(
		past: torch.Tensor,
		future: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Normalize past/future and return the absolute origin used for the shift."""
		if not torch.is_tensor(past) or past.ndim != 2 or past.size(1) != 2:
			raise ValueError("past must be a tensor of shape (T, 2)")
		if not torch.is_tensor(future) or future.ndim != 2 or future.size(1) != 2:
			raise ValueError("future must be a tensor of shape (T, 2)")

		origin = past[-1].clone()
		past = past - origin
		future = future - origin

		scale = NuScenesTrajectoryDataset.NORMALIZATION_SCALE
		past = past / scale
		future = future / scale
		return past, future, origin

	@staticmethod
	def denormalize_with_origin(relative_xy: torch.Tensor, origin_xy: torch.Tensor) -> torch.Tensor:
		"""Convert normalized relative xy trajectories back to absolute coordinates (meters)."""
		if not torch.is_tensor(relative_xy) or relative_xy.size(-1) != 2:
			raise ValueError("relative_xy must be a tensor ending with coordinate size 2")
		if not torch.is_tensor(origin_xy) or origin_xy.size(-1) != 2:
			raise ValueError("origin_xy must be a tensor ending with coordinate size 2")
		return relative_xy * NuScenesTrajectoryDataset.NORMALIZATION_SCALE + origin_xy

	def _cache_key(self) -> Dict[str, object]:
		"""Return cache metadata used to validate cache compatibility."""
		return {
			"cache_format_version": 5,
			"version": self.version,
			"past_steps": self.past_steps,
			"future_steps": self.future_steps,
			"allowed_category_prefixes": tuple(self.allowed_category_prefixes),
			"past_feature_schema": "xy_vx_vy_ax_ay_speed_heading_sin_heading_cos_yaw_rate",
			"normalization": "shared_origin_fixed_scale_10",
			"origin_included": True,
			"min_displacement": self.min_displacement,
			"target_hz": self.target_hz,
			"velocity_units": "normalized_units_per_second",
			"acceleration_units": "normalized_units_per_second2",
			"yaw_rate_units": "radians_per_second",
		}

	def _load_or_build_processed_samples(self) -> List[Dict[str, torch.Tensor | int]]:
		"""Load cached samples or build and cache them."""
		cache_path = Path(self.cache_file)
		cache_key = self._cache_key()

		if cache_path.exists():
			try:
				with cache_path.open("rb") as handle:
					payload = pickle.load(handle)
				if isinstance(payload, dict) and payload.get("cache_key") == cache_key and "samples" in payload:
					self.source_step_seconds = float(payload.get("source_step_seconds", self.source_step_seconds))
					self.effective_step_seconds = float(
						payload.get("effective_step_seconds", self.effective_step_seconds)
					)
					total_windows = int(payload.get("total_windows", len(payload["samples"])))
					filtered_size = len(payload["samples"])
					print("Original windows created:", total_windows)
					print("Filtered dataset size:", filtered_size)
					print("Filtered static samples:", total_windows - filtered_size)
					return payload["samples"]
			except Exception:
				# Rebuild cache if it is bad or outdated.
				pass

		nusc = self._load_nuscenes()
		agent_tracks, source_dt_seconds = self._build_trajectories(nusc)
		if source_dt_seconds > 0:
			self.source_step_seconds = source_dt_seconds
		self.effective_step_seconds = 1.0 / self.target_hz if self.target_hz is not None else self.source_step_seconds

		past_np, future_np, origin_np, agent_ids_np, total_windows = self._build_normalized_windows(agent_tracks)
		samples = self._to_samples(past_np, future_np, origin_np, agent_ids_np)

		print("Original windows created:", total_windows)
		print("Filtered dataset size:", len(samples))
		print("Filtered static samples:", total_windows - len(samples))

		cache_path.parent.mkdir(parents=True, exist_ok=True)
		with cache_path.open("wb") as handle:
			pickle.dump(
				{
					"cache_key": cache_key,
					"samples": samples,
					"total_windows": int(total_windows),
					"source_step_seconds": float(self.source_step_seconds),
					"effective_step_seconds": float(self.effective_step_seconds),
				},
				handle,
				protocol=pickle.HIGHEST_PROTOCOL,
			)
		return samples

	@staticmethod
	def _to_samples(
		past_np: np.ndarray,
		future_np: np.ndarray,
		origin_np: np.ndarray,
		agent_ids_np: np.ndarray,
	) -> List[Dict[str, torch.Tensor | int]]:
		"""Convert numpy arrays to sample dictionaries used by __getitem__."""
		if past_np.shape[0] == 0:
			return []

		past = torch.from_numpy(past_np).to(dtype=torch.float32)
		future = torch.from_numpy(future_np).to(dtype=torch.float32)
		origin = torch.from_numpy(origin_np).to(dtype=torch.float32)
		agent_ids = torch.from_numpy(agent_ids_np).to(dtype=torch.int64)

		samples: List[Dict[str, torch.Tensor | int]] = []
		for i in range(int(past.size(0))):
			samples.append(
				{
					"past": past[i],
					"future": future[i],
					"origin": origin[i],
					"agent_id": int(agent_ids[i].item()),
				}
			)
		return samples

	def _is_supported_category(self, category_name: str) -> bool:
		"""Return True if category is one of supported trajectory classes."""
		return any(category_name.startswith(prefix) for prefix in self.allowed_category_prefixes)

	def _safe_get_annotation(self, nusc: NuScenes, ann_token: str) -> dict | None:
		"""Safely fetch annotation record by token."""
		try:
			annotation = nusc.get("sample_annotation", ann_token)
		except Exception:
			return None

		if not isinstance(annotation, dict):
			return None
		return annotation

	def _build_trajectories(self, nusc: NuScenes) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], float]:
		"""Build per-agent timestamped xy trajectories and estimate source step size."""
		token_positions: DefaultDict[str, List[Tuple[int, float, float]]] = defaultdict(list)
		token_to_agent_id: Dict[str, int] = {}

		for sample in tqdm(nusc.sample, desc="Building trajectories", unit="sample"):
			timestamp = sample.get("timestamp")
			if not isinstance(timestamp, int):
				continue

			ann_tokens = sample.get("anns", [])
			if not isinstance(ann_tokens, Sequence):
				continue

			for ann_token in ann_tokens:
				annotation = self._safe_get_annotation(nusc, ann_token)
				if annotation is None:
					continue

				category_name = annotation.get("category_name")
				if not isinstance(category_name, str) or not self._is_supported_category(category_name):
					continue

				instance_token = annotation.get("instance_token")
				translation = annotation.get("translation")
				if not isinstance(instance_token, str):
					continue
				if not isinstance(translation, Sequence) or len(translation) < 2:
					continue

				x, y = float(translation[0]), float(translation[1])
				token_positions[instance_token].append((timestamp, x, y))

				if instance_token not in token_to_agent_id:
					token_to_agent_id[instance_token] = len(token_to_agent_id)

		agent_tracks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
		all_dt_seconds: List[float] = []
		for instance_token, time_xy in token_positions.items():
			agent_id = token_to_agent_id[instance_token]
			ordered_np = np.asarray(sorted(time_xy, key=lambda item: item[0]), dtype=np.float64)
			timestamps_us = ordered_np[:, 0].astype(np.int64, copy=False)
			positions_xy = ordered_np[:, 1:3].astype(np.float32, copy=False)

			if timestamps_us.size > 1:
				diff_s = np.diff(timestamps_us.astype(np.float64)) / 1_000_000.0
				valid = diff_s[diff_s > 0]
				if valid.size > 0:
					all_dt_seconds.extend(valid.tolist())

			agent_tracks[agent_id] = (timestamps_us, positions_xy)

		estimated_dt = float(np.median(np.asarray(all_dt_seconds, dtype=np.float64))) if all_dt_seconds else 0.5
		if estimated_dt <= 0:
			estimated_dt = 0.5
		return agent_tracks, estimated_dt

	@staticmethod
	def _resample_track(
		timestamps_us: np.ndarray,
		positions_xy: np.ndarray,
		target_hz: float | None,
	) -> Tuple[np.ndarray, np.ndarray, float]:
		"""Resample one trajectory track to target_hz using linear interpolation."""
		if timestamps_us.size < 2 or target_hz is None:
			if timestamps_us.size > 1:
				dt = np.diff(timestamps_us.astype(np.float64)) / 1_000_000.0
				dt = dt[dt > 0]
				step_seconds = float(np.median(dt)) if dt.size > 0 else 0.5
			else:
				step_seconds = 0.5
			return timestamps_us, positions_xy, step_seconds

		target_dt_s = 1.0 / float(target_hz)
		target_dt_us = int(round(target_dt_s * 1_000_000.0))
		if target_dt_us <= 0:
			return timestamps_us, positions_xy, target_dt_s

		start_us = int(timestamps_us[0])
		end_us = int(timestamps_us[-1])
		if end_us <= start_us:
			return timestamps_us, positions_xy, target_dt_s

		new_timestamps_us = np.arange(start_us, end_us + 1, target_dt_us, dtype=np.int64)
		if new_timestamps_us.size < 2:
			return timestamps_us, positions_xy, target_dt_s

		x = np.interp(
			new_timestamps_us.astype(np.float64),
			timestamps_us.astype(np.float64),
			positions_xy[:, 0].astype(np.float64),
		)
		y = np.interp(
			new_timestamps_us.astype(np.float64),
			timestamps_us.astype(np.float64),
			positions_xy[:, 1].astype(np.float64),
		)
		new_positions = np.stack([x, y], axis=-1).astype(np.float32, copy=False)
		return new_timestamps_us, new_positions, target_dt_s

	def _build_normalized_windows(
		self,
		agent_tracks: Dict[int, Tuple[np.ndarray, np.ndarray]],
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
		"""Create and normalize all windows once, then return compact numpy arrays."""
		past_chunks: List[np.ndarray] = []
		future_chunks: List[np.ndarray] = []
		origin_chunks: List[np.ndarray] = []
		agent_chunks: List[np.ndarray] = []
		total_windows = 0

		for agent_id, track in tqdm(agent_tracks.items(), desc="Creating windows", unit="agent"):
			timestamps_us, trajectory = track
			_resampled_ts, trajectory, step_seconds = self._resample_track(
				timestamps_us=timestamps_us,
				positions_xy=trajectory,
				target_hz=self.target_hz,
			)

			n_points = int(trajectory.shape[0])
			num_windows = n_points - self.window_size + 1
			if num_windows <= 0:
				continue
			total_windows += num_windows

			windows = np.stack(
				[trajectory[i : i + self.window_size] for i in range(num_windows)],
				axis=0,
			).astype(np.float32, copy=False)

			past_pos_t = torch.from_numpy(windows[:, : self.past_steps, :])
			future_t = torch.from_numpy(windows[:, self.past_steps :, :])

			norm_past_list: List[torch.Tensor] = []
			norm_future_list: List[torch.Tensor] = []
			origin_list: List[torch.Tensor] = []
			for i in range(past_pos_t.size(0)):
				norm_past, norm_future, origin = self.normalize_trajectory_with_origin(
					past_pos_t[i],
					future_t[i],
				)
				norm_past_list.append(norm_past)
				norm_future_list.append(norm_future)
				origin_list.append(origin)

			past_pos_t = torch.stack(norm_past_list, dim=0)
			norm_future_t = torch.stack(norm_future_list, dim=0)
			origin_t = torch.stack(origin_list, dim=0)

			# Drop almost static tracks using future displacement.
			displacement_m = torch.norm(future_t[:, -1, :] - future_t[:, 0, :], dim=-1)
			moving_mask = displacement_m >= self.min_displacement
			if not torch.any(moving_mask):
				continue

			past_pos_t = past_pos_t[moving_mask]
			norm_future_t = norm_future_t[moving_mask]
			origin_t = origin_t[moving_mask]
			future = norm_future_t.cpu().numpy().astype(np.float32, copy=False)

			past_features_t = self.build_motion_features(
				past_xy=past_pos_t,
				delta_t=float(step_seconds),
			)

			past = past_features_t.cpu().numpy().astype(np.float32, copy=False)
			origins = origin_t.cpu().numpy().astype(np.float32, copy=False)

			past_chunks.append(past)
			future_chunks.append(future)
			origin_chunks.append(origins)
			agent_chunks.append(np.full((int(moving_mask.sum().item()),), agent_id, dtype=np.int64))

		if not past_chunks:
			empty_past = np.empty((0, self.past_steps, 10), dtype=np.float32)
			empty_future = np.empty((0, self.future_steps, 2), dtype=np.float32)
			empty_origin = np.empty((0, 2), dtype=np.float32)
			empty_agent_ids = np.empty((0,), dtype=np.int64)
			return empty_past, empty_future, empty_origin, empty_agent_ids, total_windows

		past_np = np.concatenate(past_chunks, axis=0)
		future_np = np.concatenate(future_chunks, axis=0)
		origin_np = np.concatenate(origin_chunks, axis=0)
		agent_ids_np = np.concatenate(agent_chunks, axis=0)
		return past_np, future_np, origin_np, agent_ids_np, total_windows

	def __len__(self) -> int:
		"""Return number of trajectory samples."""
		return self._total_windows

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
		"""Return one pre-normalized sample."""
		if idx < 0:
			idx += self._total_windows
		if idx < 0 or idx >= self._total_windows:
			raise IndexError("Index out of range")

		sample = self.samples[idx]
		return {
			"past": sample["past"],
			"future": sample["future"],
			"origin": sample["origin"],
			"agent_id": int(idx),
		}
 