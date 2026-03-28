"""Advanced training pipeline for trajectory prediction."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from configurations.config_loader import load_config
from models.model_builder import TrajectoryPredictionModel
from training.validator import Validator
from utilities.L_fnc import multimodal_loss
from utilities.checkpoint import CheckpointManager
from utilities.collate_fn import build_collate_fn
from utilities.dataset import NuScenesTrajectoryDataset
from utilities.device import get_device
from utilities.logger import TrainingLogger
from utilities.metrics import compute_best_of_k
from utilities.scheduler import build_scheduler


class Trainer:
	"""Production-grade trainer with train/validation/checkpoint orchestration."""

	def __init__(
		self,
		config: Optional[Dict[str, Any]] = None,
		config_path: str = "configurations/config.yaml",
		model: Optional[TrajectoryPredictionModel] = None,
		dataset: Optional[NuScenesTrajectoryDataset] = None,
		resume: bool = False,
	) -> None:
		"""Initialize model, data, optimizer, scheduler, logger, and checkpoints."""
		self.config = config if config is not None else load_config(config_path)

		dataset_cfg = self.config.get("dataset", {})
		training_cfg = self.config.get("training", {})
		checkpoint_cfg = self.config.get("checkpoint", {})
		logging_cfg = self.config.get("logging", {})
		loss_cfg = self.config.get("loss", {})

		self.dataroot = str(dataset_cfg.get("dataroot", "./data/nuscenes"))
		self.version = str(dataset_cfg.get("version", "v1.0-mini"))
		self.batch_size = int(dataset_cfg.get("batch_size", 32))
		self.num_workers = int(dataset_cfg.get("num_workers", 0))
		self.prefetch_factor = int(dataset_cfg.get("prefetch_factor", 2))
		self.pin_memory = bool(dataset_cfg.get("pin_memory", True))
		self.shuffle = bool(dataset_cfg.get("shuffle", True))
		self.neighbor_radius = float(self.config.get("social_pooling", {}).get("neighbor_radius", 2.0))

		self.epochs = int(training_cfg.get("epochs", 50))
		self.learning_rate = float(training_cfg.get("learning_rate", 1e-3))
		self.weight_decay = float(training_cfg.get("weight_decay", 0.0))
		self.gradient_clip = float(training_cfg.get("gradient_clip", 0.0))
		self.teacher_forcing_ratio = float(training_cfg.get("teacher_forcing_ratio", 0.0))
		self.val_split = float(training_cfg.get("val_split", 0.1))
		self.max_train_batches = int(training_cfg.get("max_train_batches", 0))
		self.max_val_batches = int(training_cfg.get("max_val_batches", 0))
		self.early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))
		self.early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
		self.loss_kwargs = {
			"lambda_reg": float(loss_cfg.get("lambda_reg", 0.02)),
			"lambda_div": float(loss_cfg.get("lambda_div", 0.08)),
			"lambda_fde": float(loss_cfg.get("lambda_fde", 0.5)),
			"lambda_smooth": float(loss_cfg.get("lambda_smooth", 0.03)),
			"lambda_first": float(loss_cfg.get("lambda_first", 0.4)),
			"lambda_path": float(loss_cfg.get("lambda_path", 0.25)),
			"lambda_heading": float(loss_cfg.get("lambda_heading", 0.2)),
			"select_with_fde": float(loss_cfg.get("select_with_fde", 0.3)),
			"early_step_bias": float(loss_cfg.get("early_step_bias", 0.6)),
			"hard_mining_alpha": float(loss_cfg.get("hard_mining_alpha", 0.0)),
		}

		self.save_frequency = int(checkpoint_cfg.get("save_frequency", 1))
		self.print_frequency = max(1, int(logging_cfg.get("print_frequency", 10)))
		self.logger = TrainingLogger(
			log_dir=str(logging_cfg.get("log_dir", "logs")),
			use_tensorboard=bool(logging_cfg.get("use_tensorboard", True)),
		)

		self.device = self._resolve_device()
		self.use_amp = bool(training_cfg.get("use_amp", True) and self.device.type == "cuda")
		self.scaler = self._build_grad_scaler()

		self.model = model if model is not None else TrajectoryPredictionModel(config=self.config)
		self.model.to(self.device)

		self.dataset = dataset if dataset is not None else self._load_dataset()
		self.train_loader, self.val_loader = self._create_data_loaders(self.dataset)

		optimizer_type = str(self.config.get("optimizer", {}).get("type", "AdamW")).strip().lower()
		optimizer_cls = torch.optim.AdamW if optimizer_type == "adamw" else torch.optim.Adam
		self.optimizer = optimizer_cls(
			self.model.parameters(),
			lr=self.learning_rate,
			weight_decay=self.weight_decay,
		)
		self.scheduler = build_scheduler(self.optimizer, self.config)

		self.checkpoint_manager = CheckpointManager(
			checkpoint_dir=str(checkpoint_cfg.get("save_dir", "checkpoints")),
			filename=str(checkpoint_cfg.get("filename", "latest.pth")),
		)
		self.best_checkpoint_manager = CheckpointManager(
			checkpoint_dir=str(checkpoint_cfg.get("save_dir", "checkpoints")),
			filename=str(checkpoint_cfg.get("best_filename", "best.pth")),
		)

		self.validator = Validator(
			model=self.model,
			val_loader=self.val_loader,
			device=self.device,
			show_progress=True,
			max_batches=self.max_val_batches,
			metric_scale=float(getattr(self.dataset, "NORMALIZATION_SCALE", 1.0)),
		)

		self.start_epoch = 1
		if resume:
			try:
				loaded_epoch = self.checkpoint_manager.load(self.model, self.optimizer)
				self.start_epoch = loaded_epoch + 1
				self.logger.save_log({"resume_epoch": float(loaded_epoch)}, prefix="checkpoint")
			except FileNotFoundError:
				self.logger.save_log({"resume_epoch": 0.0}, prefix="checkpoint")

	def _resolve_device(self) -> torch.device:
		"""Resolve runtime device from config and availability."""
		device = get_device(self.config)
		self.logger.save_log({"device_is_cuda": 1.0 if device.type == "cuda" else 0.0}, prefix="system")
		return device

	def _build_grad_scaler(self):
		"""Build AMP GradScaler compatible across PyTorch versions."""
		if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
			return torch.amp.GradScaler("cuda", enabled=self.use_amp)

		if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
			return torch.cuda.amp.GradScaler(enabled=self.use_amp)

		return None

	def _autocast_context(self):
		"""Return autocast context manager compatible across PyTorch versions."""
		if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
			return torch.amp.autocast("cuda", enabled=self.use_amp)

		if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
			return torch.cuda.amp.autocast(enabled=self.use_amp)

		# Use empty context if autocast is missing.
		class _NullContext:
			def __enter__(self):
				return None

			def __exit__(self, exc_type, exc, tb):
				return False

		return _NullContext()

	def _load_dataset(self) -> NuScenesTrajectoryDataset:
		"""Load trajectory dataset from config."""
		dataset_cfg = self.config.get("dataset", {})
		supported_categories = dataset_cfg.get("supported_categories")
		return NuScenesTrajectoryDataset(
			dataroot=self.dataroot,
			version=self.version,
			past_steps=int(dataset_cfg.get("past_steps", 2)),
			future_steps=int(dataset_cfg.get("future_steps", 3)),
			supported_category_prefixes=supported_categories,
			min_displacement=float(dataset_cfg.get("min_displacement", 0.2)),
			target_hz=dataset_cfg.get("target_hz", NuScenesTrajectoryDataset.DEFAULT_TARGET_HZ),
		)

	def _create_data_loaders(self, dataset: NuScenesTrajectoryDataset) -> Tuple[DataLoader, DataLoader]:
		"""Create train and validation dataloaders."""
		dataset_size = len(dataset)
		if dataset_size == 0:
			raise ValueError("Dataset is empty. Cannot start training.")

		if dataset_size < 2:
			train_dataset = dataset
			val_dataset = dataset
		else:
			val_size = max(1, int(dataset_size * self.val_split))
			train_size = dataset_size - val_size
			if train_size == 0:
				train_size = dataset_size - 1
				val_size = 1
			train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

		loader_kwargs: Dict[str, Any] = {
			"num_workers": self.num_workers,
			"pin_memory": self.pin_memory,
			"persistent_workers": self.num_workers > 0,
			"collate_fn": build_collate_fn(neighbor_radius=self.neighbor_radius),
		}
		if self.num_workers > 0:
			loader_kwargs["prefetch_factor"] = self.prefetch_factor

		train_loader = DataLoader(
			train_dataset,
			batch_size=self.batch_size,
			# Shuffle data for better learning.
			shuffle=True,
			**loader_kwargs,
		)
		val_loader = DataLoader(
			val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			**loader_kwargs,
		)
		return train_loader, val_loader

	def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
		"""Run one full training epoch and return aggregated metrics."""
		self.model.train()

		running_loss = 0.0
		running_ade = 0.0
		running_fde = 0.0
		total_count = 0

		progress = tqdm(self.train_loader, desc=f"Train {epoch}/{self.epochs}", unit="batch")
		for step_idx, batch in enumerate(progress, start=1):
			if self.max_train_batches > 0 and step_idx > self.max_train_batches:
				break
			past = batch["past"].to(self.device, non_blocking=self.device.type == "cuda")
			future = batch["future"].to(self.device, non_blocking=self.device.type == "cuda")
			neighbor_indices = batch.get("neighbor_indices")
			batch_size = int(past.size(0))

			self.optimizer.zero_grad(set_to_none=True)

			with self._autocast_context():
				predictions = self.model(
					past,
					neighbor_indices=neighbor_indices,
					target_trajectory=future,
					teacher_forcing_ratio=self.teacher_forcing_ratio,
				)
				loss = multimodal_loss(predictions, future, **self.loss_kwargs)

			if self.use_amp and self.scaler is not None:
				self.scaler.scale(loss).backward()
				self.scaler.unscale_(self.optimizer)
				clip_grad_norm_(
					self.model.parameters(),
					max_norm=5.0,
				)
				self.scaler.step(self.optimizer)
				self.scaler.update()
			else:
				loss.backward()
				clip_grad_norm_(
					self.model.parameters(),
					max_norm=5.0,
				)
				self.optimizer.step()

			ade_value, fde_value = compute_best_of_k(predictions.detach(), future)
			metric_scale = float(getattr(self.dataset, "NORMALIZATION_SCALE", 1.0))
			ade_value = ade_value * metric_scale
			fde_value = fde_value * metric_scale

			loss_item = float(loss.detach().item())
			ade_item = float(ade_value.item())
			fde_item = float(fde_value.item())

			running_loss += loss_item * batch_size
			running_ade += ade_item * batch_size
			running_fde += fde_item * batch_size
			total_count += batch_size

			if step_idx % self.print_frequency == 0 or step_idx == 1:
				self.logger.log_step(
					epoch=epoch,
					step=step_idx,
					loss=loss_item,
					ade=ade_item,
					fde=fde_item,
				)
				progress.set_postfix(loss=f"{loss_item:.4f}", ade=f"{ade_item:.4f}", fde=f"{fde_item:.4f}")

		if total_count == 0:
			return {"loss": 0.0, "ADE": 0.0, "FDE": 0.0}

		return {
			"loss": running_loss / total_count,
			"ADE": running_ade / total_count,
			"FDE": running_fde / total_count,
		}

	def train(self, num_epochs: Optional[int] = None) -> Dict[str, Dict[str, float]]:
		"""Run advanced training loop with validation and checkpointing.

		Args:
			num_epochs: Optional override for configured epoch count.

		Returns:
			History dictionary keyed by epoch string with train/val metrics.
		"""
		total_epochs = int(num_epochs if num_epochs is not None else self.epochs)
		if total_epochs <= 0:
			raise ValueError("num_epochs must be a positive integer")

		history: Dict[str, Dict[str, float]] = {}
		best_val_fde = float("inf")
		epochs_without_improvement = 0
		last_epoch_completed = self.start_epoch - 1

		for epoch in range(self.start_epoch, total_epochs + 1):
			train_metrics = self._train_one_epoch(epoch)
			last_epoch_completed = epoch

			self.scheduler.step()
			current_lr = float(self.optimizer.param_groups[0]["lr"])

			val_metrics = self.validator.validate()

			self.logger.log_epoch(
				epoch=epoch,
				loss=float(train_metrics["loss"]),
				ade=float(val_metrics["ADE"]),
				fde=float(val_metrics["FDE"]),
			)
			self.logger.save_log(
				{
					"train_loss": float(train_metrics["loss"]),
					"train_ADE": float(train_metrics["ADE"]),
					"train_FDE": float(train_metrics["FDE"]),
					"val_ADE": float(val_metrics["ADE"]),
					"val_FDE": float(val_metrics["FDE"]),
					"lr": current_lr,
				},
				prefix=f"epoch_{epoch}",
			)

			if self.save_frequency > 0 and epoch % self.save_frequency == 0:
				self.checkpoint_manager.save(self.model, self.optimizer, epoch)

			current_val_fde = float(val_metrics["FDE"])
			if current_val_fde < (best_val_fde - self.early_stopping_min_delta):
				best_val_fde = current_val_fde
				epochs_without_improvement = 0
				self.best_checkpoint_manager.save(self.model, self.optimizer, epoch)
			else:
				epochs_without_improvement += 1

			if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
				print(
					"Early stopping triggered: "
					f"no val FDE improvement for {epochs_without_improvement} epoch(s)."
				)
				break

			history[str(epoch)] = {
				"train_loss": float(train_metrics["loss"]),
				"train_ADE": float(train_metrics["ADE"]),
				"train_FDE": float(train_metrics["FDE"]),
				"val_ADE": float(val_metrics["ADE"]),
				"val_FDE": float(val_metrics["FDE"]),
				"lr": current_lr,
			}

		# Save final checkpoint.
		self.checkpoint_manager.save(self.model, self.optimizer, max(0, last_epoch_completed))
		self.logger.close()
		return history

