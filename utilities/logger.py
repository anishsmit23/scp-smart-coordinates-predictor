"""Training logger utilities for trajectory prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional


class TrainingLogger:
	"""Logger for training metrics with console, file, and TensorBoard support."""

	def __init__(
		self,
		log_dir: str = "logs",
		log_filename: str = "training.log",
		use_tensorboard: bool = True,
		logger_name: str = "training_logger",
	) -> None:
		"""Initialize logger outputs and TensorBoard writer.

		Args:
			log_dir: Directory to store log files and TensorBoard data.
			log_filename: Log file name.
			use_tensorboard: Enable TensorBoard logging if available.
			logger_name: Python logger name.
		"""
		self.log_dir = Path(log_dir)
		self.log_dir.mkdir(parents=True, exist_ok=True)
		self.log_path = self.log_dir / log_filename

		self.logger = logging.getLogger(logger_name)
		self.logger.setLevel(logging.INFO)
		self.logger.propagate = False

		# Clear handlers to avoid duplicate logs.
		if self.logger.handlers:
			self.logger.handlers.clear()

		formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

		stream_handler = logging.StreamHandler()
		stream_handler.setLevel(logging.INFO)
		stream_handler.setFormatter(formatter)

		file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)

		self.logger.addHandler(stream_handler)
		self.logger.addHandler(file_handler)

		self.writer = None
		if use_tensorboard:
			try:
				from torch.utils.tensorboard import SummaryWriter

				self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
				self.logger.info("TensorBoard logging enabled.")
			except Exception:
				self.writer = None
				self.logger.warning("TensorBoard is unavailable. Continuing without it.")

	@staticmethod
	def _format_metrics(loss: float, ade: float, fde: float) -> str:
		"""Format metrics as a compact string."""
		return f"loss={loss:.6f} | ADE={ade:.6f} | FDE={fde:.6f}"

	def log_step(self, epoch: int, step: int, loss: float, ade: float, fde: float) -> None:
		"""Log training metrics for a single optimization step."""
		message = f"Epoch {epoch} Step {step} | {self._format_metrics(loss, ade, fde)}"
		self.logger.info(message)

		if self.writer is not None:
			global_step = max((epoch - 1), 0) * 1_000_000 + step
			self.writer.add_scalar("step/loss", loss, global_step)
			self.writer.add_scalar("step/ADE", ade, global_step)
			self.writer.add_scalar("step/FDE", fde, global_step)

	def log_epoch(self, epoch: int, loss: float, ade: float, fde: float) -> None:
		"""Log aggregated metrics for one epoch."""
		message = f"Epoch {epoch} Summary | {self._format_metrics(loss, ade, fde)}"
		self.logger.info(message)

		if self.writer is not None:
			self.writer.add_scalar("epoch/loss", loss, epoch)
			self.writer.add_scalar("epoch/ADE", ade, epoch)
			self.writer.add_scalar("epoch/FDE", fde, epoch)

	def save_log(self, metrics: Dict[str, float], prefix: str = "summary") -> None:
		"""Save arbitrary metric dictionary into log stream and TensorBoard.

		Args:
			metrics: Mapping of metric name to metric value.
			prefix: Optional prefix for TensorBoard scalar tags.
		"""
		metric_parts = [f"{key}={value:.6f}" for key, value in metrics.items()]
		self.logger.info("Saved metrics | " + " | ".join(metric_parts))

		if self.writer is not None:
			for key, value in metrics.items():
				self.writer.add_scalar(f"{prefix}/{key}", float(value))

	def close(self) -> None:
		"""Close TensorBoard writer if initialized."""
		if self.writer is not None:
			self.writer.flush()
			self.writer.close()

