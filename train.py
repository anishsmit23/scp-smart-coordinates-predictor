"""Training entry script for SCP trajectory prediction."""

from __future__ import annotations

import argparse

import torch

from configurations.config_loader import load_config
from models.model_builder import TrajectoryPredictionModel
from training.trainer import Trainer
from utilities.dataset import NuScenesTrajectoryDataset
from utilities.device import get_device
from utilities.random_seed import set_seed


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Train SCP trajectory prediction model")
	parser.add_argument(
		"--config",
		type=str,
		default="configurations/config.yaml",
		help="Path to YAML configuration file",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=None,
		help="Optional epoch override for short runs",
	)
	parser.add_argument(
		"--max_train_batches",
		type=int,
		default=None,
		help="Optional cap on train batches per epoch",
	)
	parser.add_argument(
		"--max_val_batches",
		type=int,
		default=None,
		help="Optional cap on validation batches per epoch",
	)
	parser.add_argument(
		"--print_sample_io",
		action="store_true",
		help="Print one sample input trajectory and model output before training",
	)
	parser.add_argument(
		"--sample_index",
		type=int,
		default=0,
		help="Sample index used when --print_sample_io is enabled",
	)
	parser.add_argument(
		"--export_torchscript",
		type=str,
		default="",
		help="Optional output path to save TorchScript model (e.g. model_scripted.pt)",
	)
	return parser.parse_args()


def main() -> None:
	"""Run training pipeline.

	Steps:
	1. Load config.yaml
	2. Set device
	3. Initialize dataset
	4. Build model
	5. Initialize Trainer
	6. Train model
	"""
	args = parse_args()

	# 1) Load config.
	config = load_config(args.config)
	training_cfg = config.setdefault("training", {})
	if args.epochs is not None:
		training_cfg["epochs"] = int(args.epochs)
	if args.max_train_batches is not None:
		training_cfg["max_train_batches"] = int(args.max_train_batches)
	if args.max_val_batches is not None:
		training_cfg["max_val_batches"] = int(args.max_val_batches)

	# Set seed for reproducible runs.
	seed = int(training_cfg.get("seed", 42))
	set_seed(seed)

	# 2) Set device.
	device = get_device(config)

	# 3) Build dataset.
	dataset_cfg = config.get("dataset", {})
	supported_categories = dataset_cfg.get("supported_categories")
	dataset = NuScenesTrajectoryDataset(
		dataroot=str(dataset_cfg.get("dataroot", "./data/nuscenes")),
		version=str(dataset_cfg.get("version", "v1.0-mini")),
		past_steps=int(dataset_cfg.get("past_steps", 2)),
		future_steps=int(dataset_cfg.get("future_steps", 3)),
		supported_category_prefixes=supported_categories,
		min_displacement=float(dataset_cfg.get("min_displacement", 0.2)),
		target_hz=dataset_cfg.get("target_hz", NuScenesTrajectoryDataset.DEFAULT_TARGET_HZ),
	)
	print(
		"Dataset timing mapping: "
		f"source_dt={dataset.source_step_seconds:.3f}s, "
		f"effective_dt={dataset.effective_step_seconds:.3f}s, "
		f"past_window={dataset.past_window_seconds:.3f}s, "
		f"future_horizon={dataset.future_horizon_seconds:.3f}s"
	)

	# 4) Build model.
	model = TrajectoryPredictionModel(config=config)
	model.to(device)

	if args.print_sample_io:
		sample = dataset[args.sample_index]
		past = sample["past"]
		future = sample["future"]
		with torch.no_grad():
			model.eval()
			pred = model(past.unsqueeze(0).to(device)).squeeze(0).detach().cpu()

		print("Sample Input (past):")
		print(past)
		print("Sample Target (future):")
		print(future)
		print("Model Output (predicted multimodal future):")
		print(pred)

	# 5) Build trainer.
	trainer = Trainer(config=config, model=model, dataset=dataset)

	# 6) Start training.
	history = trainer.train()
	if history:
		last_epoch = sorted(history.keys(), key=int)[-1]
		print(f"Training complete. Final epoch metrics ({last_epoch}): {history[last_epoch]}")

	if args.export_torchscript.strip():
		model.eval()
		scripted_model = torch.jit.script(model.cpu())
		scripted_model.save(args.export_torchscript)
		print(f"TorchScript model exported to: {args.export_torchscript}")


if __name__ == "__main__":
	main()

