"""Inference script for trained SCP trajectory model."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch

from configurations.config_loader import load_config
from models.model_builder import TrajectoryPredictionModel
from utilities.checkpoint import CheckpointManager
from utilities.dataset import NuScenesTrajectoryDataset
from utilities.device import get_device
from utilities.random_seed import set_seed
from utilities.visualization import plot_trajectory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""
    parser = argparse.ArgumentParser(description="Run trained trajectory model inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of trajectory sample to run inference on",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Optional comma-separated sample indices for batched inference",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=4,
        help="Number of CPU threads to use during inference",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save plotted visualization image",
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        default="best",
        choices=("best", "all", "mean", "mode"),
        help="Plotting mode for multimodal predictions",
    )
    parser.add_argument(
        "--mode_index",
        type=int,
        default=0,
        help="Mode index to plot when --plot_mode=mode",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Disable matplotlib interactive window",
    )
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="best",
        choices=("best", "latest"),
        help="Checkpoint to use: best validation model or latest epoch",
    )
    parser.add_argument(
        "--past_points",
        type=int,
        default=0,
        help="Number of last observed points to draw; 0 draws full past",
    )
    parser.add_argument(
        "--plot_frame",
        type=str,
        default="absolute",
        choices=("absolute", "local"),
        help="Plot in absolute map coordinates or local coordinates in meters",
    )
    return parser.parse_args()


def _build_dataset(config: Dict) -> NuScenesTrajectoryDataset:
    """Build configured dataset instance for inference."""
    dataset_cfg = config.get("dataset", {})
    supported_categories = dataset_cfg.get("supported_categories")
    return NuScenesTrajectoryDataset(
        dataroot=str(dataset_cfg.get("dataroot", "./data/nuscenes")),
        version=str(dataset_cfg.get("version", "v1.0-mini")),
        past_steps=int(dataset_cfg.get("past_steps", 2)),
        future_steps=int(dataset_cfg.get("future_steps", 3)),
        supported_category_prefixes=supported_categories,
        min_displacement=float(dataset_cfg.get("min_displacement", 0.2)),
        target_hz=dataset_cfg.get("target_hz", NuScenesTrajectoryDataset.DEFAULT_TARGET_HZ),
    )


def prepare_single_trajectory(
    config: Dict,
    sample_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Prepare a single trajectory sample from configured dataset."""
    dataset = _build_dataset(config)

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index out of range: {sample_index}")

    sample = dataset[sample_index]
    return sample["past"], sample["future"], sample["origin"], int(sample["agent_id"])


def prepare_batch_trajectories(
    config: Dict,
    sample_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Prepare multiple trajectory samples for batched inference."""
    dataset = _build_dataset(config)
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")

    past_list: List[torch.Tensor] = []
    future_list: List[torch.Tensor] = []
    origin_list: List[torch.Tensor] = []
    agent_ids: List[int] = []
    for index in sample_indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"sample_index out of range: {index}")
        sample = dataset[index]
        past_list.append(sample["past"])
        future_list.append(sample["future"])
        origin_list.append(sample["origin"])
        agent_ids.append(int(sample["agent_id"]))

    past_batch = torch.stack(past_list, dim=0)
    future_batch = torch.stack(future_list, dim=0)
    origin_batch = torch.stack(origin_list, dim=0)
    return past_batch, future_batch, origin_batch, agent_ids


@torch.no_grad()
def predict_futures(model: TrajectoryPredictionModel, past: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Predict future paths from past trajectory batch tensor."""
    if past.ndim == 2:
        past = past.unsqueeze(0)
    past_batched = past.to(device=device, dtype=torch.float32)
    predictions = model(past_batched)
    return predictions.detach().cpu()


def select_plot_prediction(
    predictions: torch.Tensor,
    future: torch.Tensor,
    plot_mode: str,
    mode_index: int,
) -> Tuple[torch.Tensor, Optional[int]]:
    """Select which predicted trajectory to visualize for a single sample."""
    if predictions.ndim == 2:
        return predictions, None

    if plot_mode == "all":
        return predictions, None

    if plot_mode == "mean":
        return predictions.mean(dim=0), None

    if plot_mode == "mode":
        idx = int(mode_index)
        idx = max(0, min(idx, predictions.size(0) - 1))
        return predictions[idx], idx

    # Default: use the best mode by ADE.
    if future.ndim != 2:
        raise ValueError("future must have shape (T, 2) for plot_mode=best")
    gt = future.unsqueeze(0).expand(predictions.size(0), -1, -1)
    disp = torch.norm(predictions - gt, dim=-1).mean(dim=-1)
    best_idx = int(torch.argmin(disp).item())
    return predictions[best_idx], best_idx


def main() -> None:
    """Run full inference flow.

    Steps:
    1. Load config
    2. Load trained checkpoint
    3. Prepare single trajectory
    4. Predict 3 possible futures
    5. Visualize trajectories
    """
    args = parse_args()

    # 1) Load config.
    config = load_config(args.config)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    # Keep results reproducible.
    training_cfg = config.get("training", {})
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    # Pick device from config.
    device = get_device(config)

    # Build model.
    model = TrajectoryPredictionModel(config=config).to(device)
    model.eval()

    # 2) Load checkpoint.
    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_dir = str(checkpoint_cfg.get("save_dir", "checkpoints"))
    latest_name = str(checkpoint_cfg.get("filename", "latest.pth"))
    best_name = str(checkpoint_cfg.get("best_filename", "best.pth"))

    preferred_name = best_name if args.checkpoint_mode == "best" else latest_name
    fallback_name = latest_name if args.checkpoint_mode == "best" else best_name

    try:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            filename=preferred_name,
        )
        loaded_epoch = checkpoint_manager.load(model=model, optimizer=None)
        loaded_name = preferred_name
    except FileNotFoundError:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            filename=fallback_name,
        )
        loaded_epoch = checkpoint_manager.load(model=model, optimizer=None)
        loaded_name = fallback_name

    # 3) Prepare input tensors.
    if args.sample_indices.strip():
        sample_indices = [int(part.strip()) for part in args.sample_indices.split(",") if part.strip()]
        past, future, origins, agent_ids = prepare_batch_trajectories(
            config=config,
            sample_indices=sample_indices,
        )
    else:
        sample_indices = [args.sample_index]
        single_past, single_future, single_origin, single_agent_id = prepare_single_trajectory(
            config=config,
            sample_index=args.sample_index,
        )
        past = single_past.unsqueeze(0)
        future = single_future.unsqueeze(0)
        origins = single_origin.unsqueeze(0)
        agent_ids = [single_agent_id]

    dataset_for_timing = _build_dataset(config)
    print(
        "Dataset timing mapping: "
        f"source_dt={dataset_for_timing.source_step_seconds:.3f}s, "
        f"effective_dt={dataset_for_timing.effective_step_seconds:.3f}s, "
        f"past_window={dataset_for_timing.past_window_seconds:.3f}s, "
        f"future_horizon={dataset_for_timing.future_horizon_seconds:.3f}s"
    )

    # 4) Predict future paths.
    predicted_paths = predict_futures(model=model, past=past, device=device)

    print(f"Loaded checkpoint file: {loaded_name}")
    print(f"Loaded checkpoint epoch: {loaded_epoch}")
    print(f"Sample indices: {sample_indices}")
    print(f"Agent IDs: {agent_ids}")
    print(f"Predicted trajectories shape: {tuple(predicted_paths.shape)}")
    print(predicted_paths)

    # 5) Plot first sample only.
    first_pred_all = predicted_paths[0]
    first_past = past[0, :, :2]
    first_future = future[0]
    first_origin = origins[0]
    first_pred, selected_mode_idx = select_plot_prediction(
        predictions=first_pred_all,
        future=first_future,
        plot_mode=args.plot_mode,
        mode_index=args.mode_index,
    )
    if args.plot_frame == "absolute":
        first_past = NuScenesTrajectoryDataset.denormalize_with_origin(first_past, first_origin)
        first_future = NuScenesTrajectoryDataset.denormalize_with_origin(first_future, first_origin)
        first_pred = NuScenesTrajectoryDataset.denormalize_with_origin(first_pred, first_origin)
    else:
        scale = float(NuScenesTrajectoryDataset.NORMALIZATION_SCALE)
        first_past = first_past * scale
        first_future = first_future * scale
        first_pred = first_pred * scale

    if args.past_points > 0:
        first_past = first_past[-args.past_points :]
    if selected_mode_idx is not None:
        print(f"Selected plotted mode index: {selected_mode_idx}")
    print(f"Plot frame: {args.plot_frame}")
    print("First sample absolute predicted trajectory (meters):")
    print(first_pred)

    if args.plot_mode == "all":
        plot_title = "SCP Trajectory Inference (All Modes)"
    elif args.plot_mode == "mean":
        plot_title = "SCP Trajectory Inference (Mean Mode)"
    elif args.plot_mode == "mode":
        plot_title = f"SCP Trajectory Inference (Mode {selected_mode_idx})"
    else:
        plot_title = f"SCP Trajectory Inference (Best Mode {selected_mode_idx})"

    if args.plot_frame == "local":
        plot_title = f"{plot_title} | Local Frame"
    else:
        plot_title = f"{plot_title} | Absolute Frame"

    plot_trajectory(
        past=first_past,
        future=first_future,
        predicted=first_pred,
        title=plot_title,
        save_path=args.save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

