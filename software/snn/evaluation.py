#!/usr/bin/env python3
"""Evaluate a saved ebv-eye SNN (SpikingNet8LayerBN) checkpoint.

SNN counterpart to ann/evaluation.py -- see that file's docstring for why
ann/ and snn/ each maintain their own evaluation.py (loading a checkpoint
means importing that framework's own dataloader/model/training) while the
framework-agnostic reporting tools (`best-from-history`, `report`) live one
level up in ../evaluation_report.py. Same function names as ann/evaluation.py
(evaluate/load_experiment_config/checkpoint_metadata/evaluate_for_report),
different bodies -- ../evaluation_report.py's `report` subcommand calls
whichever one applies without needing to know the difference.

SNN's current scope is left-eye-only (see dataloader.py's build_dataset --
no --eyes concept at all), so unlike ann/evaluation.py's evaluate() there is
no per-eye loop here. This reruns training.py's own run_epoch() in eval mode
(optimizer=None) against the saved test-set cache, so the reported numbers
are guaranteed to match what training itself would report for that config,
not a re-derivation of the metric formulas.

Usage:
    python evaluation.py --experiment-dir <dir> [--checkpoint best_checkpoint.pt]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import ann_dataloader, build_dataset
from model import DEFAULT_MIN_V_MEM, DEFAULT_SPIKE_THRESHOLD, SpikingNet8LayerBN
from training import build_criterion, run_epoch


def load_experiment_config(experiment_dir: Path) -> dict:
    """No historical-default backfill needed here (unlike ann/evaluation.py's
    version): snn/training.py has always serialized the full `vars(args)`
    (including `loss`) into config.json since this pipeline's inception, so
    there is no era of configs missing fields to patch."""
    return json.loads((experiment_dir / "config.json").read_text())


def build_model(config: dict) -> nn.Module:
    """按实验目录的 config.json 重建模型。本项目只有一个骨干
    (SpikingNet8LayerBN)，早期 config 里可能还留着 backbone 字段，忽略即可。"""
    return SpikingNet8LayerBN(
        num_timesteps=config["num_steps"],
        task=config["task"],
        num_digits=config.get("num_digits", 1),
        spike_threshold=config.get("spike_threshold", DEFAULT_SPIKE_THRESHOLD),
        min_v_mem=config.get("min_v_mem", DEFAULT_MIN_V_MEM),
    )


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """把 checkpoint 载入模型，过滤掉瞬态膜电位 buffer。"""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Every IAFSqueeze layer's `v_mem` is transient per-forward-pass state
    # (sinabs.layers.StatefulLayer: starts at shape (0,), lazily
    # shape-initialized to match whatever input the layer first sees, and
    # explicitly zeroed by model.reset_states() -- called before every
    # batch in run_epoch() below -- before it is ever read). The checkpoint
    # captured whatever shape training's last batch happened to leave it
    # in, which essentially never matches a freshly constructed model (and
    # is irrelevant even when it does, since it gets reset immediately
    # anyway) -- load_state_dict would otherwise hard-fail on this shape
    # mismatch. `min_v_mem` (also a StatefulLayer buffer, but a genuine
    # hyperparameter rather than transient state) is NOT filtered out and
    # loads normally. This is not a hypothetical: naively loading the full
    # state_dict here throws RuntimeError on every one of this pipeline's
    # existing checkpoints (2026-08-22, verified against real experiment
    # directories -- see .agent/ebv-eye.md).
    filtered_out = {k for k in state_dict if k.endswith(".v_mem")}
    state_dict = {k: v for k, v in state_dict.items() if k not in filtered_out}
    load_result = model.load_state_dict(state_dict, strict=False)
    # strict=False only to tolerate the .v_mem keys filtered out above --
    # anything else missing/unexpected is a real problem (e.g. an
    # architecture mismatch between build_model(config) and what actually
    # produced this checkpoint) and should fail loudly, not be silently
    # swallowed by strict=False.
    allowed_missing = {
        key for key in load_result.missing_keys if key.endswith(".v_mem")
    }
    if set(load_result.missing_keys) != allowed_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"unexpected state_dict mismatch beyond the filtered .v_mem buffers: "
            f"missing={set(load_result.missing_keys) - allowed_missing}, unexpected={load_result.unexpected_keys}"
        )

def build_test_loader(config: dict, batch_size: int, num_workers: int, users: list[int] | None = None) -> DataLoader:
    """按实验 config 重建测试集 loader，顺序与训练时一致。"""
    dataset_root = Path(config.get("dataset_root", "/data/xuzx/data/ebv-eye"))
    if users is not None:
        # Pooled multi-user evaluation (see ann/evaluation.py's
        # build_pooled_eye_dataset counterpart): evaluate on an arbitrary
        # user list -- typically the genuinely held-out test users, which
        # training.py --users/--val-users never sees -- rather than the
        # single-user test.npz this config's own `user` field points at
        # (null for a pooled-trained config anyway).
        from dataloader import build_pooled_dataset

        test_set = build_pooled_dataset(
            dataset_root,
            users,
            num_steps=config.get("num_steps", 10),
            label_mode=config.get("task", "regression"),
            valid_start_point=config.get("valid_start_point", "corrective_peak"),
            downsampled_to=config.get("roi_size", 128),
        )
    else:
        test_set = build_dataset(
            dataset_root,
            config["user"],
            "test",
            num_steps=config.get("num_steps", 10),
            label_mode=config.get("task", "regression"),
            valid_start_point=config.get("valid_start_point", "corrective_peak"),
            downsampled_to=config.get("roi_size", 128),
            cache_dir=(Path(config["cache_dir"]) if config.get("cache_dir") else None),
        )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate(
    experiment_dir: Path,
    checkpoint: str,
    batch_size: int,
    num_workers: int,
    device: str | None = None,
    users: list[int] | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    config = load_experiment_config(experiment_dir)
    model = build_model(config).to(device_obj)
    checkpoint_path = experiment_dir / checkpoint
    load_checkpoint_into_model(model, checkpoint_path, device_obj)
    loader = build_test_loader(config, batch_size, num_workers, users=users)

    task = config.get("task", "regression")
    row_values = col_values = None
    if task == "classification":
        row_values = torch.tensor(ann_dataloader.ROW_VALUES, device=device_obj)
        col_values = torch.tensor(ann_dataloader.COL_VALUES, device=device_obj)

    criterion = build_criterion(task, config.get("loss", "angular_cosine"))
    metrics = run_epoch(
        model,
        loader,
        criterion,
        device_obj,
        task,
        row_values=row_values,
        col_values=col_values,
        num_digits=config.get("num_digits", 1),
    )

    if checkpoint == "best_checkpoint.pt":
        selection_metric = config.get("selection_metric", "test_loss")
        checkpoint_selection = f"minimum {selection_metric} during training"
    else:
        checkpoint_selection = None
    return {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": checkpoint_selection,
        "eval_population": {"users": users} if users is not None else {"user": config.get("user")},
        "metrics": {"pooled" if users is not None else "left": metrics},
    }


def checkpoint_metadata(experiment_dir: Path) -> dict:
    config = load_experiment_config(experiment_dir)
    history = json.loads((experiment_dir / "history.json").read_text())
    best_epoch = config.get("best_checkpoint_epoch")
    if best_epoch is not None:
        matches = [row for row in history if row["epoch"] == best_epoch]
        if len(matches) != 1:
            raise RuntimeError(f"config best_checkpoint_epoch={best_epoch} is not unique in history")
        best = matches[0]
    else:
        # Historical compatibility: old configs did not persist the chosen
        # epoch. Respect any recorded eligibility/selection metric first,
        # then fall back to the legacy minimum-test-loss convention.
        selection_metric = config.get("selection_metric", "test_loss")
        candidates = [
            row
            for row in history
            if row.get("selection_eligible", True) and row.get(selection_metric) is not None
        ]
        best = min(candidates, key=lambda row: row[selection_metric]) if candidates else min(
            history, key=lambda row: row["test_loss"]
        )
    return {
        "checkpoint_epoch": best["epoch"],
        "validation_loss_at_checkpoint": best.get("validation_loss", best.get("test_loss")),
    }


def evaluate_for_report(experiment_dir: Path, batch_size: int, num_workers: int, device: str | None = None) -> dict:
    """Adapter called by ../evaluation_report.py's `report` subcommand --
    SNN has no population/eyes concept to filter on for a single-user run
    (every such run is left-eye-only), so unlike ann/evaluation.py's
    version, this only returns "excluded" for pooled multi-user runs (see
    that file's evaluate_for_report for why: report's batch comparison has
    no way to know which user population -- val or held-out test -- to
    evaluate a pooled checkpoint on); every complete single-user experiment
    directory is in scope."""
    config = load_experiment_config(experiment_dir)
    if config.get("users") is not None:
        return {
            "status": "excluded",
            "reason": "pooled multi-user run -- report's batch comparison has no way to know which "
            "user population (val or held-out test) to evaluate on; call evaluate()/main() directly "
            "with --users instead",
        }
    result = evaluate(experiment_dir, "best_checkpoint.pt", batch_size, num_workers, device)
    result["status"] = "ok"
    result.update(checkpoint_metadata(experiment_dir))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_checkpoint.pt")
    parser.add_argument(
        "--users",
        nargs="+",
        type=int,
        default=None,
        help="Pooled multi-user evaluation: evaluate the checkpoint on these users' pooled eye0 "
        "cache (dataloader.build_pooled_dataset) instead of config.json's single `user` test.npz "
        "-- pass the held-out test users (e.g. 15 16 25) to get a genuine test-set number for a "
        "pooled-trained checkpoint, since training.py --users/--val-users never evaluates on them.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args.experiment_dir, args.checkpoint, args.batch_size, args.num_workers, args.device, users=args.users)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
