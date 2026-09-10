#!/usr/bin/env python3
"""在 ebv-eye saccade / smooth-pursuit 任务上训练 SpikingNet8LayerBN。

数据用的是和 ANN 流水线完全相同的 dataset_preprocessing.py 缓存
(`fixed` / `downsample` 子命令, 见 dataloader.py), 区别只在于这里把事件
栅格化成 T 个真实时间 bin, 而不是累加成一帧。

训练用 sinabs.exodus 的 CUDA surrogate-gradient BPTT (必须有 GPU),
神经元是 sinabs.exodus.layers.IAFSqueeze。为什么全程用
MultiSpike / PeriodicExponential / 有限 min_v_mem / MembraneSubtract,
见 model.py。

权重裁剪 (每个 batch 在 optimizer.step() 之后把 |conv weight| 限制到
--weight-clip-value) 沿用归档的 cifar10dvs 线里已经验证过的做法:
Speck 的异步神经元在单个输入脉冲的加权贡献超过约 2 倍阈值时可能失控发放,
所以训练阶段就先把权重压住。

用法:
    LD_LIBRARY_PATH=/data/xuzx/env/02_snn_speck/lib python training.py --user 15 --task regression
    LD_LIBRARY_PATH=/data/xuzx/env/02_snn_speck/lib python training.py --user 15 --task classification
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import _load_ann_module, build_dataset, build_pooled_dataset
from model import (
    DEFAULT_MIN_V_MEM,
    DEFAULT_SPIKE_THRESHOLD,
    SpikingNet8LayerBN,
    count_parameters,
)

ann_dataloader = _load_ann_module("dataloader")
_load_ann_module("model")  # ann/training.py 也从这里 import, 必须先缓存再加载 "training"
ann_training = _load_ann_module("training")

DEFAULT_DATASET_ROOT = Path("/data/xuzx/data/ebv-eye")
DEFAULT_RESULT_ROOT = Path("/data/xuzx/result/02_snn_speck/ebv-eye/snn")


def parse_min_v_mem(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", type=int, default=None, help="Single-user mode (existing behavior): train/test split within this one user's data. Mutually exclusive with --users.")
    parser.add_argument(
        "--users", type=int, nargs="+", default=None,
        help="Multi-user pooled mode: train on the concatenation of these users' `dataset_preprocessing.py fixed "
        "--no-split` pooled caches (see dataloader.build_pooled_dataset), eye0 only. Requires --val-users too. "
        "Mutually exclusive with --user. Per-epoch metrics are named validation_* and computed only "
        "on --val-users; true test users must be evaluated separately "
        "after training, via evaluation.py, and never referenced here.",
    )
    parser.add_argument(
        "--val-users", type=int, nargs="+", default=None,
        help="Required with --users: held-out users used only for per-epoch validation metrics.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Single-user only: explicit cache containing train.npz, validation.npz and test.npz. "
        "When set, training evaluates validation.npz each epoch and never reads test.npz.",
    )
    parser.add_argument("--valid-start-point", default="corrective_peak")
    parser.add_argument("--roi-size", type=int, default=128, help="Also selects the downsampled cache to read (see dataloader.build_dataset)")
    parser.add_argument("--num-steps", type=int, default=10, help="T, number of real-time bins per sample (see dataloader.py)")
    parser.add_argument(
        "--task",
        choices=("regression", "classification"),
        default="regression",
        help="regression: 2*num_digits channels forming positional spike-rate coordinates; "
        "one digit has 11 levels/axis. classification: 22-channel "
        "head (11 row + 11 col classes), spike counts summed over time used as CrossEntropy logits.",
    )
    parser.add_argument(
        "--loss",
        choices=("normalized_mse", "angular_cosine"),
        default="angular_cosine",
        help="Regression only. Both losses compare the reconstructed coordinate with the original, "
        "continuous label. normalized_mse is MSE in normalized [row, col] coordinates; "
        "angular_cosine is 1-cos(angle) in gaze-vector space.",
    )
    parser.add_argument(
        "--num-digits",
        type=int,
        default=1,
        help="Regression only. 1 (default): single T-step rate per axis, 11 levels (0.0/0.1/.../1.0). "
        "N>1: N output neurons per axis forming a base-10 positional rate code "
        "(coarse/mid/fine/...), reconstructed as sum(rate_i * 0.1**i). After clipping "
        "to [0,1], this gives 10**N + 1 unique coordinate levels per axis (for example, "
        "1001 levels for N=3), not 11**N: encodings above 1 are saturated and therefore "
        "duplicate. Every N is trained against the same original continuous coordinate; "
        "the label is never floored, rounded, or decomposed to match the output codebook.",
    )
    parser.add_argument("--spike-threshold", type=float, default=DEFAULT_SPIKE_THRESHOLD)
    parser.add_argument(
        "--min-v-mem",
        type=parse_min_v_mem,
        default=DEFAULT_MIN_V_MEM,
        help="Explicit finite value (default -1.0) or 'none'/'None' for no membrane floor. Historical "
        "experiments used a finite value from epoch 1 through any --resume-checkpoint "
        "continuation. 'none' exists to test the archived 99_archived/snn_speck/code/evbeye line's "
        "curriculum instead: min_v_mem=None for the bulk of training (see its original "
        "configs/evbeye/user4/train_..._e200.json: 'min_v_mem': null), only introducing a finite value "
        "during a --resume-checkpoint fine-tuning continuation -- untested on this project's "
        "own architectures before 2026-08-25. "
        "Passing --min-v-mem none produces NO min_v_mem entry at all in that IAFSqueeze's state_dict "
        "(confirmed empirically, not just 'set to null') -- load_resume_checkpoint() below tolerates "
        "this specific missing-key case when resuming into a run with a finite --min-v-mem.",
    )
    parser.add_argument(
        "--weight-clip-value",
        type=float,
        default=1.0,
        help="Max |weight| for every Conv2d, applied after every optimizer step (as a multiple of "
        "--spike-threshold's units, since spike_threshold=1.0 by default this is directly comparable). "
        "None/<=0 disables clipping.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--resume-checkpoint", type=Path, default=None,
        help="Load this checkpoint's state_dict (e.g. an existing final_checkpoint.pt/best_checkpoint.pt) before "
        "training starts, instead of random init -- for continued-training runs. Uses a "
        "fresh optimizer/scheduler at --lr, not the original run's; --min-v-mem/--spike-threshold/--num-digits/etc. "
        "must match the checkpoint's own architecture or loading will fail loudly.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "test_loss", "test_pixel_l2", "test_angular_deg",
            "validation_loss", "validation_pixel_l2", "validation_angular_deg",
        ),
        default=None,
        help="best_checkpoint.pt 按哪个 validation/test 指标选（取最小值）。默认在 named-cache/pooled "
        "模式用 validation_pixel_l2，旧 single-user cache 用 test_loss。Named cache 不允许用 test_*。",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: <result-root>/{user=N|population=pooled}/<auto name>",
    )
    return parser.parse_args()


def load_resume_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Same v_mem-filtering pattern as evaluation.py's load_checkpoint_into_model()
    -- duplicated rather than imported to avoid a training.py<->evaluation.py
    circular import (evaluation.py already does `from training import
    build_criterion, run_epoch`; see dataloader.py's _load_ann_module for why
    this codebase treats that class of cycle as worth avoiding rather than
    working around)."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    filtered_out = {k for k in state_dict if k.endswith(".v_mem")}
    state_dict = {k: v for k, v in state_dict.items() if k not in filtered_out}
    load_result = model.load_state_dict(state_dict, strict=False)
    # A `.min_v_mem` entry only exists in an IAFSqueeze's own state_dict when it was
    # constructed with a finite min_v_mem (confirmed empirically, 2026-08-25 -- see
    # --min-v-mem's help text: min_v_mem=None means the attribute is plain Python None,
    # never registered as an nn.Parameter, so state_dict() has no key for it at all).
    # Resuming across a None<->finite transition -- the entire point of --min-v-mem
    # none's curriculum experiment -- means this key can legitimately be missing from
    # the checkpoint (None->finite) or present-but-unwanted (finite->None) even though
    # architecturally everything else matches; tolerate both directions here alongside
    # the .v_mem filter, not just the .v_mem case alone.
    allowed_missing = {
        key for key in load_result.missing_keys if key.endswith(".v_mem")
    } | {k for k in load_result.missing_keys if k.endswith(".min_v_mem")}
    allowed_unexpected = {
        k for k in load_result.unexpected_keys if k.endswith(".min_v_mem")
    }
    if set(load_result.missing_keys) != allowed_missing or set(load_result.unexpected_keys) != allowed_unexpected:
        raise RuntimeError(
            f"--resume-checkpoint state_dict mismatch beyond the filtered .v_mem buffers "
            f"and any .min_v_mem None<->finite transition: "
            f"missing={set(load_result.missing_keys) - allowed_missing}, "
            f"unexpected={set(load_result.unexpected_keys) - allowed_unexpected}"
        )


@torch.no_grad()
def checkpoint_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone deployable state without transient, shape-dependent ``v_mem``."""

    model.reset_states()
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.endswith(".v_mem")
    }


def save_checkpoint_state_atomically(
    state_dict: dict[str, torch.Tensor], path: Path
) -> None:
    """Write an already-selected snapshot, then atomically promote it."""

    pending = path.with_name(path.name + ".pending")
    try:
        torch.save(state_dict, pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def save_checkpoint_atomically(model: nn.Module, path: Path) -> None:
    """Save a model snapshot without transient membrane state."""

    save_checkpoint_state_atomically(checkpoint_state_cpu(model), path)


def _artifact_value(value) -> str:
    """Stable, path-safe rendering used by automatic experiment names."""
    if value is None:
        return "none"
    return str(value).lower().replace("_", "-").replace(".", "p")


def _sampling_value(value: str) -> str:
    return {
        "pursuit_continuous": "pursuit",
        "main_peak_end": "saccade-main-peak-end",
        "corrective_peak": "saccade-corrective-peak",
    }.get(value, _artifact_value(value))


def _selection_value(value: str) -> str:
    return {
        "validation_loss": "val-loss",
        "validation_pixel_l2": "val-pixel-l2",
        "validation_angular_deg": "val-angle",
        "test_loss": "test-loss",
        "test_pixel_l2": "test-pixel-l2",
        "test_angular_deg": "test-angle",
    }[value]


def cache_identity(cache_dir: Path | None) -> str:
    """Compact data identity derived from cache metadata, not path spelling."""
    if cache_dir is None:
        return "auto"
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.is_file():
        return _artifact_value(Path(cache_dir).name)
    meta = json.loads(meta_path.read_text())
    split_manifest = meta.get("split_manifest") or {}
    split = split_manifest.get("strategy", "unspecified")
    if split == "pursuit_tail_cycle":
        cycle = split_manifest.get("cycle") or {}
        split = (
            f"tail-{cycle.get('train_windows_max', 'x')}-"
            f"{cycle.get('validation_windows', 'x')}-"
            f"{cycle.get('test_windows', 'x')}"
        )
    fields = [
        f"e{meta.get('slice_length', 'x')}",
        _artifact_value(split),
        f"i{meta.get('roi_size', 'x')}",
    ]
    train_stride = meta.get("train_stride")
    if train_stride is not None:
        fields.append(f"s{train_stride}")
    return "-".join(fields)


def default_experiment_name(args: argparse.Namespace) -> str:
    """Human-readable experiment identity; config.json remains authoritative.

    ``key=value`` fields avoid ambiguous concatenations such as
    ``eyesleft_lossmse`` and make result directories easy to index with a
    filesystem search. Major options that change training behavior or data
    selection are represented here; config.json remains the complete index.
    """
    if args.users is not None:
        split_text = f"train={','.join(map(str, args.users))};val={','.join(map(str, args.val_users))}"
        split_id = hashlib.sha256(split_text.encode()).hexdigest()[:8]
        population = f"users={len(args.users)}train-{len(args.val_users)}val-{split_id}__eye=0"
    else:
        population = "eye=0"
    parts = [population, "model=conv8bn", f"task={_artifact_value(args.task)}"]
    if args.task == "regression":
        parts += [f"loss={_artifact_value(args.loss)}", f"digits={args.num_digits}"]
    parts += [
        f"sampling={_sampling_value(args.valid_start_point)}",
        f"bins={args.num_steps}",
        f"thr={_artifact_value(args.spike_threshold)}",
        f"vmin={_artifact_value(args.min_v_mem)}",
        f"clip={_artifact_value(args.weight_clip_value)}",
        f"select={_selection_value(args.selection_metric)}",
        f"data={cache_identity(args.cache_dir)}",
        f"ep={args.epochs}",
        f"bs={args.batch_size}",
        f"lr={_artifact_value(args.lr)}",
        f"seed={args.seed}",
    ]
    if args.cache_dir is None:
        parts.append(f"input={args.roi_size}")
    if args.resume_checkpoint is not None:
        parts.append("resume=yes")
    return "__".join(parts)


def clip_conv_weights(model: nn.Module, clip_value: float | None) -> None:
    if clip_value is None or clip_value <= 0:
        return
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.clamp_(min=-clip_value, max=clip_value)


def build_criterion(task: str, loss_name: str):
    if task == "classification":
        return nn.CrossEntropyLoss()
    if loss_name == "normalized_mse":
        return nn.MSELoss()
    if loss_name == "angular_cosine":
        return ann_training.angular_cosine_loss
    raise ValueError(f"unsupported regression loss: {loss_name!r}")


def decode_positional_rates(digit_rates: torch.Tensor) -> torch.Tensor:
    """Decode positional rates without modifying or quantizing a label."""
    num_digits = digit_rates.shape[-1]
    weights = torch.tensor(
        [0.1**i for i in range(num_digits)],
        device=digit_rates.device,
        dtype=digit_rates.dtype,
    )
    return (digit_rates * weights).sum(dim=-1)


def positional_rate_readout(
    raw_output: torch.Tensor,
    num_digits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """raw_output: (B, T, 2*num_digits), 通道顺序 [row_digit_0..D-1,
    col_digit_0..D-1] (digit_0 最粗), 见 model.py。返回
    (pred_norm, digit_rates): pred_norm 是 (B, 2) 的还原坐标 [row, col];
    digit_rates 是 (B, 2, num_digits)，供输出分布和饱和率审计使用。

    每一位的脉冲计数除以固定的 10, 不是除以 --num-steps (T): 单个
    digit 的输出档位是 {0, 0.1, ..., 1.0}。注意 MultiSpike 并不把神经元限制成每个 bin 最多一个
    脉冲, 所以 T=10 不构成计数 10 的物理上限; 发散的输出头可以产生更大的计数。
    这里把发放率和还原坐标都夹到 [0,1]。"""
    batch = raw_output.shape[0]
    counts = raw_output.reshape(batch, raw_output.shape[1], 2, num_digits)
    digit_rates = (counts.sum(dim=1) / 10.0).clamp(0.0, 1.0)
    pred_norm = decode_positional_rates(digit_rates).clamp(0.0, 1.0)
    return pred_norm, digit_rates


def classification_readout(raw_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """raw_output: (B, T, 22) -> (row_logits, col_logits), each (B, 11),
    summed spike counts over T used directly as CrossEntropy logits."""
    counts = raw_output.sum(dim=1)
    return counts[:, :11], counts[:, 11:]


def run_epoch(
    model, loader, criterion, device, task, row_values=None, col_values=None, optimizer=None,
    weight_clip_value=None, num_digits=1,
):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_px_err = 0.0
    total_ang_err = 0.0
    total_row_err = 0.0
    total_col_err = 0.0
    total_row_correct = 0
    total_col_correct = 0
    total_joint_correct = 0
    n_samples = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            model.reset_states()
            raw_output = model(frames)
            batch_n = frames.size(0)

            if task == "classification":
                row_target, col_target = labels[:, 0], labels[:, 1]
                row_logits, col_logits = classification_readout(raw_output)
                loss = criterion(row_logits, row_target) + criterion(col_logits, col_target)
            else:
                # Every digit count is decoded first. All head variants are
                # then optimized against the same unmodified continuous
                # coordinate, so changing num_digits changes representation
                # capacity rather than the target definition.
                pred_norm, _ = positional_rate_readout(raw_output, num_digits)
                loss = criterion(pred_norm, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_conv_weights(model, weight_clip_value)

            if task == "classification":
                row_pred = row_logits.detach().argmax(dim=1)
                col_pred = col_logits.detach().argmax(dim=1)
                pred_norm = torch.stack(ann_dataloader.normalize_label(row_values[row_pred], col_values[col_pred]), dim=1)
                true_norm = torch.stack(ann_dataloader.normalize_label(row_values[row_target], col_values[col_target]), dim=1)
                total_row_correct += (row_pred == row_target).sum().item()
                total_col_correct += (col_pred == col_target).sum().item()
                total_joint_correct += ((row_pred == row_target) & (col_pred == col_target)).sum().item()
            else:
                pred_norm = pred_norm.detach()
                true_norm = labels

            row_err, col_err = ann_training.per_axis_pixel_error(pred_norm, true_norm)
            total_loss += loss.item() * batch_n
            total_px_err += ann_training.pixel_l2_error(pred_norm, true_norm).item() * batch_n
            total_ang_err += ann_training.angular_error_deg(pred_norm, true_norm).item() * batch_n
            total_row_err += row_err.item() * batch_n
            total_col_err += col_err.item() * batch_n
            n_samples += batch_n

    result = {
        "loss": total_loss / n_samples,
        "pixel_l2": total_px_err / n_samples,
        "angular_deg": total_ang_err / n_samples,
        "row_px": total_row_err / n_samples,
        "col_px": total_col_err / n_samples,
        "n_samples": n_samples,
    }
    if task == "classification":
        result.update(
            row_acc=total_row_correct / n_samples,
            col_acc=total_col_correct / n_samples,
            joint_acc=total_joint_correct / n_samples,
        )
    return result


def max_abs_conv_weight(model: nn.Module) -> float:
    """训练诊断: 所有卷积权重绝对值的最大值。与 --weight-clip-value 配合看,
    用来确认裁剪确实生效。"""
    values = [
        float(module.weight.detach().abs().max())
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    return max(values) if values else 0.0


def main() -> int:
    args = parse_args()
    if args.num_digits < 1:
        raise SystemExit("--num-digits must be at least 1")
    if args.num_digits > 1 and args.task != "regression":
        raise SystemExit("--num-digits > 1 is only meaningful with --task regression")
    pooled_mode = args.users is not None
    if args.user is None and not pooled_mode:
        raise SystemExit("provide --user for single-user training or --users for pooled training")
    if args.user is not None and pooled_mode:
        raise SystemExit("--user and --users are mutually exclusive")
    if pooled_mode and not args.val_users:
        raise SystemExit("--users requires --val-users")
    if pooled_mode and set(args.users) & set(args.val_users or ()):
        raise SystemExit("--users and --val-users must be disjoint")
    if pooled_mode and args.cache_dir is not None:
        raise SystemExit("--cache-dir is only supported by single-user training")
    if args.selection_metric is None:
        args.selection_metric = (
            "validation_pixel_l2"
            if pooled_mode or args.cache_dir is not None
            else "test_loss"
        )
    if args.cache_dir is not None and not args.selection_metric.startswith("validation_"):
        raise SystemExit("--cache-dir requires a validation_* --selection-metric; test must remain locked")
    if pooled_mode and not args.selection_metric.startswith("validation_"):
        raise SystemExit("--users requires a validation_* --selection-metric")
    if (
        not pooled_mode
        and args.valid_start_point == "pursuit_continuous"
        and args.cache_dir is None
    ):
        raise SystemExit(
            "smooth pursuit requires --cache-dir pointing to train/validation/test; "
            "the legacy auto-resolved cache has no validation split"
        )
    if args.resume_checkpoint is not None and args.out_dir is None:
        raise SystemExit(
            "--resume-checkpoint requires an explicit --out-dir -- default_experiment_name() "
            "无法编码是从哪个 checkpoint 续训的, 自动命名会和另一次续训撞目录。"
        )
    torch.manual_seed(args.seed)

    user_dir = "population=pooled" if pooled_mode else f"user={args.user}"
    out_dir = args.out_dir or (DEFAULT_RESULT_ROOT / user_dir / default_experiment_name(args))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to reuse non-empty experiment directory {out_dir}; choose a new --out-dir "
            "so stale checkpoints/history cannot be mistaken for this run"
        )
    device = torch.device(args.device)

    if pooled_mode:
        ds_kwargs = dict(num_steps=args.num_steps, label_mode=args.task, valid_start_point=args.valid_start_point, downsampled_to=args.roi_size)
        train_set = build_pooled_dataset(args.dataset_root, args.users, **ds_kwargs)
        evaluation_set = build_pooled_dataset(args.dataset_root, args.val_users, **ds_kwargs)
        evaluation_split = "validation"
    else:
        train_set = build_dataset(
            args.dataset_root, args.user, "train", num_steps=args.num_steps, label_mode=args.task,
            valid_start_point=args.valid_start_point, downsampled_to=args.roi_size, cache_dir=args.cache_dir,
        )
        evaluation_split = "validation" if args.cache_dir is not None else "test"
        evaluation_set = build_dataset(
            args.dataset_root, args.user, evaluation_split, num_steps=args.num_steps, label_mode=args.task,
            valid_start_point=args.valid_start_point, downsampled_to=args.roi_size, cache_dir=args.cache_dir,
        )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    evaluation_loader = DataLoader(
        evaluation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = SpikingNet8LayerBN(
        num_timesteps=args.num_steps, task=args.task, num_digits=args.num_digits,
        spike_threshold=args.spike_threshold, min_v_mem=args.min_v_mem,
    ).to(device)
    if args.resume_checkpoint is not None:
        load_resume_checkpoint(model, args.resume_checkpoint, device)

    row_values = col_values = None
    if args.task == "classification":
        # EbvEyeSpikeDataset 本身不存的这两个属性；11 档实际像素值是
        # ann.dataloader 模块级常量（row_col_to_class 反查表的来源），
        # 分类 readout 用它们把预测的类别下标换回像素坐标。
        row_values = torch.tensor(ann_dataloader.ROW_VALUES, device=device, dtype=torch.float32)
        col_values = torch.tensor(ann_dataloader.COL_VALUES, device=device, dtype=torch.float32)

    criterion = build_criterion(args.task, args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device: {device}")
    log(f"out_dir: {out_dir}")
    if args.resume_checkpoint is not None:
        log(f"resumed from: {args.resume_checkpoint}")
    if pooled_mode:
        log(f"pooled train users: {args.users}; val users: {args.val_users}")
    log(f"train samples: {len(train_set)}, {evaluation_split} samples: {len(evaluation_set)}")
    log(f"model parameters: {count_parameters(model):,}")
    log(
        f"task={args.task} loss={args.loss} digits={args.num_digits} "
        f"time_bins={args.num_steps} continuous_targets=true "
        f"threshold={args.spike_threshold} min_v_mem={args.min_v_mem} "
        f"selection_metric={args.selection_metric}"
    )

    history = []
    best_selection_value = float("inf")
    best_checkpoint_saved = False
    best_checkpoint_epoch = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_result = run_epoch(
            model, train_loader, criterion, device, args.task, row_values, col_values,
            optimizer=optimizer, weight_clip_value=args.weight_clip_value, num_digits=args.num_digits,
        )
        evaluation_result = run_epoch(
            model, evaluation_loader, criterion, device, args.task, row_values, col_values,
            num_digits=args.num_digits,
        )
        scheduler.step(evaluation_result["loss"])
        elapsed = time.time() - start

        entry = {
            "epoch": epoch,
            "train_loss": train_result["loss"], "train_pixel_l2": train_result["pixel_l2"],
            "train_angular_deg": train_result["angular_deg"],
            "train_row_px": train_result["row_px"], "train_col_px": train_result["col_px"],
            "lr": optimizer.param_groups[0]["lr"], "elapsed_s": elapsed,
            "max_abs_conv_weight": max_abs_conv_weight(model),
        }
        entry.update({
            f"{evaluation_split}_loss": evaluation_result["loss"],
            f"{evaluation_split}_pixel_l2": evaluation_result["pixel_l2"],
            f"{evaluation_split}_angular_deg": evaluation_result["angular_deg"],
            f"{evaluation_split}_row_px": evaluation_result["row_px"],
            f"{evaluation_split}_col_px": evaluation_result["col_px"],
        })
        if args.task == "classification":
            entry.update(
                train_row_acc=train_result["row_acc"], train_col_acc=train_result["col_acc"],
                train_joint_acc=train_result["joint_acc"],
                **{
                    f"{evaluation_split}_row_acc": evaluation_result["row_acc"],
                    f"{evaluation_split}_col_acc": evaluation_result["col_acc"],
                    f"{evaluation_split}_joint_acc": evaluation_result["joint_acc"],
                },
            )
        history.append(entry)

        msg = (
            f"epoch {epoch:4d}/{args.epochs} train_loss={train_result['loss']:.5f} train_px={train_result['pixel_l2']:.2f} "
            f"train_deg={train_result['angular_deg']:.3f} {evaluation_split}_loss={evaluation_result['loss']:.5f} "
            f"{evaluation_split}_px={evaluation_result['pixel_l2']:.2f} "
            f"{evaluation_split}_deg={evaluation_result['angular_deg']:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"maxw={entry['max_abs_conv_weight']:.3f} ({elapsed:.1f}s)"
        )
        if args.task == "classification":
            msg += f" {evaluation_split}_joint_acc={evaluation_result['joint_acc']:.3f}"
        log(msg)

        selection_value = entry.get(args.selection_metric)
        if selection_value is not None and selection_value < best_selection_value:
            best_selection_value = selection_value
            save_checkpoint_atomically(model, out_dir / "best_checkpoint.pt")
            best_checkpoint_saved = True
            best_checkpoint_epoch = epoch
            entry["saved_as_best_checkpoint"] = True

    save_checkpoint_atomically(model, out_dir / "final_checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    config = vars(args).copy()
    config["completed_epochs"] = len(history)
    config["final_checkpoint_saved"] = True
    config["best_checkpoint_saved"] = best_checkpoint_saved
    config["best_checkpoint_epoch"] = best_checkpoint_epoch
    config["best_selection_value"] = best_selection_value if best_checkpoint_saved else None
    config["training_evaluation_split"] = evaluation_split
    config["artifact_schema_version"] = 2
    config["experiment_name"] = out_dir.name
    config["target_policy"] = "continuous_unquantized"
    config["unique_output_levels_per_axis"] = (
        10**args.num_digits + 1 if args.task == "regression" else None
    )
    for key in ("dataset_root", "cache_dir", "out_dir", "resume_checkpoint"):
        config[key] = str(config[key]) if config[key] is not None else None
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"\nWrote history/config/checkpoints to {out_dir}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
