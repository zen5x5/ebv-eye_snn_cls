#!/usr/bin/env python3
"""Train/test the ANN gaze-regression model, matching Tan et al.'s stated
training setup (Section IV-C-2, "Network Training"):

- Adam optimizer, initial learning rate 0.01.
- ReduceLROnPlateau scheduler: factor 0.3, reduces when the test-set loss
  has not improved for 5 consecutive epochs.
- Batch size 64, 200 epochs.
- Both eyes combined into one train/test set, right eye's event frame
  horizontally flipped (see dataloader.py), doubling the sample count.
  Use --eyes/--right-flip-events/--right-mirror-label to run comparison
  experiments (left-only, right-only, unflipped right, mirrored labels).

The paper does not state its loss function. Two are implemented, via --loss:
- angular_cosine (default): 1-cos(angle) between predicted/true gaze
  direction vectors (see angular_cosine_loss) -- weights errors the way the
  angular-error metric actually does (equal pixel offsets near screen
  center subtend a larger angle than the same offset near the edge), rather
  than pixel-uniformly like normalized_mse. On user15 left-eye, this eliminated
  the severe test-set overfitting seen with normalized_mse (final-epoch test
  pixel error stayed at ~73px vs. normalized_mse's collapse to ~242px -- see
  .agent/ebv-eye.md) while also reaching a *better* best-epoch accuracy.
- normalized_mse: MSE on normalized [row, col] coordinates. Kept only for
  comparison; not recommended as a default given the above.

--task classification switches to a wholly different formulation: the
saccade grid has exactly 11 distinct row values and 11 distinct col values
(see dataloader.ROW_VALUES/COL_VALUES), so instead of continuous [0,1]
regression, row and col are each treated as an independent 11-way
categorical target and trained with cross-entropy per head
(TanStyleGazeNet128Classifier or TanStyleGazeNet128NativeClassifier,
selected via --model -- see model.py). --loss is ignored in this mode.
Pixel/angular errors are still reported (by
decoding the predicted class back to its exact grid coordinate) so results
stay comparable to the regression result tables.

Usage:
    python training.py --user 15 --smoke                # a few batches, sanity check only
    python training.py --user 15                         # full 200-epoch run, default angular_cosine loss
    python training.py --user 15 --loss normalized_mse   # full 200-epoch run, normalized-coordinate MSE
    python training.py --user 15 --roi-size 128 --downsampled-cache --valid-start-point corrective_peak \\
        --task classification                             # two-head 11x11 classification on the 128x128 cache
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import (
    ROW_MAX,
    ROW_MIN,
    COL_MAX,
    COL_MIN,
    COL_VALUES,
    ROW_VALUES,
    SCREEN_WIDTH_PX,
    SCREEN_HEIGHT_PX,
    VIEW_DISTANCE_MM,
    build_dataset,
    build_pooled_dataset,
    denormalize_label,
    normalize_label,
    screen_mm_per_pixel,
)
from model import (
    TanStyleGazeNet,
    TanStyleGazeNet128,
    TanStyleGazeNet128Classifier,
    TanStyleGazeNet128Native,
    TanStyleGazeNet128NativeClassifier,
    count_parameters,
)

DEFAULT_DATASET_ROOT = Path("/data/xuzx/data/ebv-eye")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", type=int, default=None, help="Single-user mode (existing behavior): train/test split within this one user's data. Mutually exclusive with --users.")
    parser.add_argument(
        "--users", type=int, nargs="+", default=None,
        help="Multi-user pooled mode: train on the concatenation of these users' `dataset_preprocessing.py fixed "
        "--no-split` pooled caches (see dataloader.build_pooled_dataset), eye0 only. Requires --val-users too. "
        "Mutually exclusive with --user and --target-weight (untested "
        "combinations, not yet supported). Per-epoch validation_* metrics are computed on --val-users; "
        "true test users should be evaluated "
        "separately, after training, via evaluation.py/evaluation_report.py, and never referenced here.",
    )
    parser.add_argument(
        "--val-users", type=int, nargs="+", default=None,
        help="Required with --users: held-out users used only for per-epoch validation metrics.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--roi-size", type=int, default=256)
    parser.add_argument(
        "--downsampled-cache",
        action="store_true",
        help="Load the pre-baked cache downsampled to --roi-size (see dataset_preprocessing.py's `downsample` "
        "subcommand's {valid_start_point}_start_downsampled{roi_size}/ output) instead of a natively-cropped ROI cache",
    )
    parser.add_argument(
        "--model",
        choices=("default", "retuned", "native"),
        default="default",
        help="default: TanStyleGazeNet (sized for --roi-size, e.g. ~34k params if naively given a 128x128 "
        "input). retuned: TanStyleGazeNet128, a milder-first-pool variant that keeps ~83k params (matching "
        "Tan et al.'s ~91k budget) even at 128x128 input by reusing TanStyleGazeNet's exact channel/FC "
        "widths -- a retrofit for comparing against the 256x256 model, not a design chosen for 128x128 on "
        "its own terms. native: TanStyleGazeNet128Native, a from-scratch 128x128 capacity budget (~87k) with "
        "channels that grow through the network instead of staying flat, so FC1 no longer dominates the "
        "param count (~42%% vs retuned's ~79%%) -- see model.py. For --task classification on the square-ROI "
        "path, --model instead picks the backbone via TanStyleGazeNet128Classifier (default or retuned; "
        "'default' is a backward-compat alias for 'retuned' here -- there is no 34k-param classifier) or "
        "TanStyleGazeNet128NativeClassifier (native) -- see model.py. 2026-08-15: previously --model was "
        "silently ignored for classification (always retuned); fixed so run directory names disambiguate "
        "which backbone was used (see default_experiment_name()) -- the one pre-existing classification result "
        "directory was renamed accordingly, see .agent/ebv-eye.md.",
    )
    parser.add_argument(
        "--task",
        choices=("regression", "classification"),
        default="regression",
        help="regression (default): continuous [0,1] row/col via --model/--loss. classification: two 11-way "
        "heads (the saccade grid has exactly 11 distinct row/col values -- see dataloader.ROW_VALUES/"
        "COL_VALUES) trained with cross-entropy, summed 1:1 across the row/col heads; ignores --loss (--model "
        "still selects the backbone -- see --model's help). Uses TanStyleGazeNet128Classifier/"
        "TanStyleGazeNet128NativeClassifier.",
    )
    parser.add_argument("--eyes", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--valid-start-point",
        choices=("main_peak_end", "corrective_peak", "pursuit_continuous"),
        default="main_peak_end",
        help="Which dataset_preprocessing.py cache variant to load (see dataloader.build_dataset). "
        "pursuit_continuous is the smooth-pursuit continuous-label cache (see dataset_preprocessing.py) -- "
        "only meaningful with --task regression (its labels do not align with the 11x11 saccade grid, so "
        "--task classification cannot bin them).",
    )
    parser.add_argument(
        "--right-flip-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Horizontally flip the right eye's event frame (Tan et al.'s default: on)",
    )
    parser.add_argument(
        "--right-mirror-label",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Horizontally mirror the right eye's screen-coordinate label (Tan et al.: off; both eyes fixate the same point)",
    )
    parser.add_argument(
        "--loss",
        choices=("normalized_mse", "angular_cosine"),
        default="angular_cosine",
        help="angular_cosine (default): 1-cos(angle) between predicted/true gaze vectors, weighted to match "
        "angular error geometry -- empirically far more stable, no overfitting collapse (see .agent/ebv-eye.md). "
        "normalized_mse: MSE on normalized coordinates; kept for comparison, not recommended as a default.",
    )
    parser.add_argument(
        "--target-weight",
        choices=("none", "sqrt"),
        default="none",
        help="none (default): plain sample-micro training loss, every sample weighted equally. sqrt: weight "
        "each training sample by 1/sqrt(n_target) (n_target = how many train samples that target has), "
        "normalized to mean 1.0 -- a gentler middle ground than full inverse-frequency (1/n_target), which "
        "would give some targets 60x+ the gradient weight of others -- see dataloader.compute_sqrt_target_"
        "weights / .agent/ebv-eye.md. Only affects the training objective; reported train/test metrics stay plain "
        "unweighted sample-micro. Works for both --task regression "
        "and --task classification (classification support added after the first regression-only pass -- "
        "see run_epoch_classification_weighted).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <result-root>/user={N}/eyes={...}__task={...}__loss={...}__...",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a handful of batches only, to sanity-check the pipeline")
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def _artifact_value(value) -> str:
    if value is None:
        return "none"
    return str(value).lower().replace("_", "-").replace(".", "p")


def default_experiment_name(args: argparse.Namespace) -> str:
    """Build a stable, non-misleading name for an automatically named run.

    The seed is part of the identity because independently seeded runs must
    coexist. Parameters ignored by a task are deliberately omitted.
    """
    if args.users is not None:
        # Pooled multi-user mode -- no eyes/flip/mirror concept (eye0 only,
        # see build_pooled_dataset), identity is which/how-many users are in
        # each of train/val instead. The exact user lists are still in
        # config.json; this name is just enough to keep directories unique
        # and human-scannable, not a full encoding of the split.
        train_ids = "-".join(str(user) for user in args.users)
        val_ids = "-".join(str(user) for user in args.val_users)
        parts = [f"train-users={train_ids}", f"val-users={val_ids}", "eye=0"]
    else:
        # user identity is already encoded by the parent directory
        # (user={N}), so repeating it in every experiment name adds noise.
        parts = [f"eyes={args.eyes}"]
        if args.eyes in ("both", "right"):
            parts += [
                f"right-event-flip={int(args.right_flip_events)}",
                f"right-label-mirror={int(args.right_mirror_label)}",
            ]
    # task is always explicit (matches snn/training.py's convention) --
    # neither regression nor classification is a "default" the other omits
    # in favor of.
    parts.append(f"task={_artifact_value(args.task)}")
    if args.task != "classification":
        parts.append(f"loss={_artifact_value(args.loss)}")
    parts.append(f"sampling={_artifact_value(args.valid_start_point)}")
    if args.task == "classification":
        # Always explicit here (unlike regression's "omit if default"): the
        # one pre-existing classification run predates --model mattering
        # for this task and was retroactively renamed to include
        # '_modelretuned' (see .agent/ebv-eye.md) -- new runs should never again
        # produce a name that doesn't say which backbone was used.
        effective_model = "retuned" if args.model == "default" else args.model
        parts.append(f"model={_artifact_value(effective_model)}")
    else:
        parts.append(f"model={_artifact_value(args.model)}")
    if args.users is None and args.target_weight != "none":
        parts.append(f"target-weight={_artifact_value(args.target_weight)}")
    parts += [
        f"input-size={args.roi_size}",
        f"cache={'downsampled' if args.downsampled_cache else 'native'}",
        f"epochs={args.epochs}",
        f"batch-size={args.batch_size}",
        f"lr={_artifact_value(args.lr)}",
        f"seed={args.seed}",
    ]
    return "__".join(parts)


def pixel_l2_error(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean pixel distance after de-normalizing back to screen coordinates."""
    row_scale = ROW_MAX - ROW_MIN
    col_scale = COL_MAX - COL_MIN
    diff_row = (pred_norm[:, 0] - target_norm[:, 0]) * row_scale
    diff_col = (pred_norm[:, 1] - target_norm[:, 1]) * col_scale
    return torch.sqrt(diff_row**2 + diff_col**2).mean()


def per_axis_pixel_error(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean absolute per-axis pixel error: (row_error_px, col_error_px).
    Row = vertical (screen height 1080), col = horizontal (screen width
    1920) -- lets us check whether errors are actually balanced across the
    two axes, e.g. to sanity-check whether the wider horizontal axis needs
    extra loss weight (see .agent/ebv-eye.md "是否需要再手动给水平方向额外加权")."""
    row_scale = ROW_MAX - ROW_MIN
    col_scale = COL_MAX - COL_MIN
    diff_row = (pred_norm[:, 0] - target_norm[:, 0]) * row_scale
    diff_col = (pred_norm[:, 1] - target_norm[:, 1]) * col_scale
    return diff_row.abs().mean(), diff_col.abs().mean()


_MM_PER_PX = screen_mm_per_pixel()


def _gaze_vectors(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(Gpred, Ggt), each (B,3) = (x_mm, y_mm, L0) relative to screen center,
    at the subject's viewing distance L0 -- shared by angular_error_deg and
    angular_cosine_loss. Physical screen size is derived from the documented
    40" diagonal @ 1920x1080 (see dataloader.screen_mm_per_pixel); the
    papers give the angular-error metric but not an explicit mm/px
    conversion."""
    pred_row, pred_col = denormalize_label(pred_norm[:, 0], pred_norm[:, 1])
    gt_row, gt_col = denormalize_label(target_norm[:, 0], target_norm[:, 1])

    pred_x = (pred_col - SCREEN_WIDTH_PX / 2) * _MM_PER_PX
    pred_y = (pred_row - SCREEN_HEIGHT_PX / 2) * _MM_PER_PX
    gt_x = (gt_col - SCREEN_WIDTH_PX / 2) * _MM_PER_PX
    gt_y = (gt_row - SCREEN_HEIGHT_PX / 2) * _MM_PER_PX

    l0 = torch.full_like(pred_x, VIEW_DISTANCE_MM)
    g_pred = torch.stack([pred_x, pred_y, l0], dim=1)
    g_gt = torch.stack([gt_x, gt_y, l0], dim=1)
    return g_pred, g_gt


def angular_error_deg(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Mean angular error in degrees, following Tan et al. / Angelopoulos et
    al. Eq. (3): Eang = arccos((Ggt . Gpred) / (|Ggt| |Gpred|)). Evaluation
    metric only -- not used as the loss by default (arccos has unbounded
    gradient near cos=1; see angular_cosine_loss)."""
    g_pred, g_gt = _gaze_vectors(pred_norm, target_norm)
    cos_angle = (g_pred * g_gt).sum(dim=1) / (g_pred.norm(dim=1) * g_gt.norm(dim=1))
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_angle)).mean()


def angular_cosine_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """1 - cos(angle) between the predicted and true gaze direction vectors
    (equivalently, half the squared distance between their unit vectors).
    Smooth and well-behaved everywhere (unlike arccos, whose gradient blows
    up as cos->1, i.e. exactly as predictions become accurate); for small
    angles, 1-cos(theta) ~= theta^2/2, so it behaves like a squared error
    near convergence. This directly targets the same geometry as the
    angular-error evaluation metric, unlike plain pixel-space MSE, whose
    implicit per-pixel weighting is uniform across the screen even though
    equal pixel offsets near the center subtend a *larger* angle than the
    same offset near the edge (dtheta/dr = L0/(L0^2+r^2), maximal at r=0).

    reduction: 'mean' (default, matches nn.MSELoss's default) or 'none' --
    the latter returns the per-sample (B,) tensor unreduced, for
    run_epoch_weighted's per-sample weighting."""
    g_pred, g_gt = _gaze_vectors(pred_norm, target_norm)
    cos_angle = (g_pred * g_gt).sum(dim=1) / (g_pred.norm(dim=1) * g_gt.norm(dim=1))
    per_sample = 1.0 - cos_angle
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unsupported reduction: {reduction!r}")


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_px_err = 0.0
    total_ang_err = 0.0
    total_row_err = 0.0
    total_col_err = 0.0
    n_samples = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            preds = model(frames)
            loss = criterion(preds, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n = frames.size(0)
            row_err, col_err = per_axis_pixel_error(preds.detach(), labels)
            total_loss += loss.item() * batch_n
            total_px_err += pixel_l2_error(preds.detach(), labels).item() * batch_n
            total_ang_err += angular_error_deg(preds.detach(), labels).item() * batch_n
            total_row_err += row_err.item() * batch_n
            total_col_err += col_err.item() * batch_n
            n_samples += batch_n

    return {
        "loss": total_loss / n_samples,
        "pixel_l2": total_px_err / n_samples,
        "angular_deg": total_ang_err / n_samples,
        "row_px": total_row_err / n_samples,
        "col_px": total_col_err / n_samples,
        "n_samples": n_samples,
    }


def run_epoch_weighted(model, loader, per_sample_criterion, device, optimizer=None) -> dict:
    """Like run_epoch, but for per-target-weighted training (see
    dataloader.EbvEyeFrameDataset's target_weight='sqrt' option /
    .agent/ebv-eye.md's sqrt-frequency reweighting note): the loader yields (frame,
    label, weight) triples instead of (frame, label) pairs, and
    per_sample_criterion returns the *unreduced* per-sample loss (shape
    (B,), e.g. angular_cosine_loss(..., reduction="none") or
    nn.functional.mse_loss(..., reduction="none").mean(dim=1)). The
    backward pass uses sum(weight_i * loss_i) / sum(weight_i) instead of a
    plain mean -- weights are already normalized to mean 1.0 by the
    dataset, so this only *reweights* samples relative to each other, it
    doesn't change the overall loss scale.

    Reported metrics (loss/pixel_l2/angular_deg/row_px/col_px) stay the
    PLAIN unweighted sample-micro average -- same definition as run_epoch --
    so numbers are directly comparable to unweighted runs; only the
    optimizer's actual gradient signal changes, not what's logged or
    scheduled on. Intended for training batches only; use plain run_epoch
    for evaluation (dataloader.build_dataset's target_weight=None default
    already gives (frame, label) pairs for that)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_px_err = 0.0
    total_ang_err = 0.0
    total_row_err = 0.0
    total_col_err = 0.0
    n_samples = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for frames, labels, weights in loader:
            frames, labels, weights = frames.to(device), labels.to(device), weights.to(device)
            preds = model(frames)
            per_sample_loss = per_sample_criterion(preds, labels)
            weighted_loss = (per_sample_loss * weights).sum() / weights.sum()

            if train_mode:
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            batch_n = frames.size(0)
            row_err, col_err = per_axis_pixel_error(preds.detach(), labels)
            total_loss += per_sample_loss.detach().mean().item() * batch_n
            total_px_err += pixel_l2_error(preds.detach(), labels).item() * batch_n
            total_ang_err += angular_error_deg(preds.detach(), labels).item() * batch_n
            total_row_err += row_err.item() * batch_n
            total_col_err += col_err.item() * batch_n
            n_samples += batch_n

    return {
        "loss": total_loss / n_samples,
        "pixel_l2": total_px_err / n_samples,
        "angular_deg": total_ang_err / n_samples,
        "row_px": total_row_err / n_samples,
        "col_px": total_col_err / n_samples,
        "n_samples": n_samples,
    }


def run_epoch_classification(model, loader, criterion, device, row_values: torch.Tensor, col_values: torch.Tensor, optimizer=None) -> dict:
    """Same bookkeeping as run_epoch, for the two-head classifier. Loss is
    cross-entropy summed 1:1 across the row/col heads. Pixel/angular errors
    are computed by decoding the argmax class (and the true class) back to
    its exact grid coordinate via row_values/col_values, then reusing the
    same pixel_l2_error/angular_error_deg/per_axis_pixel_error functions as
    the regression task -- so results are directly comparable to the
    regression result tables (see .agent/ebv-eye.md)."""
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
            row_target, col_target = labels[:, 0], labels[:, 1]
            row_logits, col_logits = model(frames)
            loss = criterion(row_logits, row_target) + criterion(col_logits, col_target)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n = frames.size(0)
            row_pred = row_logits.detach().argmax(dim=1)
            col_pred = col_logits.detach().argmax(dim=1)

            pred_norm = torch.stack(normalize_label(row_values[row_pred], col_values[col_pred]), dim=1)
            true_norm = torch.stack(normalize_label(row_values[row_target], col_values[col_target]), dim=1)

            row_err, col_err = per_axis_pixel_error(pred_norm, true_norm)
            total_loss += loss.item() * batch_n
            total_px_err += pixel_l2_error(pred_norm, true_norm).item() * batch_n
            total_ang_err += angular_error_deg(pred_norm, true_norm).item() * batch_n
            total_row_err += row_err.item() * batch_n
            total_col_err += col_err.item() * batch_n
            total_row_correct += (row_pred == row_target).sum().item()
            total_col_correct += (col_pred == col_target).sum().item()
            total_joint_correct += ((row_pred == row_target) & (col_pred == col_target)).sum().item()
            n_samples += batch_n

    return {
        "loss": total_loss / n_samples,
        "pixel_l2": total_px_err / n_samples,
        "angular_deg": total_ang_err / n_samples,
        "row_px": total_row_err / n_samples,
        "col_px": total_col_err / n_samples,
        "row_acc": total_row_correct / n_samples,
        "col_acc": total_col_correct / n_samples,
        "joint_acc": total_joint_correct / n_samples,
        "n_samples": n_samples,
    }


def run_epoch_classification_weighted(model, loader, criterion, device, row_values: torch.Tensor, col_values: torch.Tensor, optimizer=None) -> dict:
    """run_epoch_classification's counterpart for per-target-weighted
    training (see run_epoch_weighted / dataloader.EbvEyeFrameDataset's
    target_weight='sqrt'): the loader yields (frame, label, weight)
    triples, and `criterion` must be constructed with reduction='none' --
    nn.CrossEntropyLoss supports this natively (returns a per-sample (B,)
    tensor), unlike the regression losses which need an extra per-sample
    reduction step (mse: .mean(dim=1); angular_cosine: its own reduction
    argument). Backward pass uses sum(weight_i * loss_i) / sum(weight_i);
    reported metrics stay the plain unweighted sample-micro average, same
    as run_epoch_classification."""
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
        for frames, labels, weights in loader:
            frames, labels, weights = frames.to(device), labels.to(device), weights.to(device)
            row_target, col_target = labels[:, 0], labels[:, 1]
            row_logits, col_logits = model(frames)
            per_sample_loss = criterion(row_logits, row_target) + criterion(col_logits, col_target)
            weighted_loss = (per_sample_loss * weights).sum() / weights.sum()

            if train_mode:
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            batch_n = frames.size(0)
            row_pred = row_logits.detach().argmax(dim=1)
            col_pred = col_logits.detach().argmax(dim=1)

            pred_norm = torch.stack(normalize_label(row_values[row_pred], col_values[col_pred]), dim=1)
            true_norm = torch.stack(normalize_label(row_values[row_target], col_values[col_target]), dim=1)

            row_err, col_err = per_axis_pixel_error(pred_norm, true_norm)
            total_loss += per_sample_loss.detach().mean().item() * batch_n
            total_px_err += pixel_l2_error(pred_norm, true_norm).item() * batch_n
            total_ang_err += angular_error_deg(pred_norm, true_norm).item() * batch_n
            total_row_err += row_err.item() * batch_n
            total_col_err += col_err.item() * batch_n
            total_row_correct += (row_pred == row_target).sum().item()
            total_col_correct += (col_pred == col_target).sum().item()
            total_joint_correct += ((row_pred == row_target) & (col_pred == col_target)).sum().item()
            n_samples += batch_n

    return {
        "loss": total_loss / n_samples,
        "pixel_l2": total_px_err / n_samples,
        "angular_deg": total_ang_err / n_samples,
        "row_px": total_row_err / n_samples,
        "col_px": total_col_err / n_samples,
        "row_acc": total_row_correct / n_samples,
        "col_acc": total_col_correct / n_samples,
        "joint_acc": total_joint_correct / n_samples,
        "n_samples": n_samples,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    pooled_mode = args.users is not None
    if args.user is None and not pooled_mode:
        raise SystemExit("must pass exactly one of --user or --users")
    if args.user is not None and pooled_mode:
        raise SystemExit("--user and --users are mutually exclusive")
    if pooled_mode and not args.val_users:
        raise SystemExit("--users requires --val-users (held-out users for per-epoch validation)")
    if pooled_mode and set(args.users) & set(args.val_users or ()):
        raise SystemExit(f"--users and --val-users overlap: {set(args.users) & set(args.val_users)}")
    if pooled_mode and args.target_weight != "none":
        raise SystemExit("--users (pooled mode) does not support --target-weight yet")

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        experiment_name = default_experiment_name(args)
        user_dir = "population=pooled" if pooled_mode else f"user={args.user}"
        out_dir = Path("/data/xuzx/result/02_snn_speck/ebv-eye/ann") / user_dir / experiment_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to reuse non-empty experiment directory {out_dir}; "
            "choose a new --out-dir so runs cannot overwrite each other"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "a")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # default=str keeps native JSON types (int/float/bool/None) instead of
    # stringifying everything, while still handling non-serializable values
    # like Path (--dataset-root/--out-dir) by falling back to str().
    config = vars(args).copy()
    config["artifact_schema_version"] = 2
    config["experiment_name"] = out_dir.name
    config["target_policy"] = (
        "continuous_unquantized" if args.task == "regression" else "exact_class"
    )
    evaluation_split = "validation" if pooled_mode else "test"
    config["training_evaluation_split"] = evaluation_split
    config["selection_metric"] = f"{evaluation_split}_loss"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    device = torch.device(args.device)
    log(f"device: {device}")
    log(f"out_dir: {out_dir}")

    if pooled_mode:
        ds_kwargs_pooled = dict(
            valid_start_point=args.valid_start_point,
            downsampled_to=args.roi_size if args.downsampled_cache else None,
            label_mode=args.task,
        )
        train_data = build_pooled_dataset(args.dataset_root, args.users, **ds_kwargs_pooled)
        evaluation_data = build_pooled_dataset(args.dataset_root, args.val_users, **ds_kwargs_pooled)
    else:
        ds_kwargs = dict(
            eyes=args.eyes,
            right_flip_events=args.right_flip_events,
            right_mirror_label=args.right_mirror_label,
            valid_start_point=args.valid_start_point,
            downsampled_to=args.roi_size if args.downsampled_cache else None,
            label_mode=args.task,
        )
        target_weight = None if args.target_weight == "none" else args.target_weight
        train_data = build_dataset(args.dataset_root, args.user, "train", target_weight=target_weight, **ds_kwargs)
        evaluation_data = build_dataset(args.dataset_root, args.user, "test", **ds_kwargs)
    if pooled_mode:
        log(f"pooled mode: {len(args.users)} train users {args.users}, {len(args.val_users)} val users {args.val_users} (eye0 only)")
    else:
        log(f"eyes={args.eyes} right_flip_events={args.right_flip_events} right_mirror_label={args.right_mirror_label}")
    log(f"train samples: {len(train_data)}, {evaluation_split} samples: {len(evaluation_data)}")

    if args.smoke:
        train_data = torch.utils.data.Subset(train_data, range(min(len(train_data), 8 * args.batch_size)))
        evaluation_data = torch.utils.data.Subset(
            evaluation_data, range(min(len(evaluation_data), 4 * args.batch_size))
        )
        args.epochs = 2
        log(
            f"--smoke: using {len(train_data)} train / {len(evaluation_data)} "
            f"{evaluation_split} samples, {args.epochs} epochs"
        )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    evaluation_loader = DataLoader(
        evaluation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.task == "classification":
        # 'default' is a backward-compat alias for 'retuned' here: there is
        # no 34k-param classifier, and pre-2026-08-15 classification runs
        # always used the retuned backbone with --model left at its
        # argparse default -- see --model's help / .agent/ebv-eye.md.
        classifier_cls = TanStyleGazeNet128NativeClassifier if args.model == "native" else TanStyleGazeNet128Classifier
        model = classifier_cls(roi_size=args.roi_size).to(device)
        log(f"model: {classifier_cls.__name__} (--model {args.model})")
    else:
        model_cls = {
            "default": TanStyleGazeNet,
            "retuned": TanStyleGazeNet128,
            "native": TanStyleGazeNet128Native,
        }[args.model]
        model = model_cls(roi_size=args.roi_size).to(device)
        log(f"model: {model_cls.__name__} (--model {args.model})")
    log(f"model parameters: {count_parameters(model):,}")

    if args.task == "classification":
        criterion = nn.CrossEntropyLoss()
        log("loss: cross_entropy (row head + col head, summed 1:1)")
        row_values_t = torch.tensor(ROW_VALUES, device=device)
        col_values_t = torch.tensor(COL_VALUES, device=device)
    elif args.loss == "normalized_mse":
        criterion = nn.MSELoss()
        log(f"loss: {args.loss}")
    else:
        criterion = angular_cosine_loss
        log(f"loss: {args.loss}")

    per_sample_criterion = None
    if args.target_weight != "none":
        if args.task == "classification":
            per_sample_criterion = nn.CrossEntropyLoss(reduction="none")
        elif args.loss == "normalized_mse":
            def per_sample_criterion(preds, labels):
                return nn.functional.mse_loss(preds, labels, reduction="none").mean(dim=1)
        else:
            def per_sample_criterion(preds, labels):
                return angular_cosine_loss(preds, labels, reduction="none")
        log(f"target_weight: {args.target_weight} (train loss only; test/eval metrics stay unweighted)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)

    history = []
    best_evaluation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if args.task == "classification" and per_sample_criterion is not None:
            train_metrics = run_epoch_classification_weighted(model, train_loader, per_sample_criterion, device, row_values_t, col_values_t, optimizer)
            evaluation_metrics = run_epoch_classification(model, evaluation_loader, criterion, device, row_values_t, col_values_t, optimizer=None)
        elif args.task == "classification":
            train_metrics = run_epoch_classification(model, train_loader, criterion, device, row_values_t, col_values_t, optimizer)
            evaluation_metrics = run_epoch_classification(model, evaluation_loader, criterion, device, row_values_t, col_values_t, optimizer=None)
        elif per_sample_criterion is not None:
            train_metrics = run_epoch_weighted(model, train_loader, per_sample_criterion, device, optimizer)
            evaluation_metrics = run_epoch(model, evaluation_loader, criterion, device, optimizer=None)
        else:
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            evaluation_metrics = run_epoch(model, evaluation_loader, criterion, device, optimizer=None)
        scheduler.step(evaluation_metrics["loss"])
        elapsed = time.time() - start

        lr_now = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pixel_l2": train_metrics["pixel_l2"],
            "train_angular_deg": train_metrics["angular_deg"],
            "train_row_px": train_metrics["row_px"],
            "train_col_px": train_metrics["col_px"],
            f"{evaluation_split}_loss": evaluation_metrics["loss"],
            f"{evaluation_split}_pixel_l2": evaluation_metrics["pixel_l2"],
            f"{evaluation_split}_angular_deg": evaluation_metrics["angular_deg"],
            f"{evaluation_split}_row_px": evaluation_metrics["row_px"],
            f"{evaluation_split}_col_px": evaluation_metrics["col_px"],
            "lr": lr_now,
            "elapsed_s": elapsed,
        }
        if args.task == "classification":
            record.update(
                {
                    "train_row_acc": train_metrics["row_acc"],
                    "train_col_acc": train_metrics["col_acc"],
                    "train_joint_acc": train_metrics["joint_acc"],
                    f"{evaluation_split}_row_acc": evaluation_metrics["row_acc"],
                    f"{evaluation_split}_col_acc": evaluation_metrics["col_acc"],
                    f"{evaluation_split}_joint_acc": evaluation_metrics["joint_acc"],
                }
            )
        history.append(record)
        log_msg = (
            f"epoch {epoch:4d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} train_px={train_metrics['pixel_l2']:.2f} train_deg={train_metrics['angular_deg']:.3f} "
            f"train_row={train_metrics['row_px']:.2f} train_col={train_metrics['col_px']:.2f} "
            f"{evaluation_split}_loss={evaluation_metrics['loss']:.5f} "
            f"{evaluation_split}_px={evaluation_metrics['pixel_l2']:.2f} "
            f"{evaluation_split}_deg={evaluation_metrics['angular_deg']:.3f} "
            f"{evaluation_split}_row={evaluation_metrics['row_px']:.2f} "
            f"{evaluation_split}_col={evaluation_metrics['col_px']:.2f} "
            f"lr={lr_now:.2e} ({elapsed:.1f}s)"
        )
        if args.task == "classification":
            log_msg += (
                f" train_rowacc={train_metrics['row_acc']:.3f} train_colacc={train_metrics['col_acc']:.3f} "
                f"train_jointacc={train_metrics['joint_acc']:.3f} "
                f"{evaluation_split}_rowacc={evaluation_metrics['row_acc']:.3f} "
                f"{evaluation_split}_colacc={evaluation_metrics['col_acc']:.3f} "
                f"{evaluation_split}_jointacc={evaluation_metrics['joint_acc']:.3f}"
            )
        log(log_msg)

        if not args.smoke and evaluation_metrics["loss"] < best_evaluation_loss:
            best_evaluation_loss = evaluation_metrics["loss"]
            torch.save(model.state_dict(), out_dir / "best_checkpoint.pt")

    if not args.smoke:
        torch.save(model.state_dict(), out_dir / "final_checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log(f"\nWrote history and checkpoints to {out_dir}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
