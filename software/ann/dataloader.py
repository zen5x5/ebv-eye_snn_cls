#!/usr/bin/env python3
"""PyTorch Dataset for the ebv-eye ANN pipeline (Tan et al.-style single
accumulated event frame per sample, as opposed to the SNN line's temporal
spike sequence).

Loads the per-user/eye train.npz/test.npz produced by
code/ebv-eye/dataset_preprocessing.py (see .agent/ebv-eye.md) and rasterizes each
500-event sample into a 2-channel (polarity) event-count frame at ROI
resolution. Tan et al. describe this step only as "each event's coordinates
map to a specific memory address, forming a 2-channel feature map" --
they do not specify count vs. binary vs. normalized accumulation. This
project uses raw per-pixel event counts (float32), the most direct reading
of that description.

The cached .npz files store raw, **unflipped** events in ROI-local
coordinates -- no mirroring is baked into the cache. Any event-frame flip
and/or label mirroring is applied on the fly in `__getitem__`, so it can be
toggled per experiment without re-running preprocessing.

Right-eye handling (configurable, for comparison experiments):
- `flip_events`: horizontally flip the accumulated event frame (Tan et al.'s
  choice: right eye flipped so both eyes share one appearance convention).
- `mirror_label`: horizontally mirror the screen-coordinate label about the
  screen's horizontal center (col' = 1920 - col; row is untouched, since the
  mirror axis is only horizontal). Tan et al. do *not* mirror the label --
  both eyes fixate the same physical point at once, so the true gaze
  target is identical regardless of which eye's frame is flipped. Defaults
  to False to match that; expose it anyway so the (event-flip x
  label-mirror) 2x2 can be compared empirically.

Labels are normalized to [0,1] using the ebv-eye 11x11 grid's known bounds
(row in [150,930], col in [150,1770] -- see .agent/ebv-eye.md "121 个注视点"); the
grid is symmetric about the screen's horizontal center (960px, per the
official README), so mirrored labels still fall inside [COL_MIN, COL_MAX].
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

ROW_MIN, ROW_MAX = 150.0, 930.0
COL_MIN, COL_MAX = 150.0, 1770.0
SCREEN_WIDTH_PX = 1920  # official monitor resolution; horizontal mirror axis
SCREEN_HEIGHT_PX = 1080
SCREEN_DIAGONAL_INCH = 40.0  # Sceptre X415BV_FSR, per Angelopoulos et al. / .agent/ebv-eye.md
VIEW_DISTANCE_MM = 400.0  # 40cm viewing distance, per Angelopoulos et al. / .agent/ebv-eye.md
AER_SENSOR_HEIGHT = 260  # DAVIS346b event-sensor resolution -- see dataset_analysis.SENSOR_HEIGHT / .agent/ebv-eye.md
AER_SENSOR_WIDTH = 346   # (distinct from SCREEN_HEIGHT_PX/SCREEN_WIDTH_PX above, which are the monitor's)


def screen_mm_per_pixel() -> float:
    """Physical screen size derived from the 40" diagonal at the dataset's
    1920x1080 resolution (16:9), assuming square pixels -- not stated
    directly in the papers, but follows from the documented diagonal +
    resolution. Used to convert pixel-space errors into physical/angular
    ones for the angular-error metric."""
    diagonal_mm = SCREEN_DIAGONAL_INCH * 25.4
    aspect = SCREEN_WIDTH_PX / SCREEN_HEIGHT_PX
    height_mm = diagonal_mm / (aspect**2 + 1) ** 0.5
    return height_mm / SCREEN_HEIGHT_PX


def normalize_label(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_n = (row - ROW_MIN) / (ROW_MAX - ROW_MIN)
    col_n = (col - COL_MIN) / (COL_MAX - COL_MIN)
    return row_n, col_n


def denormalize_label(row_n, col_n):
    row = row_n * (ROW_MAX - ROW_MIN) + ROW_MIN
    col = col_n * (COL_MAX - COL_MIN) + COL_MIN
    return row, col


# The saccade grid's 11 distinct row values and 11 distinct col values (see
# .agent/ebv-eye.md "121 个注视点") -- every sample's label is always exactly one of
# these, never jittered off-grid. Used by the classification task, which
# treats row and col as two independent 11-way categorical targets instead
# of continuous regression.
ROW_VALUES = (150.0, 228.0, 306.0, 384.0, 462.0, 540.0, 618.0, 696.0, 774.0, 852.0, 930.0)
COL_VALUES = (150.0, 312.0, 474.0, 636.0, 798.0, 960.0, 1122.0, 1284.0, 1446.0, 1608.0, 1770.0)

_ROW_CLASS_BY_VALUE = {int(v): i for i, v in enumerate(ROW_VALUES)}
_COL_CLASS_BY_VALUE = {int(v): i for i, v in enumerate(COL_VALUES)}


def row_col_to_class(row: float, col: float) -> tuple[int, int]:
    """Exact lookup into the known 11x11 grid -- no binning/thresholding
    needed since labels never fall between grid points."""
    return _ROW_CLASS_BY_VALUE[int(round(row))], _COL_CLASS_BY_VALUE[int(round(col))]


def compute_sqrt_target_weights(segment_index: np.ndarray) -> np.ndarray:
    """Per-sample training weight = 1/sqrt(n_target), n_target = how many
    times this sample's segment_index (== target, 1:1 for this dataset's
    saccade paradigm -- see .agent/ebv-eye.md) appears in *this* dataset. This is
    inverse-*square-root*-frequency reweighting, not full inverse frequency
    (1/n_target) -- a deliberately gentler middle ground between no
    reweighting and fully equalizing every target's contribution (see
    .agent/ebv-eye.md "sample-micro 与 target-macro 指标" / the sqrt-weighting
    discussion): full inverse frequency would give some targets 60x+ the
    per-sample gradient weight of others (this dataset's corrective_peak
    train counts range 4..265), which is enough to make training unstable;
    sqrt narrows that to ~8x while still correcting the imbalance.

    Weights are normalized to mean 1.0 over the dataset, so the overall
    loss/gradient scale matches the unweighted case and existing lr
    defaults remain valid without retuning."""
    unique, counts = np.unique(segment_index, return_counts=True)
    count_by_segment = dict(zip(unique.tolist(), counts.tolist()))
    raw = np.array([1.0 / np.sqrt(count_by_segment[int(s)]) for s in segment_index], dtype=np.float64)
    return (raw / raw.mean()).astype(np.float32)


def events_to_frame(events: np.ndarray, roi_size: int) -> np.ndarray:
    """events: structured array (one sample) with fields p, row, col, t,
    already in ROI-local coordinates. Returns float32 (2, roi_size, roi_size):
    channel 0 = polarity-0 event counts, channel 1 = polarity-1 counts."""
    frame = np.zeros((2, roi_size, roi_size), dtype=np.float32)
    rows = events["row"].astype(np.int64)
    cols = events["col"].astype(np.int64)
    pol = events["p"].astype(np.int64)
    valid = (rows >= 0) & (rows < roi_size) & (cols >= 0) & (cols < roi_size) & (pol >= 0) & (pol < 2)
    np.add.at(frame, (pol[valid], rows[valid], cols[valid]), 1.0)
    return frame


class EbvEyeFrameDataset(Dataset):
    """One item = one accumulated 2-channel event frame + a label.

    label_mode='regression' (default): label is normalized [row_n,col_n] in
    [0,1] (float32), for the continuous-regression models.
    label_mode='classification': label is [row_class,col_class] (int64),
    each in [0,10] -- exact indices into ROW_VALUES/COL_VALUES -- for the
    two-head classification model.

    The event-frame flip and label mirror are independent switches, applied
    fresh on every __getitem__ call (nothing is pre-flipped in the cache).

    target_weight=None (default): __getitem__ returns (frame, label), as
    before. target_weight='sqrt': also loads this split's segment_index
    (already in the cache -- see dataset_preprocessing.py) and computes
    per-sample weights via compute_sqrt_target_weights; __getitem__ then
    returns a (frame, label, weight) triple instead -- see
    training.run_epoch_weighted. Intended for the *train* split only (test
    metrics should stay plain sample-micro for comparability -- callers are
    responsible for not setting this on the test split)."""

    def __init__(
        self,
        npz_path: Path,
        roi_size: int | None = None,
        flip_events: bool = False,
        mirror_label: bool = False,
        label_mode: str = "regression",
        target_weight: str | None = None,
    ):
        if label_mode not in ("regression", "classification"):
            raise ValueError(f"label_mode must be 'regression' or 'classification', got {label_mode!r}")
        if target_weight not in (None, "sqrt"):
            raise ValueError(f"target_weight must be None or 'sqrt', got {target_weight!r}")
        npz_path = Path(npz_path)
        data = np.load(npz_path)
        self.events = data["events"]  # (N, slice_length) structured
        self.labels = data["labels"].astype(np.float32)  # (N, 2) raw [row, col]
        self.flip_events = flip_events
        self.mirror_label = mirror_label
        self.label_mode = label_mode
        self.weights = compute_sqrt_target_weights(data["segment_index"]) if target_weight == "sqrt" else None

        if roi_size is None:
            meta_path = npz_path.parent / "meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(f"roi_size not given and no meta.json next to {npz_path}")
            roi_size = json.loads(meta_path.read_text())["roi_size"]
        self.roi_size = roi_size

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        frame = events_to_frame(self.events[idx], self.roi_size)
        if self.flip_events:
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        row, col = self.labels[idx]
        if self.mirror_label:
            col = SCREEN_WIDTH_PX - col

        if self.label_mode == "classification":
            row_class, col_class = row_col_to_class(row, col)
            label = torch.tensor([row_class, col_class], dtype=torch.long)
        else:
            row_n, col_n = normalize_label(row, col)
            label = torch.tensor([row_n, col_n], dtype=torch.float32)

        if self.weights is not None:
            return torch.from_numpy(frame), label, torch.tensor(self.weights[idx])
        return torch.from_numpy(frame), label


def build_dataset(
    dataset_root: Path,
    user: int,
    split: str,
    eyes: str = "both",
    right_flip_events: bool = True,
    right_mirror_label: bool = False,
    valid_start_point: str = "main_peak_end",
    downsampled_to: int | None = None,
    label_mode: str = "regression",
    target_weight: str | None = None,
) -> Dataset:
    """eyes: 'left' (eye0 only), 'right' (eye1 only), or 'both' (concatenated).
    right_flip_events / right_mirror_label only affect the right eye (eye0 is
    always used as-is, unflipped, unmirrored -- it is the canonical
    orientation the right eye is aligned to).

    valid_start_point selects which dataset_preprocessing.py cache to load:
    Every value, including 'main_peak_end', reads the matching
    eye_data/user{N}/{eye}/{valid_start_point}_start/ subdirectory -- see
    dataset_preprocessing.py's --valid-start-point / out-dir logic.

    downsampled_to, if given, reads the pre-baked dataset_preprocessing.py `downsample`-subcommand output
    eye_data/user{N}/{eye}/{valid_start_point}_start_downsampled{N}/ instead
    (e.g. downsampled_to=128 for a 2x-downsampled 256x256 ROI).

    label_mode: see EbvEyeFrameDataset.

    target_weight: see EbvEyeFrameDataset -- forwarded to every eye's
    dataset unchanged. Callers should only set this when building the
    *train* split (leave it None for test/eval)."""
    if eyes not in ("left", "right", "both"):
        raise ValueError(f"eyes must be 'left', 'right', or 'both', got {eyes!r}")

    eye_data = Path(dataset_root) / "eye_data" / f"user{user}"

    def split_dir(eye: str) -> Path:
        eye_dir = eye_data / eye
        if downsampled_to is not None:
            src_suffix = f"{valid_start_point}_start"
            return eye_dir / f"{src_suffix}_downsampled{downsampled_to}"
        return eye_dir / f"{valid_start_point}_start"

    parts = []
    if eyes in ("left", "both"):
        parts.append(
            EbvEyeFrameDataset(
                split_dir("0") / f"{split}.npz",
                flip_events=False,
                mirror_label=False,
                label_mode=label_mode,
                target_weight=target_weight,
            )
        )
    if eyes in ("right", "both"):
        parts.append(
            EbvEyeFrameDataset(
                split_dir("1") / f"{split}.npz",
                flip_events=right_flip_events,
                mirror_label=right_mirror_label,
                label_mode=label_mode,
                target_weight=target_weight,
            )
        )
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def load_split(dataset_root: Path, user: int, split: str) -> Dataset:
    """Backward-compatible alias for build_dataset(..., eyes='both') with
    Tan et al.'s defaults (right eye flipped, label not mirrored)."""
    return build_dataset(dataset_root, user, split, eyes="both", right_flip_events=True, right_mirror_label=False)


def build_pooled_dataset(
    dataset_root: Path,
    users: list[int],
    eye: str = "0",
    valid_start_point: str = "corrective_peak",
    downsampled_to: int | None = 128,
    label_mode: str = "regression",
) -> Dataset:
    """Concatenates multiple users' `dataset_preprocessing.py fixed
    --no-split` pooled caches (eye_data/user{N}/{eye}/
    {valid_start_point}_start_pooled[_downsampled{M}]/all.npz) into one
    Dataset -- for multi-user pooled training, where the train/val/test
    split is decided at the *user* level (call this once per split with
    that split's own user list, e.g. once for the training users, once for
    a held-out set of validation users) rather than at the window level
    within a single user's data (unlike build_dataset's train.npz/test.npz,
    which splits within one user).

    Reuses EbvEyeFrameDataset completely unchanged -- it already reads any
    npz path with the events/labels/segment_index schema, not specifically
    files named train.npz/test.npz -- and torch's ConcatDataset, the same
    mechanism build_dataset already uses to join left+right eye data. No
    new Dataset class needed.

    Only eye0 has been preprocessed with --no-split so far (see
    .agent/ebv-eye.md), so `eye` defaults to "0"; there is no flip_events/
    mirror_label here since this pooled line hasn't touched the right eye
    at all yet -- add that if/when eye1 gets pooled-preprocessed too."""
    if not users:
        raise ValueError("users must be a non-empty list")
    parts = []
    for user in users:
        eye_dir = Path(dataset_root) / "eye_data" / f"user{user}" / eye
        if downsampled_to is not None:
            split_dir = eye_dir / f"{valid_start_point}_start_pooled_downsampled{downsampled_to}"
        else:
            split_dir = eye_dir / f"{valid_start_point}_start_pooled"
        parts.append(EbvEyeFrameDataset(split_dir / "all.npz", label_mode=label_mode))
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
