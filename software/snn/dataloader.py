#!/usr/bin/env python3
"""SNN-side dataset: reads the exact same dataset_preprocessing.py
(`fixed`/`downsample` subcommands) cache the ANN pipeline uses (train.npz/test.npz under
eye_data/user{N}/0/{valid_start_point}_start_downsampled{roi_size}/), but
rasterizes each 500-event sample into T genuine time bins instead of one
accumulated frame.

Binning is by real elapsed TIME (each sample's own [t_start, t_end] span cut
into T equal-duration bins), not by event COUNT. This is a deliberate choice
-- see .agent/ebv-eye.md's SNN design notes: fixed-event-count bins (e.g. exactly 50
events/bin) throw away the event-rate variation within the window (whether a
bin's events arrived in 2ms or 20ms), which is itself informative dynamics
an SNN's temporal structure can exploit; fixed-time bins preserve it.
"""

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

_ANN_DIR = Path(__file__).resolve().parent.parent / "ann"
_ann_module_cache: dict[str, object] = {}


def _load_ann_module(name: str):
    """Load code/ebv-eye/ann/{name}.py, returning a module object distinct
    from any same-named module in this directory (both trees have a
    dataloader.py/model.py/training.py).

    ann/training.py itself does plain `from dataloader import ...` / `from
    model import ...` -- if this directory's own dataloader.py/model.py
    were already imported under those plain names (which snn/training.py's
    own top-level `from dataloader import ...` does), that plain import
    would silently resolve to the wrong (snn-side) module instead of
    ann/dataloader.py. To avoid that, sys.modules[name] is swapped to this
    ann module (and any already-loaded ann dependencies are made visible
    under their own plain names) only for the duration of exec_module, then
    restored -- so this directory's own same-named modules are never
    permanently shadowed. Caller must load dependencies before dependents
    (e.g. "dataloader" before "training")."""
    if name in _ann_module_cache:
        return _ann_module_cache[name]

    spec = importlib.util.spec_from_file_location(name, _ANN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)

    saved_self = sys.modules.get(name)
    sys.modules[name] = module
    saved_deps = {}
    for dep_name, dep_module in _ann_module_cache.items():
        saved_deps[dep_name] = sys.modules.get(dep_name)
        sys.modules[dep_name] = dep_module
    try:
        spec.loader.exec_module(module)
    finally:
        if saved_self is not None:
            sys.modules[name] = saved_self
        else:
            sys.modules.pop(name, None)
        for dep_name, prev in saved_deps.items():
            if prev is not None:
                sys.modules[dep_name] = prev
            else:
                sys.modules.pop(dep_name, None)

    _ann_module_cache[name] = module
    return module


ann_dataloader = _load_ann_module("dataloader")


def events_to_time_binned_frames(events: np.ndarray, t_start: int, t_end: int, roi_size: int, num_steps: int) -> np.ndarray:
    """(T, 2, roi_size, roi_size) polarity-separated event-count frames.
    Bin index for event i = floor((t_i - t_start) / (t_end - t_start) * T),
    clipped to [0, T-1]. Per-bin event counts are NOT normalized/balanced --
    an empty or very sparse bin is a real, informative outcome (e.g. a
    settled/low-activity stretch), not something to correct for."""
    frames = np.zeros((num_steps, 2, roi_size, roi_size), dtype=np.float32)
    span = max(int(t_end) - int(t_start), 1)
    bin_idx = np.clip(((events["t"].astype(np.int64) - int(t_start)) * num_steps) // span, 0, num_steps - 1)
    pol = events["p"].astype(np.int64)
    row = events["row"].astype(np.int64)
    col = events["col"].astype(np.int64)
    np.add.at(frames, (bin_idx, pol, row, col), 1.0)
    return frames


class EbvEyeSpikeDataset(Dataset):
    """T-timestep counterpart to ann.dataloader.EbvEyeFrameDataset. Reads the
    identical cache, so results are directly comparable to the ANN's 128x128
    corrective_peak numbers -- only the input representation (T real-time
    bins vs one accumulated frame) and the model differ.

    __getitem__ returns (frames, label): frames is (T, 2, roi_size, roi_size)
    float32; label is normalized [row_n, col_n] float32 for
    label_mode='regression' (same convention as ann.dataloader, see
    normalize_label), or [row_class, col_class] int64 for 'classification'
    (see ann.dataloader.row_col_to_class)."""

    def __init__(self, npz_path: Path, num_steps: int = 10, roi_size: int | None = None, label_mode: str = "regression"):
        if label_mode not in ("regression", "classification"):
            raise ValueError(f"label_mode must be 'regression' or 'classification', got {label_mode!r}")
        npz_path = Path(npz_path)
        self.npz_path = npz_path.resolve()
        data = np.load(npz_path)
        self.events = data["events"]
        self.labels = data["labels"].astype(np.float64)
        self.segment_index = (
            data["segment_index"].astype(np.int64)
            if "segment_index" in data.files
            else None
        )
        self.t_start = data["t_start"]
        self.t_end = data["t_end"]
        self.num_steps = num_steps
        self.label_mode = label_mode

        if roi_size is None:
            meta_path = npz_path.parent / "meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(f"roi_size not given and no meta.json next to {npz_path}")
            roi_size = json.loads(meta_path.read_text())["roi_size"]
        self.roi_size = roi_size

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        frames = events_to_time_binned_frames(
            self.events[idx], self.t_start[idx], self.t_end[idx], self.roi_size, self.num_steps
        )
        row, col = self.labels[idx]
        if self.label_mode == "classification":
            row_c, col_c = ann_dataloader.row_col_to_class(row, col)
            label = torch.tensor([row_c, col_c], dtype=torch.int64)
        else:
            row_n, col_n = ann_dataloader.normalize_label(row, col)
            label = torch.tensor([row_n, col_n], dtype=torch.float32)
        return torch.from_numpy(frames), label


def build_dataset(
    dataset_root: Path,
    user: int,
    split: str,
    num_steps: int = 10,
    label_mode: str = "regression",
    valid_start_point: str = "corrective_peak",
    downsampled_to: int | None = 128,
    cache_dir: Path | None = None,
) -> Dataset:
    """Left eye (eye0) only -- matches this project's SNN scope so far (see
    .agent/ebv-eye.md). Directory-naming convention matches ann.dataloader.build_dataset
    exactly (valid_start_point + optional downsampled_to), so it resolves to
    the same on-disk cache. ``cache_dir`` overrides that legacy resolution
    and is used by named grouped-split caches containing an independent
    validation.npz."""
    if cache_dir is not None:
        split_dir = Path(cache_dir)
    else:
        eye_dir = Path(dataset_root) / "eye_data" / f"user{user}" / "0"
        if downsampled_to is not None:
            src_suffix = f"{valid_start_point}_start"
            split_dir = eye_dir / f"{src_suffix}_downsampled{downsampled_to}"
        else:
            split_dir = eye_dir / f"{valid_start_point}_start"
    return EbvEyeSpikeDataset(split_dir / f"{split}.npz", num_steps=num_steps, label_mode=label_mode)


def build_pooled_dataset(
    dataset_root: Path,
    users: list[int],
    num_steps: int = 10,
    label_mode: str = "regression",
    valid_start_point: str = "corrective_peak",
    downsampled_to: int | None = 128,
) -> Dataset:
    """SNN counterpart to ann.dataloader.build_pooled_dataset -- concatenates
    multiple users' `dataset_preprocessing.py fixed --no-split` pooled
    caches (eye_data/user{N}/0/{valid_start_point}_start_pooled[_downsampled{M}]/
    all.npz) into one Dataset via ConcatDataset, reusing EbvEyeSpikeDataset
    completely unchanged. See ann.dataloader.build_pooled_dataset's
    docstring for the full rationale (train/val/test split decided at the
    user level, not the window level). Eye0 only, same as build_dataset
    above -- only eye0 has been --no-split preprocessed so far."""
    if not users:
        raise ValueError("users must be a non-empty list")
    parts = []
    for user in users:
        eye_dir = Path(dataset_root) / "eye_data" / f"user{user}" / "0"
        if downsampled_to is not None:
            split_dir = eye_dir / f"{valid_start_point}_start_pooled_downsampled{downsampled_to}"
        else:
            split_dir = eye_dir / f"{valid_start_point}_start_pooled"
        parts.append(EbvEyeSpikeDataset(split_dir / "all.npz", num_steps=num_steps, label_mode=label_mode))
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
