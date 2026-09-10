#!/usr/bin/env python3
"""ebv-eye preprocessing: turn one user/eye's raw events.aerdat + frame
labels into fixed-length event-sample caches (train/test, optional validation,
and meta.json under eye_data/user{N}/{eye}/...) for the ANN/SNN pipelines.

Two subcommands, because "raw events -> trainable cache" has two facets that
used to live in separate files with ~90% duplicated CLI surface -- merged
2026-08-22, see .agent/ebv-eye.md.
Always name a subcommand; there is no bare/default action.

  fixed       This project's own design (not from Tan et al.): a fixed
              (default centered) square crop, per-target-segment sampling
              using only the part of each segment judged reliable (see
              `run_fixed`'s docstring below for the valid-start-point
              choices and sampling cycle). Was dataset_preprocessing.py's only
              behavior before this merge.
  downsample  Post-process an existing `fixed`-subcommand cache: floor-
              divide cached ROI-local row/col by --factor (a sum-pool
              equivalent -- see `run_downsample`'s docstring), without
              re-reading events.aerdat or re-running segment/phase
              detection. Was downsample_cache.py.

Usage:
    python dataset_preprocessing.py fixed --user 15 --eye 0 --stimulus s --dry-run
    python dataset_preprocessing.py fixed --user 15 --eye 0 --stimulus s
    python dataset_preprocessing.py downsample --user 15 --eye 0 --valid-start-point corrective_peak --factor 2
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from dataset_analysis import (
    DEFAULT_DATASET_ROOT,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    AERDAT_DTYPE,
    build_target_segments,
    centered_offset,
    compute_rate_curve,
    detect_phases,
    parse_frame_labels_full,
    read_events,
    roi_mask,
    sensor_valid_mask,
)


# ======================================================================
# Shared helpers
# ======================================================================


def plan_cycle_windows(num_valid_events: int, window_size: int, train_stride: int, test_stride: int, train_per_cycle: int) -> list[tuple[int, str]]:
    """Slot-by-slot: `train_per_cycle` training windows at `train_stride`
    apart, then one test window reached by `test_stride` from the last
    training window's start. If a test window would not fully fit, testing
    is permanently disabled for the rest of this segment and only
    `train_stride`-spaced training windows are produced until none fit."""
    plan: list[tuple[int, str]] = []
    pos = 0
    cycle_count = 0
    testing_enabled = True

    while pos + window_size <= num_valid_events:
        plan.append((pos, "train"))
        cycle_count += 1
        if testing_enabled and cycle_count == train_per_cycle:
            test_pos = pos + test_stride
            if test_pos + window_size <= num_valid_events:
                plan.append((test_pos, "test"))
                pos = test_pos + test_stride
            else:
                testing_enabled = False
                pos += train_stride
            cycle_count = 0
            continue
        pos += train_stride

    return plan


def plan_consecutive_windows(num_valid_events: int, window_size: int) -> list[tuple[int, str]]:
    """Non-overlapping consecutive windows, no train/test distinction --
    every window tagged "all". Used by --no-split, where the caller wants
    one pooled bucket of samples per (user, eye, stimulus) and intends to
    decide the train/val/test split at the *user* level instead of the
    window level (e.g. pool many users' samples for training, hold a few
    whole users out for validation/test)."""
    return [(pos, "all") for pos in range(0, num_valid_events - window_size + 1, window_size)]


def plan_overlapping_grouped_windows(
    num_valid_events: int, window_size: int, stride: int
) -> list[tuple[int, str]]:
    """All stride-spaced windows from one segment, pending a segment-level split.

    Unlike the legacy 4-train+1-test cycle, every window from this segment
    receives its split only after the *whole segment* has been assigned to
    train/validation/test. Overlap is therefore confined within one split.
    """
    return [
        (pos, "unassigned")
        for pos in range(0, num_valid_events - window_size + 1, stride)
    ]


def plan_tail_cycle_windows(
    num_valid_events: int,
    window_size: int,
    train_stride: int,
    train_per_cycle: int,
) -> list[tuple[int, str]]:
    """Plan pursuit windows backwards from the recording tail.

    Each backwards cycle reserves one non-overlapping test window, then one
    non-overlapping validation window, then up to ``train_per_cycle`` train
    windows. Train windows are the only windows allowed to overlap, using
    ``train_stride``. If the head remainder cannot hold a complete cycle,
    allocation priority remains test, validation, then train.

    The returned plan is sorted chronologically even though allocation is
    performed backwards. This makes cache order natural without changing the
    tail-first allocation rule.
    """
    if num_valid_events < 0:
        raise ValueError("num_valid_events must be non-negative")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 < train_stride <= window_size:
        raise ValueError("train_stride must lie in [1, window_size]")
    if train_per_cycle < 0:
        raise ValueError("train_per_cycle must be non-negative")

    plan: list[tuple[int, str]] = []
    cursor = num_valid_events

    while cursor >= window_size:
        test_start = cursor - window_size
        plan.append((test_start, "test"))
        cursor = test_start

        if cursor < window_size:
            break
        validation_start = cursor - window_size
        plan.append((validation_start, "validation"))
        cursor = validation_start

        earliest_train_start = cursor
        for train_index in range(train_per_cycle):
            if train_index == 0:
                train_start = cursor - window_size
            else:
                train_start = earliest_train_start - train_stride
            if train_start < 0:
                break
            plan.append((train_start, "train"))
            earliest_train_start = train_start
        cursor = earliest_train_start

    return sorted(plan, key=lambda item: item[0])


def _plan_windows(args: argparse.Namespace, num_valid_events: int) -> list[tuple[int, str]]:
    """Dispatches to plan_consecutive_windows (--no-split) or
    plan_cycle_windows (default) -- shared by process_segment and
    process_continuous_segment so the --no-split branch is only written
    once."""
    if args.no_split:
        return plan_consecutive_windows(num_valid_events, args.slice_length)
    train_stride = round(args.slice_length * (1 - args.overlap))
    if args.tail_cycle_split:
        return plan_tail_cycle_windows(
            num_valid_events,
            args.slice_length,
            train_stride,
            args.train_samples_per_cycle,
        )
    if args.segment_split:
        return plan_overlapping_grouped_windows(
            num_valid_events, args.slice_length, train_stride
        )
    return plan_cycle_windows(num_valid_events, args.slice_length, train_stride, args.test_stride, args.train_samples_per_cycle)


def _fraction_percent(value: float) -> int:
    percentage = round(100 * value)
    if not np.isclose(value, percentage / 100.0):
        raise ValueError(
            f"split fractions must be whole percentages for unambiguous cache names, got {value}"
        )
    return percentage


def segment_split_cache_name(args: argparse.Namespace) -> str:
    train_pct = _fraction_percent(args.train_fraction)
    validation_pct = _fraction_percent(args.validation_fraction)
    test_pct = 100 - train_pct - validation_pct
    return (
        f"{args.valid_start_point}_start_segment_split_"
        f"train{train_pct}_validation{validation_pct}_test{test_pct}_"
        f"seed{args.split_seed}"
    )


def tail_cycle_split_cache_name(args: argparse.Namespace) -> str:
    train_stride = round(args.slice_length * (1 - args.overlap))
    return (
        f"{args.valid_start_point}_start_tail_cycle_"
        f"train{args.train_samples_per_cycle}_validation1_test1_"
        f"slice{args.slice_length}_trainstride{train_stride}"
    )


def choose_segment_split(
    samples: list[dict],
    segment_reports: list[dict],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    search_trials: int,
) -> tuple[dict[int, str], dict]:
    """Choose a deterministic 70/15/15-style split over complete segments.

    Candidate assignments have exact group counts. They are ranked
    lexicographically by (1) missing row/col values, (2) sample-fraction
    imbalance, and (3) row/col marginal imbalance. Model outputs are never
    consulted. A random search is used only to find a well-balanced grouped
    assignment; the seed and final manifest are stored in meta.json.
    """
    test_fraction = 1.0 - train_fraction - validation_fraction
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--train-fraction must lie strictly between 0 and 1")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("--validation-fraction must lie strictly between 0 and 1")
    if test_fraction <= 0.0:
        raise ValueError("train + validation fractions must be less than 1")
    if search_trials < 1:
        raise ValueError("--split-search-trials must be positive")

    sample_counts = Counter(int(sample["segment_index"]) for sample in samples)
    records = [
        {
            "segment_index": int(report["segment_index"]),
            "row": float(report["target_row"]),
            "col": float(report["target_col"]),
            "sample_count": int(sample_counts[int(report["segment_index"])]),
        }
        for report in segment_reports
        if report["status"] == "ok"
        and sample_counts[int(report["segment_index"])] > 0
    ]
    if not records:
        raise ValueError("no sample-bearing segments are available for grouped split")

    group_names = ("train", "validation", "test")
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    n_segments = len(records)
    n_train = int(round(n_segments * train_fraction))
    n_validation = int(round(n_segments * validation_fraction))
    counts = {
        "train": n_train,
        "validation": n_validation,
        "test": n_segments - n_train - n_validation,
    }
    if min(counts.values()) <= 0:
        raise ValueError(f"too few segments for requested split: {counts}")

    all_rows = sorted({record["row"] for record in records})
    all_cols = sorted({record["col"] for record in records})
    total_samples = sum(record["sample_count"] for record in records)
    row_totals = Counter(record["row"] for record in records)
    col_totals = Counter(record["col"] for record in records)

    def assignment_from_order(order: np.ndarray) -> dict[str, list[dict]]:
        train_end = counts["train"]
        validation_end = train_end + counts["validation"]
        positions = {
            "train": order[:train_end],
            "validation": order[train_end:validation_end],
            "test": order[validation_end:],
        }
        return {
            name: [records[int(position)] for position in positions[name]]
            for name in group_names
        }

    def score(groups: dict[str, list[dict]]) -> tuple[int, float, float]:
        missing = 0
        sample_error = 0.0
        marginal_error = 0.0
        for name in group_names:
            group = groups[name]
            rows = Counter(record["row"] for record in group)
            cols = Counter(record["col"] for record in group)
            missing += sum(rows[row] == 0 for row in all_rows)
            missing += sum(cols[col] == 0 for col in all_cols)
            observed_fraction = (
                sum(record["sample_count"] for record in group) / total_samples
            )
            sample_error += abs(observed_fraction - fractions[name])
            marginal_error += sum(
                abs(rows[row] - row_totals[row] * fractions[name])
                for row in all_rows
            )
            marginal_error += sum(
                abs(cols[col] - col_totals[col] * fractions[name])
                for col in all_cols
            )
        return missing, sample_error, marginal_error

    generator = np.random.default_rng(seed)
    best_groups = None
    best_score = None
    for _ in range(search_trials):
        candidate = assignment_from_order(generator.permutation(n_segments))
        candidate_score = score(candidate)
        if best_score is None or candidate_score < best_score:
            best_groups = candidate
            best_score = candidate_score

    assert best_groups is not None and best_score is not None
    # Refine the best random candidate with count-preserving pair swaps.
    # This is especially helpful here because target segments contain very
    # different event/window counts.
    for _ in range(search_trials):
        left_name, right_name = generator.choice(group_names, size=2, replace=False)
        left_index = int(generator.integers(len(best_groups[left_name])))
        right_index = int(generator.integers(len(best_groups[right_name])))
        best_groups[left_name][left_index], best_groups[right_name][right_index] = (
            best_groups[right_name][right_index],
            best_groups[left_name][left_index],
        )
        candidate_score = score(best_groups)
        if candidate_score < best_score:
            best_score = candidate_score
        else:
            best_groups[left_name][left_index], best_groups[right_name][right_index] = (
                best_groups[right_name][right_index],
                best_groups[left_name][left_index],
            )

    if best_score[0] != 0:
        raise RuntimeError(
            "could not find a split covering every row and col value in every split; "
            f"best missing-count={best_score[0]}, increase --split-search-trials"
        )

    assignment = {
        record["segment_index"]: name
        for name in group_names
        for record in best_groups[name]
    }
    manifest = {
        "strategy": "complete_target_segment_grouped",
        "seed": int(seed),
        "search_trials": int(search_trials),
        "fractions_requested": fractions,
        "score": {
            "missing_row_col_values": int(best_score[0]),
            "sample_fraction_absolute_error": float(best_score[1]),
            "row_col_marginal_absolute_error": float(best_score[2]),
        },
        "splits": {
            name: {
                "segment_count": len(best_groups[name]),
                "sample_count": sum(
                    record["sample_count"] for record in best_groups[name]
                ),
                "sample_fraction": (
                    sum(record["sample_count"] for record in best_groups[name])
                    / total_samples
                ),
                "segment_indices": sorted(
                    record["segment_index"] for record in best_groups[name]
                ),
                "targets": sorted(
                    [
                        {
                            "segment_index": record["segment_index"],
                            "row": record["row"],
                            "col": record["col"],
                            "sample_count": record["sample_count"],
                        }
                        for record in best_groups[name]
                    ],
                    key=lambda item: item["segment_index"],
                ),
            }
            for name in group_names
        },
    }
    return assignment, manifest


def stack_samples(samples: list[dict], slice_length: int) -> dict:
    """把每个样本的 dict ("events"/"label"/"segment_index"/"t_start"/"t_end")
    堆成定长数组，可以直接交给 np.savez。"""
    n = len(samples)
    events = np.empty((n, slice_length), dtype=AERDAT_DTYPE)
    labels = np.empty((n, 2), dtype=np.float64)
    segment_index = np.empty(n, dtype=np.int64)
    t_start = np.empty(n, dtype=np.int64)
    t_end = np.empty(n, dtype=np.int64)
    for i, s in enumerate(samples):
        events[i] = s["events"]
        labels[i] = s["label"]
        segment_index[i] = s["segment_index"]
        t_start[i] = s["t_start"]
        t_end[i] = s["t_end"]
    return {"events": events, "labels": labels, "segment_index": segment_index, "t_start": t_start, "t_end": t_end}


# ======================================================================
# `fixed`: this project's own fixed-crop, per-segment design
# ======================================================================


def process_segment(
    seg: dict,
    seg_events: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    """Returns (segment report dict, list of produced sample dicts)."""
    t_start, t_end, row, col = seg["t_start"], seg["t_end"], seg["row"], seg["col"]

    if len(seg_events) == 0:
        return {"status": "no_events", **seg}, []

    bin_centers_ms, counts = compute_rate_curve(seg_events["t"], t_start, t_end, args.bin_us)
    detection = detect_phases(
        counts, args.bin_us, args.baseline_tail_frac, args.min_run_bins, args.corrective_search_ms, args.stable_sustain_ms
    )

    if args.valid_start_point == "last_500ms":
        # Fixed per-segment cutoff, independent of detect_phases -- see
        # run_fixed's docstring. detect_phases is still run above so its
        # output remains available in the report for cross-referencing,
        # same as for the other two modes.
        duration_ms = (t_end - t_start) / 1000.0
        valid_start_ms = max(0.0, duration_ms - 500.0)
    else:
        start_key = f"{args.valid_start_point}_ms"
        if detection[start_key] is None:
            return {"status": f"no_{args.valid_start_point}_detected", "detection": detection, **seg}, []
        valid_start_ms = detection[start_key]

    valid_start_us = t_start + int(round(valid_start_ms * 1000))
    valid_start_idx = int(np.searchsorted(seg_events["t"], valid_start_us, side="left"))
    valid_events = seg_events[valid_start_idx:]

    plan = _plan_windows(args, len(valid_events))

    samples = []
    for pos, kind in plan:
        window = valid_events[pos : pos + args.slice_length]
        samples.append(
            {
                "kind": kind,
                "events": window,
                "label": (row, col),
                "t_start": int(window["t"][0]),
                "t_end": int(window["t"][-1]),
            }
        )

    report = {
        "status": "ok",
        "segment_index": seg["segment_index"],
        "target_row": row,
        "target_col": col,
        "duration_ms": (t_end - t_start) / 1000.0,
        "main_peak_end_ms": detection["main_peak_end_ms"],
        "corrective_peak_ms": detection["corrective_peak_ms"],
        "valid_start_ms": valid_start_ms,
        "valid_event_count": int(len(valid_events)),
        "train_samples": sum(1 for s in samples if s["kind"] == "train"),
        "validation_samples": 0,
        "test_samples": sum(1 for s in samples if s["kind"] == "test"),
        "pooled_samples": sum(1 for s in samples if s["kind"] == "all"),
        "grouped_samples": sum(1 for s in samples if s["kind"] == "unassigned"),
    }
    return report, samples


def process_continuous_segment(
    seg_events: np.ndarray,
    label_ts: np.ndarray,
    label_row: np.ndarray,
    label_col: np.ndarray,
    warmup_us: int,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    """pursuit_continuous counterpart to process_segment: one trajectory
    instead of one fixed-target segment. Each window's label is interpolated
    at its own time center rather than copied from a segment-constant
    (row, col). Window planning is selected by _plan_windows; in particular,
    --tail-cycle-split implements the pursuit 4-train/1-validation/1-test
    policy allocated backwards from the recording tail."""
    if len(seg_events) == 0:
        return {"status": "no_events"}, []

    valid_start_us = int(seg_events["t"][0]) + warmup_us
    valid_start_idx = int(np.searchsorted(seg_events["t"], valid_start_us, side="left"))
    valid_events = seg_events[valid_start_idx:]

    plan = _plan_windows(args, len(valid_events))

    samples = []
    for pos, kind in plan:
        window = valid_events[pos : pos + args.slice_length]
        t_center = (int(window["t"][0]) + int(window["t"][-1])) / 2.0
        row = float(np.interp(t_center, label_ts, label_row))
        col = float(np.interp(t_center, label_ts, label_col))
        samples.append(
            {
                "kind": kind,
                "events": window,
                "label": (row, col),
                "t_start": int(window["t"][0]),
                "t_end": int(window["t"][-1]),
            }
        )

    report = {
        "status": "ok",
        "segment_index": 0,
        "warmup_us": warmup_us,
        "valid_event_count": int(len(valid_events)),
        "train_samples": sum(1 for s in samples if s["kind"] == "train"),
        "validation_samples": sum(1 for s in samples if s["kind"] == "validation"),
        "test_samples": sum(1 for s in samples if s["kind"] == "test"),
        "pooled_samples": sum(1 for s in samples if s["kind"] == "all"),
        "grouped_samples": sum(1 for s in samples if s["kind"] == "unassigned"),
    }
    return report, samples


def run_fixed(args: argparse.Namespace) -> int:
    """This project's own design:

    - Segments: split the stimulus-filtered stream into constant-(row,col)
      target segments (see `build_target_segments` in dataset_analysis.py).
      Each segment corresponds to one fixation target.
    - Valid region per segment: events are only used from a start point
      onward, selected via --valid-start-point. Data before that point is
      dropped as unreliable: the eye has not plausibly settled near the
      target yet. Every sample drawn from a segment is labeled with that
      segment's fixed target [row, col] -- no interpolation needed, since we
      only draw from the part of the segment judged to already be on-target.
      Four choices:
      - `main_peak_end` (default): the valley right after the main saccade
        peak, detected per-segment from the event-rate curve (see
        detect_phases in dataset_analysis.py).
      - `corrective_peak`: the peak of the secondary/corrective saccade,
        ~20ms later at the median for user15 eye0 -- see .agent/ebv-eye.md
        "saccade 段内部的时间结构". Also event-rate-curve-detected.
      - `last_500ms`: a fixed per-segment cutoff at (segment_duration -
        500ms) -- keep only the last 0.5s of each segment, whatever its
        actual duration. Not detected from the event-rate curve at all.
        Inspired by Angelopoulos et al.'s "only the last half second (out of
        a 1.5s stimulus presentation) is used for each saccade" (Fig. 8
        caveat, not a general preprocessing rule in their paper -- see
        .agent/ebv-eye.md), but adapted to be relative to each segment's own
        real duration rather than a fixed absolute 1000ms cut; this is not a
        literal replication of their method (no reference implementation was
        available to check the exact semantics against).
      - `pursuit_continuous`: not a per-segment start point at all -- for
        smooth-pursuit stimulus types (e.g. --stimulus p) the target moves
        continuously, so build_target_segments' "constant-(row,col) run"
        grouping does not apply. Instead the entire contiguous span of
        --stimulus-typed frame labels is treated as one trajectory: after
        skipping --pursuit-warmup-ms at the very start (letting pursuit
        lock onto the moving target), every sampling window's label is
        linearly interpolated (np.interp) from the two nearest frame labels
        at that window's time center, rather than reading one constant
        (row, col) for the whole segment. See process_continuous_segment.
    - Sampling pattern within each segment's valid region (project decision,
      not from Tan et al.): a repeating cycle of 4 training windows (500
      events, stride 200, i.e. 60% overlap -- same as Tan et al.'s event
      slice aggregation) followed by 1 test window reached by a 500-event
      stride from the 4th training window's start (so the test window never
      overlaps any training window, before or after it -- the cycle resumes
      exactly at the test window's own end). This repeats until a full next
      test window no longer fits; from that point on, only training windows
      (stride 200) are produced until even one more full window no longer
      fits.
    - ROI: fixed square crop (default: centered), applied before
      segmentation.

    Output: legacy mode writes train.npz + test.npz (each holding a fixed-shape
    (N, slice_length) structured event array in ROI-local coordinates, an
    (N, 2) label array, and per-sample segment/start/timestamp metadata)
    plus a meta.json with the resolved configuration, per-segment counts,
    and detection statistics. Default --out-dir is a descriptively named
    subdirectory under eye_data/user{N}/{eye}/ (override with --out-dir).

    --segment-split: assign complete saccade target segments to
    train/validation/test, then keep every overlapping window from one
    segment inside that segment's split. The default 70/15/15 assignment
    balances complete-segment counts, window counts, and row/col coverage,
    and is frozen in meta.json. This mode intentionally never applies the
    legacy 4-train+1-test cycle.

    --tail-cycle-split: pursuit-only 4/1/1 split. Starting at the end of the
    continuous recording and moving backwards, each cycle reserves one test
    window, one validation window, then up to four overlapping train windows.
    Test and validation windows never share events with another split; only
    train windows overlap (default train stride: 200 events). An incomplete
    head remainder is allocated with test, validation, train priority.

    --no-split: for building multi-user pooled datasets where the
    train/val/test split will be decided at the *user* level (e.g. pool
    many users' samples for training, hold a few whole users out), not the
    window level. Replaces the 4-train+1-test cycling above with plain
    non-overlapping consecutive --slice-length windows (--overlap/
    --test-stride/--train-samples-per-cycle are ignored) and writes a
    single all.npz (not train.npz/test.npz) to its own
    {valid_start_point}_start_pooled/ subdirectory, including for
    main_peak_end.
    """
    split_modes = sum((args.no_split, args.segment_split, args.tail_cycle_split))
    if split_modes > 1:
        raise SystemExit(
            "--no-split, --segment-split, and --tail-cycle-split are mutually exclusive"
        )
    if args.segment_split and args.valid_start_point == "pursuit_continuous":
        raise SystemExit(
            "--segment-split currently supports fixed-target saccade segments only; "
            "pursuit needs a separate complete-sweep/block splitter"
        )
    if args.tail_cycle_split and args.valid_start_point != "pursuit_continuous":
        raise SystemExit(
            "--tail-cycle-split is defined only for --valid-start-point pursuit_continuous"
        )

    roi_x0 = args.roi_x0
    roi_y0 = args.roi_y0
    if roi_x0 is None or roi_y0 is None:
        default_x0, default_y0 = centered_offset(args.roi_size)
        roi_x0 = default_x0 if roi_x0 is None else roi_x0
        roi_y0 = default_y0 if roi_y0 is None else roi_y0

    user_dir = args.dataset_root / "eye_data" / f"user{args.user}"
    eye_dir = user_dir / str(args.eye)
    aerdat_path = eye_dir / "events.aerdat"
    frames_dir = eye_dir / "frames"
    if not aerdat_path.is_file():
        raise SystemExit(f"events.aerdat not found: {aerdat_path}")

    out_dir = args.out_dir
    if out_dir is None:
        if args.no_split:
            out_dir = eye_dir / f"{args.valid_start_point}_start_pooled"
        elif args.segment_split:
            out_dir = eye_dir / segment_split_cache_name(args)
        elif args.tail_cycle_split:
            out_dir = eye_dir / tail_cycle_split_cache_name(args)
        else:
            out_dir = eye_dir / f"{args.valid_start_point}_start"

    events = read_events(aerdat_path)
    events = events[sensor_valid_mask(events)]
    events = events[roi_mask(events, roi_x0, roi_y0, args.roi_size)]
    # ROI-local coordinates for storage.
    events = events.copy()
    events["row"] = events["row"] - roi_y0
    events["col"] = events["col"] - roi_x0

    labels = parse_frame_labels_full(frames_dir)
    event_t = events["t"]
    all_samples: list[dict] = []  # train, all (--no-split), or pending grouped samples
    all_validation_samples: list[dict] = []
    all_test_samples: list[dict] = []
    segment_reports = []

    def _bucket(s: dict) -> None:
        if args.no_split or args.segment_split or s["kind"] == "train":
            all_samples.append(s)
        elif s["kind"] == "validation":
            all_validation_samples.append(s)
        elif s["kind"] == "test":
            all_test_samples.append(s)
        else:
            raise ValueError(f"unhandled sample kind: {s['kind']}")

    if args.valid_start_point == "pursuit_continuous":
        stim_labels = labels[labels["stimulus"] == args.stimulus]
        if len(stim_labels) < 2:
            raise SystemExit(f"not enough '{args.stimulus}' labeled frames for pursuit_continuous (need >= 2)")
        t0, t1 = int(stim_labels["t"][0]), int(stim_labels["t"][-1])
        seg_events = events[(event_t >= t0) & (event_t < t1)]
        warmup_us = int(round(args.pursuit_warmup_ms * 1000))
        report, samples = process_continuous_segment(
            seg_events, stim_labels["t"], stim_labels["row"], stim_labels["col"], warmup_us, args
        )
        segment_reports.append(report)
        for s in samples:
            s["segment_index"] = 0
            _bucket(s)
        num_segments_total = 1
    else:
        segments = build_target_segments(labels, args.stimulus)
        if not segments:
            raise SystemExit(f"no '{args.stimulus}' segments found")

        for idx, seg in enumerate(segments):
            seg = {**seg, "segment_index": idx}
            seg_mask = (event_t >= seg["t_start"]) & (event_t < seg["t_end"])
            seg_events = events[seg_mask]

            report, samples = process_segment(seg, seg_events, args)
            segment_reports.append(report)
            for s in samples:
                s["segment_index"] = idx
                _bucket(s)
        num_segments_total = len(segments)

    split_manifest = None
    if args.tail_cycle_split:
        split_manifest = {
            "strategy": "pursuit_tail_cycle",
            "allocation_direction": "recording_tail_to_head",
            "allocation_priority": ["test", "validation", "train"],
            "cycle": {
                "train_windows_max": args.train_samples_per_cycle,
                "validation_windows": 1,
                "test_windows": 1,
            },
            "cross_split_event_overlap": False,
            "train_only_overlap": args.overlap,
        }
    if args.segment_split:
        assignment, split_manifest = choose_segment_split(
            all_samples,
            segment_reports,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed,
            search_trials=args.split_search_trials,
        )
        grouped_samples = {"train": [], "validation": [], "test": []}
        for sample in all_samples:
            split_name = assignment[int(sample["segment_index"])]
            sample["kind"] = split_name
            grouped_samples[split_name].append(sample)
        all_samples = grouped_samples["train"]
        all_validation_samples = grouped_samples["validation"]
        all_test_samples = grouped_samples["test"]
        for report in segment_reports:
            if report["status"] != "ok":
                continue
            split_name = assignment.get(int(report["segment_index"]))
            report["split"] = split_name
            report["train_samples"] = (
                report["grouped_samples"] if split_name == "train" else 0
            )
            report["validation_samples"] = (
                report["grouped_samples"] if split_name == "validation" else 0
            )
            report["test_samples"] = (
                report["grouped_samples"] if split_name == "test" else 0
            )

    stacked = stack_samples(all_samples, args.slice_length)
    validation = (
        stack_samples(all_validation_samples, args.slice_length)
        if args.segment_split or args.tail_cycle_split
        else None
    )
    test = None if args.no_split else stack_samples(all_test_samples, args.slice_length)

    n_ok = sum(1 for r in segment_reports if r["status"] == "ok")
    n_skipped = len(segment_reports) - n_ok

    meta = {
        "user": args.user,
        "eye": args.eye,
        "stimulus": args.stimulus,
        "roi_x0": roi_x0,
        "roi_y0": roi_y0,
        "roi_size": args.roi_size,
        "slice_length": args.slice_length,
        "no_split": args.no_split,
        "segment_split": args.segment_split,
        "tail_cycle_split": args.tail_cycle_split,
        "split_names": (
            ["all"]
            if args.no_split
            else (
                ["train", "validation", "test"]
                if args.segment_split or args.tail_cycle_split
                else ["train", "test"]
            )
        ),
        "split_manifest": split_manifest,
        "overlap": None if args.no_split else args.overlap,
        "train_stride": None if args.no_split else round(args.slice_length * (1 - args.overlap)),
        "test_stride": (
            None
            if args.no_split or args.segment_split or args.tail_cycle_split
            else args.test_stride
        ),
        "train_samples_per_cycle": (
            None if args.no_split or args.segment_split else args.train_samples_per_cycle
        ),
        "valid_start_point": args.valid_start_point,
        "valid_region_definition": (
            "(segment_duration - 500ms) onward, fixed per segment, not detected"
            if args.valid_start_point == "last_500ms"
            else (
                f"whole contiguous '{args.stimulus}' trajectory, {args.pursuit_warmup_ms}ms warmup skipped, "
                "per-window label linearly interpolated at window time-center"
                if args.valid_start_point == "pursuit_continuous"
                else f"{args.valid_start_point}_ms onward (see detect_phases in dataset_analysis.py)"
            )
        ),
        "pursuit_warmup_ms": args.pursuit_warmup_ms if args.valid_start_point == "pursuit_continuous" else None,
        "phase_detection_params": {
            "bin_us": args.bin_us,
            "baseline_tail_frac": args.baseline_tail_frac,
            "min_run_bins": args.min_run_bins,
            "corrective_search_ms": args.corrective_search_ms,
            "stable_sustain_ms": args.stable_sustain_ms,
        },
        "sensor_height": SENSOR_HEIGHT,
        "sensor_width": SENSOR_WIDTH,
        "num_segments_total": num_segments_total,
        "num_segments_ok": n_ok,
        "num_segments_skipped": n_skipped,
        "train_samples": 0 if args.no_split else int(len(all_samples)),
        "validation_samples": int(len(all_validation_samples)),
        "test_samples": 0 if args.no_split else int(len(all_test_samples)),
        "pooled_samples": int(len(all_samples)) if args.no_split else 0,
        "generated_at": time.time(),
        "segments": segment_reports,
    }

    print(
        json.dumps(
            {k: v for k, v in meta.items() if k != "segments"},
            indent=2,
        )
    )
    if n_skipped:
        print(f"\n{n_skipped} segment(s) skipped (no events or no {args.valid_start_point} detected):")
        for r in segment_reports:
            if r["status"] != "ok":
                print(f"  segment {r.get('segment_index')}: {r['status']}")

    if args.dry_run:
        print("\n--dry-run: writing nothing")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_split:
        splits = (("all", stacked),)
    elif args.segment_split or args.tail_cycle_split:
        splits = (
            ("train", stacked),
            ("validation", validation),
            ("test", test),
        )
    else:
        splits = (("train", stacked), ("test", test))
    for split_name, split in splits:
        np.savez(
            out_dir / f"{split_name}.npz",
            events=split["events"],
            labels=split["labels"],
            segment_index=split["segment_index"],
            t_start=split["t_start"],
            t_end=split["t_end"],
        )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    written = ", ".join(f"{split_name}.npz" for split_name, _ in splits)
    print(f"\nWrote {out_dir}/{written}, meta.json")
    return 0


# ======================================================================
# `downsample`: post-process an existing `fixed` cache to a lower
# resolution, without re-reading raw events
# ======================================================================


def run_downsample(args: argparse.Namespace) -> int:
    """Pre-bakes a spatially-downsampled copy of an existing `fixed`-
    subcommand cache, without re-reading events.aerdat or re-running
    segment/phase detection.

    Floor-divides each cached event's already-ROI-local row/col by
    --factor (e.g. factor=2 turns a 256x256 ROI cache into a
    128x128-equivalent one). This is a sum-pooling equivalent: when
    dataloader.py's events_to_frame() rasterizes a sample, multiple
    original events that land on the same downsampled pixel are simply
    counted together via np.add.at -- identical to rasterizing at full
    resolution and then 2x2-sum-pooling the frame, but done once here
    instead of on every __getitem__ call.

    Output goes to
    eye_data/user{N}/{eye}/{valid_start_point}_start_downsampled{new_roi_size}/
    (e.g. corrective_peak_start_downsampled128/ for factor=2 on a 256x256
    ROI) -- named after the resulting resolution rather than the factor, so
    it's unambiguous without cross-referencing the source cache's native
    roi_size.

    只支持 `fixed` 子命令产生的缓存 (方形 ROI, meta.json 里是单个 roi_size 标量)。
    """
    eye_dir = args.dataset_root / "eye_data" / f"user{args.user}" / str(args.eye)
    if args.source_dir is not None:
        src_dir = args.source_dir
    elif args.pooled:
        src_dir = eye_dir / f"{args.valid_start_point}_start_pooled"
    else:
        src_dir = eye_dir / f"{args.valid_start_point}_start"
    if not (src_dir / "meta.json").is_file():
        raise SystemExit(f"source cache not found: {src_dir} (run `dataset_preprocessing.py fixed` first)")

    meta = json.loads((src_dir / "meta.json").read_text())
    roi_size = meta["roi_size"]
    if roi_size % args.factor != 0:
        raise SystemExit(f"roi_size ({roi_size}) not divisible by --factor ({args.factor})")
    new_roi_size = roi_size // args.factor

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = src_dir.parent / f"{src_dir.name}_downsampled{new_roi_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = tuple(
        meta.get(
            "split_names",
            ["all"] if args.pooled else ["train", "test"],
        )
    )
    for split in splits:
        data = np.load(src_dir / f"{split}.npz")
        events = data["events"].copy()
        events["row"] = events["row"] // args.factor
        events["col"] = events["col"] // args.factor
        np.savez(
            out_dir / f"{split}.npz",
            events=events,
            labels=data["labels"],
            segment_index=data["segment_index"],
            t_start=data["t_start"],
            t_end=data["t_end"],
        )
        print(f"{split}: {len(events)} samples, row max {events['row'].max()}, col max {events['col'].max()}")

    new_meta = {
        **meta,
        "roi_size": new_roi_size,
        "roi_size_native": roi_size,
        "downsample_factor": args.factor,
        "downsampled_from": str(src_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(new_meta, indent=2))
    written = ", ".join(f"{split}.npz" for split in splits)
    print(f"\nWrote {out_dir}/{written}, meta.json")
    return 0


# ======================================================================
# CLI
# ======================================================================


def _add_dataset_args(parser: argparse.ArgumentParser, with_stimulus: bool = True) -> None:
    parser.add_argument("--user", type=int, required=True)
    parser.add_argument("--eye", type=int, required=True, choices=(0, 1))
    if with_stimulus:
        parser.add_argument("--stimulus", default="s", help="Stimulus type defining target segments (default: s)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)


def _add_slicing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slice-length", type=int, default=500, help="Events per sample (Tan et al. SL)")
    parser.add_argument("--overlap", type=float, default=0.6, help="Train stride = slice_length * (1-overlap)")
    parser.add_argument("--train-samples-per-cycle", type=int, default=4)
    parser.add_argument("--test-stride", type=int, default=500, help="Event-count stride used to reach each test window")


def _add_phase_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bin-us", type=int, default=5000, help="Event-rate curve bin width for phase detection")
    parser.add_argument("--baseline-tail-frac", type=float, default=0.4)
    parser.add_argument("--min-run-bins", type=int, default=2)
    parser.add_argument("--corrective-search-ms", type=float, default=400.0)
    parser.add_argument("--stable-sustain-ms", type=float, default=100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fixed = sub.add_parser("fixed", help="Fixed-crop, per-target-segment preprocessing (this project's own design)")
    _add_dataset_args(p_fixed)
    p_fixed.add_argument("--roi-size", type=int, default=256)
    p_fixed.add_argument("--roi-x0", type=int, default=None, help="Default: centered for --roi-size")
    p_fixed.add_argument("--roi-y0", type=int, default=None, help="Default: centered for --roi-size")
    p_fixed.add_argument(
        "--valid-start-point",
        choices=("main_peak_end", "corrective_peak", "last_500ms", "pursuit_continuous"),
        default="main_peak_end",
        help="Which point begins the valid (sampleable) region of each segment. main_peak_end/corrective_peak "
        "are event-rate-curve-detected phases; last_500ms is a fixed (segment_duration - 500ms) cutoff, not "
        "detected at all; pursuit_continuous abandons the per-segment model entirely and interpolates a "
        "continuous label per window -- see run_fixed's docstring.",
    )
    _add_slicing_args(p_fixed)
    p_fixed.add_argument(
        "--pursuit-warmup-ms",
        type=float,
        default=300.0,
        help="Only used with --valid-start-point pursuit_continuous: skip this much time at the very start of "
        "the continuous stimulus run before the first sampleable window.",
    )
    _add_phase_detection_args(p_fixed)
    p_fixed.add_argument("--out-dir", type=Path, default=None, help="Default: <dataset-root>/processed/<auto name>")
    p_fixed.add_argument("--dry-run", action="store_true", help="Report planned sample counts, write nothing")
    p_fixed.add_argument(
        "--no-split",
        action="store_true",
        help="Pool ALL windows into one all.npz instead of train.npz/test.npz -- non-overlapping consecutive "
        "--slice-length windows (--overlap/--test-stride/--train-samples-per-cycle are ignored), for building "
        "multi-user pooled datasets where train/val/test is decided at the user level. Output goes to its own "
        "{valid_start_point}_start_pooled/ subdirectory -- see run_fixed's docstring.",
    )
    p_fixed.add_argument(
        "--segment-split",
        action="store_true",
        help="Assign complete fixed-target segments to train/validation/test instead of using the legacy "
        "4-train+1-test window cycle. Windows may overlap within a split but never across splits.",
    )
    p_fixed.add_argument(
        "--tail-cycle-split",
        action="store_true",
        help="For pursuit_continuous only: allocate repeating 4-train/1-validation/1-test cycles backwards "
        "from the recording tail. In an incomplete head remainder, preserve test first, then validation, "
        "then as many train windows as fit. Only train windows may overlap.",
    )
    p_fixed.add_argument("--train-fraction", type=float, default=0.70)
    p_fixed.add_argument("--validation-fraction", type=float, default=0.15)
    p_fixed.add_argument("--split-seed", type=int, default=2021)
    p_fixed.add_argument(
        "--split-search-trials",
        type=int,
        default=20000,
        help="Deterministic candidate assignments searched when balancing complete segments",
    )

    p_down = sub.add_parser("downsample", help="Sum-pool an existing `fixed` cache to a lower resolution")
    _add_dataset_args(p_down, with_stimulus=False)
    p_down.add_argument(
        "--valid-start-point",
        choices=("main_peak_end", "corrective_peak", "pursuit_continuous"),
        default="main_peak_end",
        help="Which existing `fixed`-subcommand cache to read from (must already exist)",
    )
    p_down.add_argument("--factor", type=int, required=True, help="Downsample factor, e.g. 2 turns 256x256 into 128x128")
    p_down.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Explicit cache directory to downsample; useful for named segment-split caches",
    )
    p_down.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: append _downsampled<new size> to the source cache directory name",
    )
    p_down.add_argument(
        "--pooled",
        action="store_true",
        help="Downsample a `fixed --no-split` pooled cache ({valid_start_point}_start_pooled/all.npz) instead of "
        "a normal train/test-split cache. Must be passed explicitly -- not auto-detected, since a pooled and a "
        "non-pooled cache can both exist for the same --valid-start-point at once.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "fixed":
        return run_fixed(args)
    if args.command == "downsample":
        return run_downsample(args)
    raise AssertionError(f"unhandled subcommand: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
