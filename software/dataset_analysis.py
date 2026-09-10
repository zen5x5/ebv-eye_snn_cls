#!/usr/bin/env python3
"""ebv-eye data analysis tools (ROI offset, saccade timing, preprocessing
coverage) -- the project's general-purpose analysis toolkit, distinct from
`dataset_preprocessing.py` (which writes real training caches). Nothing in this
file ever writes to a `train.npz`/`test.npz` cache location.

`coverage`: for one user, scan every possible top-left offset of a fixed-size
square crop over the raw 346x260 sensor and report what percentage of that
eye's events each offset retains. One offset is used globally for the whole
eye's event stream (no per-segment refitting). Reported once over all events,
and once restricted to events inside 's' (saccade) stimulus segments, for
eye0 and eye1 separately.

`sample-bins`: for one user/eye/stimulus and a chosen ROI offset+size, filter
the raw event stream down to that ROI, randomly sample N non-repeating start
positions, and save one scatter-plot PNG per 500-consecutive-event bin.

`segment-profile`: for a handful of representative saccade target segments
(constant-(row,col) runs of a given stimulus type), plot the event-rate-over-
time curve within each segment, and save representative 500-event scatter
visualizations at each *auto-detected* phase: pre-movement (reaction-time
latency, still on the old target), main saccade peak, corrective-saccade
peak (if any), and stable-onset. Detection runs on every segment (not just
the chosen ones) and a phase_statistics.json aggregates the timing across
all of them, so those numbers can inform the real preprocessing design later.
Pass --phase-fractions to fall back to the old fixed-fraction behavior
instead.

`valid-start-coverage`: for one user/eye/stimulus, compare two or more
`dataset_preprocessing.py --valid-start-point` choices by calling that script's
own `process_segment()` once per target segment for each choice (not a
re-implementation) and tabulating the resulting train/test sample counts
per target -- so a candidate start-point definition can be sanity-checked
for coverage *before* anyone commits to actually caching/training on it.
Writes a JSON report (summary + full per-target breakdown) but never calls
`dataset_preprocessing.py` outside of `--dry-run`-equivalent, read-only use --
no train/test cache is ever written by this subcommand.

`multiuser-timeline`: for a set of users/eyes, pick the same fixed N saccade
grid targets (row,col identity, not list position, so it's robust to any one
subject missing a target elsewhere in the sequence) and save 5 representative
full-sensor (346x260, uncropped) 500-event snapshots spanning each target
segment start-to-end, with the centered ROI box overlaid in red for
reference. Meant for eyeballing cross-user differences (event density,
sensor coverage) independent of the ROI crop used for training.

`pursuit-montage`: for one eye, align every user to the same screen positions
along the smooth-pursuit trajectory and place one full-sensor event window per
user in a compact contact sheet. It also writes a summary of event-window
duration, ROI retention, event-cloud position/spread and polarity balance.

`research-audit`: match pursuit stimulus positions and directions across
users/eyes, measure full-sensor event statistics and saccade jump/count
correlations, and run time-only negative controls on explicitly supplied
caches. Writes descriptive JSON/CSV and a cohort figure outside raw data.

Usage:
    python dataset_analysis.py coverage --user 15
    python dataset_analysis.py coverage --user 15 --crop-size 256 --top-k 3
    python dataset_analysis.py sample-bins --user 15 --eye 0 --stimulus s \\
        --roi-size 256 --num-bins 50 --out-dir /data/xuzx/data/ebv-eye/visualizations/user15_eye0_saccade_roi256_centered
    python dataset_analysis.py segment-profile --user 15 --eye 0 --stimulus s \\
        --num-segments 5 --out-dir /data/xuzx/data/ebv-eye/analysis/segment_profiles/user15_eye0_s
    python dataset_analysis.py valid-start-coverage --user 15 --eye 0 --stimulus s \\
        --valid-start-points main_peak_end corrective_peak last_500ms
    python dataset_analysis.py multiuser-timeline --users 4 5 6 --eyes 0 1
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

SENSOR_HEIGHT = 260
SENSOR_WIDTH = 346
# Matches reference/ebv-eye/event_based_gaze_tracking/visualize.py: read_aerdat
# unpacks 'BHHI' and, after its reverse/pop dance, yields records in original
# file order as (polarity, row, col, timestamp) -- see .agent/ebv-eye.md.
AERDAT_DTYPE = np.dtype([("p", "u1"), ("row", "<u2"), ("col", "<u2"), ("t", "<u4")])

DEFAULT_DATASET_ROOT = Path("/data/xuzx/data/ebv-eye")

# 11x11 saccade grid coordinates, shared by every subject (see .agent/ebv-eye.md).
ROW_VALUES = [150, 228, 306, 384, 462, 540, 618, 696, 774, 852, 930]
COL_VALUES = [150, 312, 474, 636, 798, 960, 1122, 1284, 1446, 1608, 1770]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage = subparsers.add_parser("coverage", help="Scan all ROI offsets and rank by event retention")
    coverage.add_argument("--user", type=int, required=True, help="Subject number, e.g. 15")
    coverage.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    coverage.add_argument("--crop-size", type=int, default=256)
    coverage.add_argument("--top-k", type=int, default=3)

    sample = subparsers.add_parser("sample-bins", help="Visualize randomly sampled 500-event bins from a fixed ROI")
    sample.add_argument("--user", type=int, required=True)
    sample.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    sample.add_argument("--eye", type=int, required=True, choices=(0, 1))
    sample.add_argument(
        "--stimulus",
        default="s",
        help="Stimulus type to restrict events to ('s', 'p', 'st'), or 'all' for no filtering (default: s)",
    )
    sample.add_argument("--roi-size", type=int, default=256)
    sample.add_argument(
        "--roi-x0",
        type=int,
        default=None,
        help="ROI top-left x (col). Default: centered for the given --roi-size",
    )
    sample.add_argument(
        "--roi-y0",
        type=int,
        default=None,
        help="ROI top-left y (row). Default: centered for the given --roi-size",
    )
    sample.add_argument("--num-bins", type=int, default=50)
    sample.add_argument("--bin-size", type=int, default=500, help="Consecutive real events per bin")
    sample.add_argument("--seed", type=int, default=2021)
    sample.add_argument("--out-dir", type=Path, required=True)

    profile = subparsers.add_parser(
        "segment-profile", help="Plot event-rate curves and phase snapshots for representative target segments"
    )
    profile.add_argument("--user", type=int, required=True)
    profile.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    profile.add_argument("--eye", type=int, required=True, choices=(0, 1))
    profile.add_argument("--stimulus", default="s", help="Stimulus type defining target segments (default: s)")
    profile.add_argument("--roi-size", type=int, default=256)
    profile.add_argument("--roi-x0", type=int, default=None, help="Default: centered for --roi-size")
    profile.add_argument("--roi-y0", type=int, default=None, help="Default: centered for --roi-size")
    profile.add_argument("--bin-us", type=int, default=5000, help="Event-rate curve bin width in microseconds")
    profile.add_argument("--num-segments", type=int, default=5, help="How many representative segments to profile")
    profile.add_argument(
        "--segment-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit segment indices to use instead of an evenly-spaced pick",
    )
    profile.add_argument(
        "--phase-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Manual override: snapshot at these segment-duration fractions instead of auto-detecting "
        "phases from each segment's own rate curve (reaction end / main peak / corrective peak / "
        "stable onset). Auto-detection is the default.",
    )
    profile.add_argument("--baseline-tail-frac", type=float, default=0.4, help="Fraction of the segment's tail used to estimate the baseline rate")
    profile.add_argument("--min-run-bins", type=int, default=2, help="Consecutive above-threshold bins required to mark burst onset")
    profile.add_argument("--corrective-search-ms", type=float, default=400.0, help="Window after the post-main-peak valley to search for a corrective-saccade peak")
    profile.add_argument("--stable-sustain-ms", type=float, default=100.0, help="Below-threshold duration required to confirm stable onset")
    profile.add_argument("--window-size", type=int, default=500, help="Events per phase snapshot")
    profile.add_argument("--out-dir", type=Path, default=None, help="Default: <dataset-root>/analysis/segment_profiles/<auto name>")

    vsc = subparsers.add_parser(
        "valid-start-coverage",
        help="Compare --valid-start-point choices' per-target train/test sample counts (no cache written)",
    )
    vsc.add_argument("--user", type=int, required=True)
    vsc.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    vsc.add_argument("--eye", type=int, required=True, choices=(0, 1))
    vsc.add_argument("--stimulus", default="s", help="Stimulus type defining target segments (default: s)")
    vsc.add_argument("--roi-size", type=int, default=256)
    vsc.add_argument("--roi-x0", type=int, default=None, help="Default: centered for --roi-size")
    vsc.add_argument("--roi-y0", type=int, default=None, help="Default: centered for --roi-size")
    vsc.add_argument(
        "--valid-start-points",
        nargs="+",
        default=["main_peak_end", "corrective_peak", "last_500ms"],
        choices=("main_peak_end", "corrective_peak", "last_500ms"),
        help="Which dataset_preprocessing.py --valid-start-point values to compare (default: all three)",
    )
    vsc.add_argument(
        "--sort-by",
        default=None,
        choices=("main_peak_end", "corrective_peak", "last_500ms"),
        help="Rank targets by this point's test-sample count (default: the first entry in --valid-start-points)",
    )
    # Mirrors dataset_preprocessing.py's own defaults so a comparison run here
    # matches what an actual (non-dry-run) dataset_preprocessing.py invocation
    # would produce.
    vsc.add_argument("--slice-length", type=int, default=500)
    vsc.add_argument("--overlap", type=float, default=0.6)
    vsc.add_argument("--train-samples-per-cycle", type=int, default=4)
    vsc.add_argument("--test-stride", type=int, default=500)
    vsc.add_argument("--bin-us", type=int, default=5000)
    vsc.add_argument("--baseline-tail-frac", type=float, default=0.4)
    vsc.add_argument("--min-run-bins", type=int, default=2)
    vsc.add_argument("--corrective-search-ms", type=float, default=400.0)
    vsc.add_argument("--stable-sustain-ms", type=float, default=100.0)
    vsc.add_argument("--out-dir", type=Path, default=None, help="Default: <dataset-root>/analysis/valid_start_coverage/<auto name>")

    mt = subparsers.add_parser(
        "multiuser-timeline",
        help="Same fixed saccade targets across users/eyes, 5 full-sensor snapshots each spanning the target segment, ROI box overlaid",
    )
    mt.add_argument("--users", type=int, nargs="+", default=list(range(4, 28)), help="Subject numbers (default: 4-27)")
    mt.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    mt.add_argument("--eyes", type=int, nargs="+", default=[0, 1], choices=(0, 1))
    mt.add_argument("--stimulus", default="s", help="Stimulus type defining target segments (default: s)")
    mt.add_argument("--num-targets", type=int, default=3, help="How many grid targets to pick (same ones for every user/eye)")
    mt.add_argument("--num-points", type=int, default=5, help="Representative snapshot points per target, evenly spaced start-to-end")
    mt.add_argument("--window-size", type=int, default=500, help="Events per snapshot")
    mt.add_argument("--roi-size", type=int, default=256, help="Centered ROI box drawn for reference; events are NOT cropped to it")
    mt.add_argument("--seed", type=int, default=2021, help="Selects which --num-targets grid points are used, shared across every user/eye")
    mt.add_argument("--out-dir", type=Path, default=None, help="Default: <dataset-root>/eye_data/analysis/<auto name>")

    pm = subparsers.add_parser(
        "pursuit-montage",
        help="Target-aligned smooth-pursuit event montages and cross-user geometry summary",
    )
    pm.add_argument("--users", type=int, nargs="+", default=list(range(4, 28)), help="Subject numbers (default: 4-27)")
    pm.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    pm.add_argument("--eye", type=int, default=0, choices=(0, 1), help="Analyze one eye at a time (default: 0)")
    pm.add_argument(
        "--screen-targets",
        nargs="+",
        default=["300,900", "540,1770", "750,960"],
        help="Pursuit screen row,col anchors (default: 300,900 540,1770 750,960)",
    )
    pm.add_argument("--window-size", type=int, default=500, help="Valid full-sensor events per user panel")
    pm.add_argument("--roi-size", type=int, default=256, help="Centered training ROI drawn for reference")
    pm.add_argument("--out-dir", type=Path, default=None, help="Default: <dataset-root>/eye_data/analysis/<auto name>")

    audit = subparsers.add_parser(
        "research-audit", help="Matched-stimulus user/eye statistics, saccade counts, and time-only cache controls"
    )
    audit.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    audit.add_argument("--users", type=int, nargs="+", default=list(range(4, 28)))
    audit.add_argument("--eyes", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    audit.add_argument("--reference-user", type=int, default=15)
    audit.add_argument("--anchors", type=int, default=61)
    audit.add_argument("--target-tolerance-px", type=float, default=8.0)
    audit.add_argument("--window-size", type=int, default=500)
    audit.add_argument("--fixed-window-ms", type=float, default=25.0)
    audit.add_argument("--cache-dirs", type=Path, nargs="*", default=[])
    audit.add_argument("--out-dir", type=Path, required=True)

    return parser.parse_args()


def read_events(aerdat_path: Path) -> np.ndarray:
    itemsize = AERDAT_DTYPE.itemsize
    filesize = aerdat_path.stat().st_size
    num_events = filesize // itemsize
    usable_bytes = num_events * itemsize
    with open(aerdat_path, "rb") as f:
        buf = f.read(usable_bytes)
    return np.frombuffer(buf, dtype=AERDAT_DTYPE, count=num_events)


def sensor_valid_mask(events: np.ndarray) -> np.ndarray:
    return (events["row"] < SENSOR_HEIGHT) & (events["col"] < SENSOR_WIDTH)


def parse_frame_labels(frames_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Returns (timestamps sorted ascending, stimulus type per timestamp)."""
    records = []
    for path in frames_dir.glob("*.png"):
        parts = path.stem.split("_")
        if len(parts) != 5:
            continue
        _, _, _, stimulus, timestamp = parts
        records.append((int(timestamp), stimulus))
    records.sort(key=lambda r: r[0])
    timestamps = np.array([r[0] for r in records], dtype=np.int64)
    types = [r[1] for r in records]
    return timestamps, types


def stimulus_mask(
    event_t: np.ndarray, label_ts: np.ndarray, label_types: list[str], stimulus: str
) -> np.ndarray:
    """True where an event's timestamp falls in a [label_i, label_i+1) interval
    whose stimulus type equals `stimulus`. Events before the first or after/at
    the last label are excluded (ambiguous stimulus)."""
    if len(label_ts) < 2:
        return np.zeros(event_t.shape, dtype=bool)
    idx = np.searchsorted(label_ts, event_t, side="right") - 1
    valid = (idx >= 0) & (idx < len(label_types) - 1)
    idx_clipped = np.clip(idx, 0, len(label_types) - 2)
    types_arr = np.array(label_types)
    matches = types_arr[idx_clipped] == stimulus
    return valid & matches


def saccade_mask(event_t: np.ndarray, label_ts: np.ndarray, label_types: list[str]) -> np.ndarray:
    return stimulus_mask(event_t, label_ts, label_types, "s")


LABEL_DTYPE = np.dtype([("t", "<i8"), ("row", "<f8"), ("col", "<f8"), ("stimulus", "U2")])


def parse_frame_labels_full(frames_dir: Path) -> np.ndarray:
    """Every frame's (timestamp, screen row, screen col, stimulus type),
    sorted by timestamp ascending. Filename format:
    Index_Row_Col_Stimulus_Timestamp.png."""
    records = []
    for path in frames_dir.glob("*.png"):
        parts = path.stem.split("_")
        if len(parts) != 5:
            continue
        _, row, col, stimulus, timestamp = parts
        records.append((int(timestamp), float(row), float(col), stimulus))
    records.sort(key=lambda r: r[0])
    return np.array(records, dtype=LABEL_DTYPE)


def build_target_segments(labels: np.ndarray, stimulus: str = "s") -> list[dict]:
    """Group consecutive frame labels of the given stimulus type into
    constant-(row,col) target segments. A segment covers
    [first frame's timestamp, first frame's timestamp of the next differing
    run). Not every one of the dataset's 121 grid targets is guaranteed to
    have a segment -- some recordings are missing one (see .agent/ebv-eye.md)."""
    segments = []
    i, n = 0, len(labels)
    while i < n:
        if labels["stimulus"][i] != stimulus:
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and labels["stimulus"][j + 1] == stimulus
            and labels["row"][j + 1] == labels["row"][i]
            and labels["col"][j + 1] == labels["col"][i]
        ):
            j += 1
        t_start = int(labels["t"][i])
        t_end = int(labels["t"][j + 1]) if j + 1 < n else int(labels["t"][j])
        segments.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "row": float(labels["row"][i]),
                "col": float(labels["col"][i]),
                "num_frames": j - i + 1,
            }
        )
        i = j + 1
    return segments


def roi_mask(events: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    return (
        (events["row"] >= y0)
        & (events["row"] < y0 + size)
        & (events["col"] >= x0)
        & (events["col"] < x0 + size)
    )


def centered_offset(size: int) -> tuple[int, int]:
    return (SENSOR_WIDTH - size) // 2, (SENSOR_HEIGHT - size) // 2


def build_integral_image(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, int]:
    hist = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.int64)
    np.add.at(hist, (rows, cols), 1)
    integral = np.zeros((SENSOR_HEIGHT + 1, SENSOR_WIDTH + 1), dtype=np.int64)
    integral[1:, 1:] = hist.cumsum(axis=0).cumsum(axis=1)
    return integral, int(hist.sum())


def window_sum(integral: np.ndarray, y0: int, x0: int, size: int) -> int:
    y1, x1 = y0 + size, x0 + size
    return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def scan_offsets(integral: np.ndarray, total: int, size: int) -> list[dict]:
    results = []
    for y0 in range(0, SENSOR_HEIGHT - size + 1):
        for x0 in range(0, SENSOR_WIDTH - size + 1):
            count = window_sum(integral, y0, x0, size)
            pct = (count / total * 100.0) if total else 0.0
            results.append({"x0": x0, "y0": y0, "retained_events": count, "retained_pct": pct})
    results.sort(key=lambda r: r["retained_pct"], reverse=True)
    return results


def summarize(rows: np.ndarray, cols: np.ndarray, size: int, top_k: int) -> dict:
    integral, total = build_integral_image(rows, cols)
    ranked = scan_offsets(integral, total, size)
    centered_x0, centered_y0 = centered_offset(size)
    centered_rank = next(
        i + 1 for i, r in enumerate(ranked) if r["x0"] == centered_x0 and r["y0"] == centered_y0
    )
    centered_entry = next(r for r in ranked if r["x0"] == centered_x0 and r["y0"] == centered_y0)
    return {
        "total_events": total,
        "num_candidate_offsets": len(ranked),
        "top_k": ranked[:top_k],
        "centered_offset": {**centered_entry, "rank": centered_rank},
    }


def print_summary(label: str, summary: dict, top_k: int) -> None:
    print(f"\n=== {label} ===")
    print(f"total events considered: {summary['total_events']:,}")
    print(f"candidate offsets scanned: {summary['num_candidate_offsets']}")
    print(f"top {top_k} offsets by retained event percentage:")
    for rank, entry in enumerate(summary["top_k"], start=1):
        print(
            f"  #{rank}: (x0={entry['x0']}, y0={entry['y0']}) "
            f"-> {entry['retained_events']:,} events ({entry['retained_pct']:.4f}%)"
        )
    c = summary["centered_offset"]
    print(
        f"centered offset (x0={c['x0']}, y0={c['y0']}), rank #{c['rank']} of "
        f"{summary['num_candidate_offsets']}: {c['retained_events']:,} events "
        f"({c['retained_pct']:.4f}%)"
    )


def run_coverage(args: argparse.Namespace) -> int:
    user_dir = args.dataset_root / "eye_data" / f"user{args.user}"
    if not user_dir.is_dir():
        raise SystemExit(f"user dataset not found: {user_dir}")

    report = {
        "user": args.user,
        "crop_size": args.crop_size,
        "sensor_height": SENSOR_HEIGHT,
        "sensor_width": SENSOR_WIDTH,
        "generated_at": time.time(),
        "eyes": {},
    }

    for eye in (0, 1):
        eye_dir = user_dir / str(eye)
        aerdat_path = eye_dir / "events.aerdat"
        frames_dir = eye_dir / "frames"
        if not aerdat_path.is_file():
            print(f"eye{eye}: events.aerdat not found, skipping")
            continue

        events = read_events(aerdat_path)
        valid = sensor_valid_mask(events)
        events = events[valid]

        all_summary = summarize(events["row"], events["col"], args.crop_size, args.top_k)
        print_summary(f"user{args.user} eye{eye} - all events", all_summary, args.top_k)

        label_ts, label_types = parse_frame_labels(frames_dir)
        s_mask = saccade_mask(events["t"], label_ts, label_types)
        saccade_events = events[s_mask]
        saccade_summary = summarize(
            saccade_events["row"], saccade_events["col"], args.crop_size, args.top_k
        )
        print_summary(f"user{args.user} eye{eye} - saccade ('s') events only", saccade_summary, args.top_k)

        report["eyes"][str(eye)] = {"all": all_summary, "saccade": saccade_summary}

    out_dir = args.dataset_root / "analysis" / "roi_offset_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"user{args.user}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {out_path}")
    return 0


def plot_event_bin(events: np.ndarray, x0: int, y0: int, size: int, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    colors = np.where(events["p"] == 0, "red", "green")
    ax.scatter(events["col"] - x0, events["row"] - y0, c=colors, s=4, linewidths=0)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)  # row 0 at top, matching image/sensor convention
    ax.set_aspect("equal")
    ax.set_xlabel("col (ROI-local)")
    ax.set_ylabel("row (ROI-local)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_sample_bins(args: argparse.Namespace) -> int:
    user_dir = args.dataset_root / "eye_data" / f"user{args.user}"
    eye_dir = user_dir / str(args.eye)
    aerdat_path = eye_dir / "events.aerdat"
    frames_dir = eye_dir / "frames"
    if not aerdat_path.is_file():
        raise SystemExit(f"events.aerdat not found: {aerdat_path}")

    roi_x0 = args.roi_x0
    roi_y0 = args.roi_y0
    if roi_x0 is None or roi_y0 is None:
        default_x0, default_y0 = centered_offset(args.roi_size)
        roi_x0 = default_x0 if roi_x0 is None else roi_x0
        roi_y0 = default_y0 if roi_y0 is None else roi_y0

    events = read_events(aerdat_path)
    events = events[sensor_valid_mask(events)]

    if args.stimulus != "all":
        label_ts, label_types = parse_frame_labels(frames_dir)
        events = events[stimulus_mask(events["t"], label_ts, label_types, args.stimulus)]

    events = events[roi_mask(events, roi_x0, roi_y0, args.roi_size)]

    num_windows = len(events) - args.bin_size + 1
    if num_windows <= 0:
        raise SystemExit(
            f"only {len(events)} events after filtering, need at least {args.bin_size} for one bin"
        )

    rng = np.random.default_rng(args.seed)
    replace = num_windows < args.num_bins
    start_indices = rng.choice(num_windows, size=args.num_bins, replace=replace)
    start_indices.sort()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "user": args.user,
        "eye": args.eye,
        "stimulus": args.stimulus,
        "roi_x0": roi_x0,
        "roi_y0": roi_y0,
        "roi_size": args.roi_size,
        "bin_size": args.bin_size,
        "num_bins": args.num_bins,
        "seed": args.seed,
        "filtered_event_count": len(events),
        "num_candidate_windows": num_windows,
        "generated_at": time.time(),
        "bins": [],
    }

    for i, start in enumerate(start_indices):
        window = events[start : start + args.bin_size]
        t_start, t_end = int(window["t"][0]), int(window["t"][-1])
        pol0 = int((window["p"] == 0).sum())
        pol1 = int((window["p"] == 1).sum())
        file_name = f"bin{i:03d}_start{start}_t{t_start}-{t_end}.png"
        title = (
            f"user{args.user} eye{args.eye} {args.stimulus} "
            f"roi=({roi_x0},{roi_y0},{args.roi_size}) bin{i:03d} "
            f"t=[{t_start},{t_end}]"
        )
        plot_event_bin(window, roi_x0, roi_y0, args.roi_size, title, args.out_dir / file_name)
        manifest["bins"].append(
            {
                "index": i,
                "start_event_ordinal": int(start),
                "t_start": t_start,
                "t_end": t_end,
                "polarity0_count": pol0,
                "polarity1_count": pol1,
                "file": file_name,
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {args.num_bins} bin visualizations to {args.out_dir}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


def compute_rate_curve(
    seg_events_t: np.ndarray, t_start: int, t_end: int, bin_us: int
) -> tuple[np.ndarray, np.ndarray]:
    nbins = int((t_end - t_start) // bin_us) + 1
    bin_idx = np.clip(((seg_events_t - t_start) // bin_us).astype(np.int64), 0, nbins - 1)
    counts = np.bincount(bin_idx, minlength=nbins)
    bin_centers_ms = (np.arange(nbins) + 0.5) * bin_us / 1000.0
    return bin_centers_ms, counts


def detect_phases(
    counts: np.ndarray,
    bin_us: int,
    baseline_tail_frac: float = 0.4,
    min_run_bins: int = 2,
    corrective_search_ms: float = 400.0,
    stable_sustain_ms: float = 100.0,
) -> dict:
    """Detect, from one segment's event-count-per-bin curve: where the
    (elevated) main saccade burst begins after the pre-movement reaction-time
    latency, the main peak, an optional corrective-saccade peak, and where
    the curve settles into a sustained stable (near-baseline) region."""
    n = len(counts)
    tail_start = max(int(n * (1 - baseline_tail_frac)), 1)
    baseline = float(np.median(counts[tail_start:])) if n > tail_start else float(np.median(counts))
    threshold = max(baseline * 2, baseline + 3)
    above = counts > threshold

    reaction_end_idx = None
    run = 0
    for i in range(n):
        if above[i]:
            run += 1
            if run >= min_run_bins:
                reaction_end_idx = i - min_run_bins + 1
                break
        else:
            run = 0

    main_peak_idx = int(np.argmax(counts))

    valley_idx = None
    for i in range(main_peak_idx + 1, n - 1):
        if counts[i] <= counts[i - 1] and counts[i] <= counts[i + 1]:
            valley_idx = i
            break

    corrective_peak_idx = None
    if valley_idx is not None:
        search_bins = max(int(corrective_search_ms * 1000 / bin_us), 1)
        window_end = min(n, valley_idx + search_bins)
        if window_end > valley_idx:
            local_slice = counts[valley_idx:window_end]
            candidate_idx = valley_idx + int(np.argmax(local_slice))
            is_local_max = candidate_idx == n - 1 or counts[candidate_idx] >= counts[candidate_idx + 1]
            if candidate_idx > valley_idx and counts[candidate_idx] > threshold and is_local_max:
                corrective_peak_idx = candidate_idx

    sustain_bins = max(int(stable_sustain_ms * 1000 / bin_us), 1)
    search_from = (corrective_peak_idx if corrective_peak_idx is not None else main_peak_idx) + 1
    stable_onset_idx = None
    i = search_from
    while i < n:
        if counts[i] <= threshold:
            j = i
            while j < n and counts[j] <= threshold:
                j += 1
            if j - i >= sustain_bins:
                stable_onset_idx = i
                break
            i = j
        else:
            i += 1

    def idx_to_ms(idx):
        return None if idx is None else float((idx + 0.5) * bin_us / 1000.0)

    return {
        "baseline_count_per_bin": baseline,
        "threshold_count_per_bin": threshold,
        "reaction_end_ms": idx_to_ms(reaction_end_idx),
        "main_peak_ms": idx_to_ms(main_peak_idx),
        "main_peak_count": int(counts[main_peak_idx]),
        # First local minimum after the main peak -- doubles as "main peak end"
        # and "corrective-peak search start" / "corrective peak onset", since a
        # secondary rise (if any) begins exactly here.
        "main_peak_end_ms": idx_to_ms(valley_idx),
        "corrective_peak_ms": idx_to_ms(corrective_peak_idx),
        "corrective_peak_count": int(counts[corrective_peak_idx]) if corrective_peak_idx is not None else None,
        "stable_onset_ms": idx_to_ms(stable_onset_idx),
    }


def plot_rate_curve(bin_centers_ms: np.ndarray, counts: np.ndarray, markers: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    ax.plot(bin_centers_ms, counts, linewidth=1)
    colors = {
        "reaction_end": "orange",
        "main_peak": "red",
        "main_peak_end": "brown",
        "corrective_peak": "purple",
        "stable_onset": "green",
    }
    for name, t_ms in markers.items():
        if t_ms is not None:
            ax.axvline(t_ms, color=colors.get(name, "red"), linestyle="--", linewidth=1, label=name)
    ax.set_xlabel(f"time since segment start (ms), bin={bin_centers_ms[1]-bin_centers_ms[0]:.1f}ms" if len(bin_centers_ms) > 1 else "time (ms)")
    ax.set_ylabel("events per bin")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def pick_segment_indices(num_available: int, num_segments: int, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        for idx in explicit:
            if not (0 <= idx < num_available):
                raise SystemExit(f"segment index {idx} out of range [0,{num_available})")
        return explicit
    if num_segments >= num_available:
        return list(range(num_available))
    return sorted(set(int(round(x)) for x in np.linspace(0, num_available - 1, num_segments)))


def snapshot_at(seg_events: np.ndarray, t_marker_us: int, window_size: int) -> np.ndarray:
    start_idx = int(np.searchsorted(seg_events["t"], t_marker_us, side="left"))
    return seg_events[start_idx : start_idx + window_size]


def count_events_from_ms(seg_events: np.ndarray, t_start: int, t_marker_ms: float | None) -> int | None:
    """Number of events from the given segment-relative time (ms) to the
    segment's end. None if the marker itself was not detected."""
    if t_marker_ms is None:
        return None
    t_marker_us = t_start + int(round(t_marker_ms * 1000))
    start_idx = int(np.searchsorted(seg_events["t"], t_marker_us, side="left"))
    return int(len(seg_events) - start_idx)


def run_segment_profile(args: argparse.Namespace) -> int:
    user_dir = args.dataset_root / "eye_data" / f"user{args.user}"
    eye_dir = user_dir / str(args.eye)
    aerdat_path = eye_dir / "events.aerdat"
    frames_dir = eye_dir / "frames"
    if not aerdat_path.is_file():
        raise SystemExit(f"events.aerdat not found: {aerdat_path}")

    roi_x0 = args.roi_x0
    roi_y0 = args.roi_y0
    if roi_x0 is None or roi_y0 is None:
        default_x0, default_y0 = centered_offset(args.roi_size)
        roi_x0 = default_x0 if roi_x0 is None else roi_x0
        roi_y0 = default_y0 if roi_y0 is None else roi_y0

    events = read_events(aerdat_path)
    events = events[sensor_valid_mask(events)]
    events = events[roi_mask(events, roi_x0, roi_y0, args.roi_size)]

    labels = parse_frame_labels_full(frames_dir)
    segments = build_target_segments(labels, args.stimulus)
    if not segments:
        raise SystemExit(f"no '{args.stimulus}' segments found")

    chosen = set(pick_segment_indices(len(segments), args.num_segments, args.segment_indices))

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.dataset_root / "analysis" / "segment_profiles" / f"user{args.user}_eye{args.eye}_{args.stimulus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "user": args.user,
        "eye": args.eye,
        "stimulus": args.stimulus,
        "roi_x0": roi_x0,
        "roi_y0": roi_y0,
        "roi_size": args.roi_size,
        "bin_us": args.bin_us,
        "window_size": args.window_size,
        "auto_detect": args.phase_fractions is None,
        "phase_fractions": args.phase_fractions,
        "num_segments_available": len(segments),
        "chosen_segment_indices": sorted(chosen),
        "generated_at": time.time(),
        "segments": [],
    }

    event_t = events["t"]
    detections_for_stats = []

    for idx, seg in enumerate(segments):
        t_start, t_end, row, col = seg["t_start"], seg["t_end"], seg["row"], seg["col"]
        seg_mask = (event_t >= t_start) & (event_t < t_end)
        seg_events = events[seg_mask]
        if len(seg_events) == 0:
            continue

        bin_centers_ms, counts = compute_rate_curve(seg_events["t"], t_start, t_end, args.bin_us)

        is_chosen = idx in chosen
        seg_dir = out_dir / f"seg{idx:03d}_row{int(row)}_col{int(col)}" if is_chosen else None
        if seg_dir is not None:
            seg_dir.mkdir(parents=True, exist_ok=True)

        seg_info = {
            "segment_index": idx,
            "target_row": row,
            "target_col": col,
            "t_start": t_start,
            "t_end": t_end,
            "duration_ms": (t_end - t_start) / 1000.0,
            "num_frames": seg["num_frames"],
            "num_events_in_roi": int(len(seg_events)),
            "phases": [],
        }

        if args.phase_fractions is not None:
            markers_ms = {
                f"frac{f:.2f}": f * (t_end - t_start) / 1000.0 for f in args.phase_fractions
            }
        else:
            detection = detect_phases(
                counts,
                args.bin_us,
                args.baseline_tail_frac,
                args.min_run_bins,
                args.corrective_search_ms,
                args.stable_sustain_ms,
            )
            detection["total_events_in_segment"] = int(len(seg_events))
            detection["events_from_main_peak_to_end"] = count_events_from_ms(
                seg_events, t_start, detection["main_peak_ms"]
            )
            detection["events_from_corrective_peak_to_end"] = count_events_from_ms(
                seg_events, t_start, detection["corrective_peak_ms"]
            )
            detection["events_from_main_peak_end_to_end"] = count_events_from_ms(
                seg_events, t_start, detection["main_peak_end_ms"]
            )
            seg_info["detection"] = detection
            detections_for_stats.append(detection)
            markers_ms = {
                "reaction_end": detection["reaction_end_ms"],
                "main_peak": detection["main_peak_ms"],
                "main_peak_end": detection["main_peak_end_ms"],
                "corrective_peak": detection["corrective_peak_ms"],
                "stable_onset": detection["stable_onset_ms"],
            }

        if seg_dir is not None:
            plot_rate_curve(
                bin_centers_ms,
                counts,
                markers_ms,
                title=(
                    f"user{args.user} eye{args.eye} {args.stimulus} seg{idx:03d} "
                    f"target=(row={row:.0f},col={col:.0f}) dur={(t_end-t_start)/1000:.0f}ms events={len(seg_events)}"
                ),
                out_path=seg_dir / "event_rate_curve.png",
            )

            for p_i, (name, t_ms) in enumerate(markers_ms.items()):
                if t_ms is None:
                    seg_info["phases"].append({"name": name, "t_offset_ms": None, "event_count": 0, "file": None})
                    continue
                t_marker_us = t_start + int(round(t_ms * 1000))
                window = snapshot_at(seg_events, t_marker_us, args.window_size)
                file_name = f"phase{p_i}_{name}_t{t_ms:.0f}ms_n{len(window)}.png"
                if len(window) == 0:
                    print(f"seg{idx:03d} {name}: no events available, skipping plot")
                else:
                    plot_event_bin(
                        window,
                        roi_x0,
                        roi_y0,
                        args.roi_size,
                        title=f"user{args.user} eye{args.eye} seg{idx:03d} {name} t={t_ms:.0f}ms n={len(window)}",
                        out_path=seg_dir / file_name,
                    )
                seg_info["phases"].append(
                    {
                        "name": name,
                        "t_offset_ms": t_ms,
                        "start_index_in_segment": int(np.searchsorted(seg_events["t"], t_marker_us, side="left")),
                        "event_count": int(len(window)),
                        "file": file_name if len(window) else None,
                    }
                )

            (seg_dir / "info.json").write_text(json.dumps(seg_info, indent=2))
            print(f"seg{idx:03d} (row={row:.0f},col={col:.0f}, {seg_info['duration_ms']:.0f}ms): wrote {seg_dir}")

        if is_chosen:
            manifest["segments"].append(seg_info)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nProfiled {len(chosen)} segments in detail under {out_dir}")
    print(f"Wrote manifest to {manifest_path}")

    if args.phase_fractions is None and detections_for_stats:
        stats_path = out_dir / "phase_statistics.json"
        stats = summarize_phase_detections(detections_for_stats)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"\nAggregate phase statistics over {stats['num_segments_analyzed']} segments written to {stats_path}")
        print(json.dumps(stats, indent=2))

    return 0


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def summarize_phase_detections(detections: list[dict]) -> dict:
    n_total = len(detections)
    corrective_offsets = [
        d["corrective_peak_ms"] - d["main_peak_ms"] for d in detections if d["corrective_peak_ms"] is not None
    ]

    def pct_of_total(count_key: str) -> list[float]:
        return [
            d[count_key] / d["total_events_in_segment"] * 100
            for d in detections
            if d[count_key] is not None and d["total_events_in_segment"]
        ]

    return {
        "num_segments_analyzed": n_total,
        "reaction_end_ms": _distribution([d["reaction_end_ms"] for d in detections if d["reaction_end_ms"] is not None]),
        "main_peak_ms": _distribution([d["main_peak_ms"] for d in detections if d["main_peak_ms"] is not None]),
        "main_peak_end_ms": _distribution([d["main_peak_end_ms"] for d in detections if d["main_peak_end_ms"] is not None]),
        "corrective_peak_detected_fraction": len(corrective_offsets) / n_total if n_total else None,
        "corrective_peak_offset_from_main_peak_ms": _distribution(corrective_offsets),
        "stable_onset_ms": _distribution([d["stable_onset_ms"] for d in detections if d["stable_onset_ms"] is not None]),
        "events_from_main_peak_to_seg_end": _distribution(
            [d["events_from_main_peak_to_end"] for d in detections if d["events_from_main_peak_to_end"] is not None]
        ),
        "events_from_main_peak_to_seg_end_pct_of_total": _distribution(pct_of_total("events_from_main_peak_to_end")),
        "events_from_main_peak_end_to_seg_end": _distribution(
            [d["events_from_main_peak_end_to_end"] for d in detections if d["events_from_main_peak_end_to_end"] is not None]
        ),
        "events_from_main_peak_end_to_seg_end_pct_of_total": _distribution(pct_of_total("events_from_main_peak_end_to_end")),
        "events_from_corrective_peak_to_seg_end": _distribution(
            [d["events_from_corrective_peak_to_end"] for d in detections if d["events_from_corrective_peak_to_end"] is not None]
        ),
        "events_from_corrective_peak_to_seg_end_pct_of_total": _distribution(pct_of_total("events_from_corrective_peak_to_end")),
    }


def run_valid_start_coverage(args: argparse.Namespace) -> int:
    # Local import (not at module top-level): dataset_preprocessing.py itself
    # does `from dataset_analysis import ...`, so importing it back here at
    # module load time would be circular. Deferring the import to call time
    # sidesteps that -- by the time this function runs, both modules resolve
    # fine regardless of which one was launched as __main__.
    from dataset_preprocessing import process_segment

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

    events = read_events(aerdat_path)
    events = events[sensor_valid_mask(events)]
    events = events[roi_mask(events, roi_x0, roi_y0, args.roi_size)]
    events = events.copy()
    events["row"] = events["row"] - roi_y0
    events["col"] = events["col"] - roi_x0

    labels = parse_frame_labels_full(frames_dir)
    segments = build_target_segments(labels, args.stimulus)
    if not segments:
        raise SystemExit(f"no '{args.stimulus}' segments found")
    event_t = events["t"]

    def make_ps_args(valid_start_point: str) -> argparse.Namespace:
        # A stand-in for dataset_preprocessing.py's own argparse.Namespace --
        # only the fields process_segment() reads, mirrored from --user's
        # sampling/detection knobs above so results match a real run.
        return argparse.Namespace(
            valid_start_point=valid_start_point,
            slice_length=args.slice_length,
            overlap=args.overlap,
            train_samples_per_cycle=args.train_samples_per_cycle,
            test_stride=args.test_stride,
            bin_us=args.bin_us,
            baseline_tail_frac=args.baseline_tail_frac,
            min_run_bins=args.min_run_bins,
            corrective_search_ms=args.corrective_search_ms,
            stable_sustain_ms=args.stable_sustain_ms,
        )

    per_point_reports: dict[str, list[dict]] = {}
    for vsp in args.valid_start_points:
        ps_args = make_ps_args(vsp)
        reports = []
        for idx, seg in enumerate(segments):
            seg_with_idx = {**seg, "segment_index": idx}
            seg_mask = (event_t >= seg_with_idx["t_start"]) & (event_t < seg_with_idx["t_end"])
            seg_events = events[seg_mask]
            report, _samples = process_segment(seg_with_idx, seg_events, ps_args)
            reports.append(report)
        per_point_reports[vsp] = reports

    targets = []
    for idx, seg in enumerate(segments):
        row = {
            "segment_index": idx,
            "target_row": seg["row"],
            "target_col": seg["col"],
            "duration_ms": (seg["t_end"] - seg["t_start"]) / 1000.0,
        }
        for vsp in args.valid_start_points:
            r = per_point_reports[vsp][idx]
            row[f"{vsp}_status"] = r["status"]
            row[f"{vsp}_valid_event_count"] = r.get("valid_event_count")
            row[f"{vsp}_train_samples"] = r.get("train_samples")
            row[f"{vsp}_test_samples"] = r.get("test_samples")
        targets.append(row)

    sort_point = args.sort_by if args.sort_by in args.valid_start_points else args.valid_start_points[0]
    targets.sort(key=lambda r: -(r[f"{sort_point}_test_samples"] or 0))
    for rank, row in enumerate(targets, start=1):
        row["rank"] = rank

    summary = {}
    for vsp in args.valid_start_points:
        ok_rows = [r for r in targets if r[f"{vsp}_status"] == "ok"]
        test_vals = [r[f"{vsp}_test_samples"] for r in ok_rows]
        train_vals = [r[f"{vsp}_train_samples"] for r in ok_rows]
        summary[vsp] = {
            "segments_ok": len(ok_rows),
            "segments_total": len(targets),
            "total_test_samples": int(sum(test_vals)),
            "total_train_samples": int(sum(train_vals)),
            "zero_test_targets": sum(1 for v in test_vals if v == 0),
            "zero_train_targets": sum(1 for v in train_vals if v == 0),
            "median_test_samples": float(np.median(test_vals)) if test_vals else None,
            "median_train_samples": float(np.median(train_vals)) if train_vals else None,
            "max_test_samples": max(test_vals) if test_vals else None,
            "max_train_samples": max(train_vals) if train_vals else None,
        }

    out_dir = args.out_dir
    if out_dir is None:
        points_tag = "-".join(args.valid_start_points)
        out_dir = args.dataset_root / "analysis" / "valid_start_coverage" / f"user{args.user}_eye{args.eye}_{args.stimulus}_{points_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "user": args.user,
        "eye": args.eye,
        "stimulus": args.stimulus,
        "roi_x0": roi_x0,
        "roi_y0": roi_y0,
        "roi_size": args.roi_size,
        "valid_start_points": list(args.valid_start_points),
        "sorted_by": sort_point,
        "num_segments": len(segments),
        "generated_at": time.time(),
        "summary": summary,
        "targets": targets,
    }
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== user{args.user} eye{args.eye} {args.stimulus}: valid-start-point coverage ({', '.join(args.valid_start_points)}) ===")
    for vsp in args.valid_start_points:
        s = summary[vsp]
        print(
            f"  {vsp:>16}: test total={s['total_test_samples']:,} "
            f"(median {s['median_test_samples']:.1f}, {s['zero_test_targets']}/{s['segments_total']} targets=0) | "
            f"train total={s['total_train_samples']:,} "
            f"(median {s['median_train_samples']:.1f}, {s['zero_train_targets']}/{s['segments_total']} targets=0)"
        )
    print(f"\nWrote {out_path}")
    return 0


def pick_canonical_targets(num_targets: int, seed: int) -> list[tuple[float, float]]:
    """Pick `num_targets` (row,col) grid points from the fixed 11x11 saccade
    grid. Identity-based (not list position), so the same physical target is
    compared across users regardless of any one subject missing a segment
    elsewhere in their sequence."""
    grid = [(float(r), float(c)) for r in ROW_VALUES for c in COL_VALUES]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(grid), size=num_targets, replace=False).tolist())
    return [grid[i] for i in idx]


def pick_snapshot_starts(num_available: int, window_size: int, num_points: int) -> list[int]:
    """Evenly spaced (by event index) window start positions spanning the
    full available range, start to end. If there aren't enough events for a
    full window, every point falls back to start=0 (a short/partial window)."""
    if num_available <= 0:
        return []
    if num_available <= window_size:
        return [0] * num_points
    max_start = num_available - window_size
    fracs = np.linspace(0.0, 1.0, num_points)
    return [int(round(f * max_start)) for f in fracs]


def plot_full_sensor_snapshot(
    events: np.ndarray, roi_x0: int, roi_y0: int, roi_size: int, title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=100)
    colors = np.where(events["p"] == 0, "red", "green")
    ax.scatter(events["col"], events["row"], c=colors, s=3, linewidths=0)
    ax.add_patch(Rectangle((roi_x0, roi_y0), roi_size, roi_size, edgecolor="red", facecolor="none", linewidth=1.5))
    ax.set_xlim(0, SENSOR_WIDTH)
    ax.set_ylim(SENSOR_HEIGHT, 0)  # row 0 at top, matching image/sensor convention
    ax.set_aspect("equal")
    ax.set_xlabel("col (sensor)")
    ax.set_ylabel("row (sensor)")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_multiuser_timeline(args: argparse.Namespace) -> int:
    targets = pick_canonical_targets(args.num_targets, args.seed)
    roi_x0, roi_y0 = centered_offset(args.roi_size)

    out_dir = args.out_dir
    if out_dir is None:
        users_span = f"user{min(args.users)}-{max(args.users)}"
        eyes_span = "eye" + "".join(str(e) for e in sorted(set(args.eyes)))
        out_dir = (
            args.dataset_root
            / "eye_data"
            / "analysis"
            / f"{users_span}_{eyes_span}_{args.stimulus}_{args.num_targets}targets_{args.num_points}pts_roi{args.roi_size}_seed{args.seed}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": time.time(),
        "users": args.users,
        "eyes": args.eyes,
        "stimulus": args.stimulus,
        "num_targets": args.num_targets,
        "num_points": args.num_points,
        "window_size": args.window_size,
        "roi": {"x0": roi_x0, "y0": roi_y0, "size": args.roi_size},
        "seed": args.seed,
        "targets": [{"index": i, "row": r, "col": c} for i, (r, c) in enumerate(targets)],
        "entries": [],
        "missing": [],
    }

    for user in args.users:
        user_dir = args.dataset_root / "eye_data" / f"user{user}"
        for eye in args.eyes:
            eye_dir = user_dir / str(eye)
            aerdat_path = eye_dir / "events.aerdat"
            frames_dir = eye_dir / "frames"
            if not aerdat_path.is_file():
                print(f"user{user} eye{eye}: events.aerdat not found, skipping")
                manifest["missing"].append({"user": user, "eye": eye, "reason": "no events.aerdat"})
                continue

            events = read_events(aerdat_path)
            events = events[sensor_valid_mask(events)]
            labels = parse_frame_labels_full(frames_dir)
            segments = build_target_segments(labels, args.stimulus)
            by_target: dict[tuple[float, float], list[dict]] = {}
            for seg in segments:
                by_target.setdefault((seg["row"], seg["col"]), []).append(seg)

            for t_idx, (row, col) in enumerate(targets):
                matches = by_target.get((row, col))
                if not matches:
                    print(f"user{user} eye{eye}: target{t_idx} (row={row:.0f},col={col:.0f}) missing, skipping")
                    manifest["missing"].append(
                        {"user": user, "eye": eye, "target_index": t_idx, "row": row, "col": col}
                    )
                    continue
                if len(matches) > 1:
                    print(f"user{user} eye{eye} target{t_idx}: {len(matches)} segments matched this grid point, using the first")
                seg = matches[0]
                seg_events = events[(events["t"] >= seg["t_start"]) & (events["t"] < seg["t_end"])]
                starts = pick_snapshot_starts(len(seg_events), args.window_size, args.num_points)

                target_dir = out_dir / f"user{user}_eye{eye}" / f"target{t_idx}_row{int(row)}_col{int(col)}"
                if starts:
                    target_dir.mkdir(parents=True, exist_ok=True)

                for p_idx, start in enumerate(starts):
                    window = seg_events[start : start + args.window_size]
                    if len(window) == 0:
                        print(f"user{user} eye{eye} target{t_idx} point{p_idx}: no events, skipping")
                        continue
                    frac = p_idx / (args.num_points - 1) if args.num_points > 1 else 0.0
                    t_start_us, t_end_us = int(window["t"][0]), int(window["t"][-1])
                    file_name = f"point{p_idx}_frac{frac:.2f}_t{t_start_us}-{t_end_us}.png"
                    title = (
                        f"user{user} eye{eye} target{t_idx}(row={row:.0f},col={col:.0f}) "
                        f"pt{p_idx} frac={frac:.2f} n={len(window)} t=[{t_start_us},{t_end_us}]"
                    )
                    plot_full_sensor_snapshot(window, roi_x0, roi_y0, args.roi_size, title, target_dir / file_name)
                    manifest["entries"].append(
                        {
                            "user": user,
                            "eye": eye,
                            "target_index": t_idx,
                            "row": row,
                            "col": col,
                            "point_index": p_idx,
                            "fraction": frac,
                            "segment_event_count": int(len(seg_events)),
                            "window_event_count": int(len(window)),
                            "t_start": t_start_us,
                            "t_end": t_end_us,
                            "file": str((target_dir / file_name).relative_to(out_dir)),
                        }
                    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest['entries'])} snapshots + manifest to {out_dir}")
    if manifest["missing"]:
        print(f"{len(manifest['missing'])} user/eye/target combinations were missing (see manifest.json)")
    return 0


def _parse_screen_targets(values: list[str]) -> list[tuple[float, float]]:
    targets = []
    for value in values:
        try:
            row, col = value.split(",", maxsplit=1)
            targets.append((float(row), float(col)))
        except ValueError as exc:
            raise SystemExit(f"invalid --screen-target {value!r}; expected row,col") from exc
    return targets


def _valid_window_centered_at(events: np.ndarray, timestamp: int, window_size: int) -> np.ndarray:
    """Return valid sensor events centred by event index around timestamp.

    Raw files occasionally contain out-of-sensor records, so the search slice
    expands until it contains enough valid events instead of silently making
    different users' panels different sizes.
    """
    center = int(np.searchsorted(events["t"], timestamp, side="left"))
    radius = max(window_size * 2, 2000)
    while True:
        lo = max(0, center - radius)
        hi = min(len(events), center + radius)
        candidates = events[lo:hi]
        candidates = candidates[sensor_valid_mask(candidates)]
        if len(candidates) >= window_size or (lo == 0 and hi == len(events)):
            break
        radius *= 2
    if not len(candidates):
        return candidates
    local_center = int(np.searchsorted(candidates["t"], timestamp, side="left"))
    start = max(0, min(local_center - window_size // 2, len(candidates) - window_size))
    return candidates[start : start + window_size]


def _event_geometry(events: np.ndarray, roi_x0: int, roi_y0: int, roi_size: int) -> dict:
    rows = events["row"].astype(np.float64)
    cols = events["col"].astype(np.float64)
    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())
    spread = float(np.sqrt(np.mean((rows - centroid_row) ** 2 + (cols - centroid_col) ** 2)))
    inside = roi_mask(events, roi_x0, roi_y0, roi_size)
    return {
        "duration_ms": float((int(events["t"][-1]) - int(events["t"][0])) / 1000.0),
        "roi_fraction": float(inside.mean()),
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "rms_spread_px": spread,
        "positive_polarity_fraction": float((events["p"] != 0).mean()),
    }


def _plot_pursuit_montage(
    entries: list[dict], users: list[int], eye: int, roi: dict, target: dict, out_path: Path
) -> None:
    by_user = {entry["user"]: entry for entry in entries}
    ncols = 6
    nrows = int(np.ceil(len(users) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.05 * nrows), dpi=160, squeeze=False)
    for ax, user in zip(axes.flat, users):
        entry = by_user.get(user)
        if entry is None:
            ax.text(0.5, 0.5, f"user{user}\nmissing", ha="center", va="center", transform=ax.transAxes)
        else:
            window = entry.pop("_events")
            colors = np.where(window["p"] == 0, "#d73027", "#1a9850")
            ax.scatter(window["col"], window["row"], c=colors, s=1.6, linewidths=0, alpha=0.75)
            entry["_events"] = window
            ax.set_title(
                f"user{user}  {entry['duration_ms']:.2f} ms  ROI {entry['roi_fraction']*100:.0f}%",
                fontsize=7,
            )
        ax.add_patch(
            Rectangle(
                (roi["x0"], roi["y0"]), roi["size"], roi["size"],
                edgecolor="#2c7bb6", facecolor="none", linewidth=0.8,
            )
        )
        ax.set_xlim(0, SENSOR_WIDTH)
        ax.set_ylim(SENSOR_HEIGHT, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(users) :]:
        ax.axis("off")
    fig.suptitle(
        f"EBV-Eye pursuit, eye{eye} — same screen target "
        f"({target['requested_row']:.0f}, {target['requested_col']:.0f}); "
        "red/green = polarity, blue = centred 256×256 ROI",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pursuit_summary(
    entries: list[dict], targets: list[dict], users: list[int], eye: int, out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=160, sharex=True)
    metrics = [
        ("duration_ms", "500-event duration (ms)"),
        ("roi_fraction", "Events inside centred ROI"),
        ("centroid_col", "Event centroid: sensor col"),
        ("centroid_row", "Event centroid: sensor row"),
        ("rms_spread_px", "Event-cloud RMS spread (px)"),
        ("positive_polarity_fraction", "Positive-polarity fraction"),
    ]
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for target_index, target in enumerate(targets):
        rows = {e["user"]: e for e in entries if e["target_index"] == target_index}
        xs = [u for u in users if u in rows]
        for ax, (key, label) in zip(axes.flat, metrics):
            ys = [rows[u][key] for u in xs]
            if key in {"roi_fraction", "positive_polarity_fraction"}:
                ys = [100.0 * y for y in ys]
            ax.plot(xs, ys, marker="o", markersize=2.8, linewidth=1.0, color=colors[target_index % len(colors)],
                    label=f"target ({target['requested_row']:.0f},{target['requested_col']:.0f})")
            ax.set_ylabel(label + (" (%)" if key in {"roi_fraction", "positive_polarity_fraction"} else ""), fontsize=8)
            ax.grid(alpha=0.25)
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
        ax.set_xticks(users[::2])
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_xlabel("user")
    axes[1, 1].set_xlabel("user")
    axes[1, 2].set_xlabel("user")
    fig.suptitle(f"EBV-Eye pursuit eye{eye}: cross-user raw-event geometry (same three screen targets)")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path)
    plt.close(fig)


def run_pursuit_montage(args: argparse.Namespace) -> int:
    targets_requested = _parse_screen_targets(args.screen_targets)
    roi_x0, roi_y0 = centered_offset(args.roi_size)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            args.dataset_root / "eye_data" / "analysis" /
            f"user{min(args.users)}-{max(args.users)}_eye{args.eye}_pursuit_target_aligned_win{args.window_size}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    roi = {"x0": roi_x0, "y0": roi_y0, "size": args.roi_size}
    targets = [
        {"target_index": i, "requested_row": row, "requested_col": col}
        for i, (row, col) in enumerate(targets_requested)
    ]
    entries = []
    missing = []
    for user in args.users:
        eye_dir = args.dataset_root / "eye_data" / f"user{user}" / str(args.eye)
        aerdat_path = eye_dir / "events.aerdat"
        frames_dir = eye_dir / "frames"
        if not aerdat_path.is_file() or not frames_dir.is_dir():
            missing.append({"user": user, "reason": "missing raw events or frames"})
            continue
        labels = parse_frame_labels_full(frames_dir)
        pursuit = labels[labels["stimulus"] == "p"]
        if not len(pursuit):
            missing.append({"user": user, "reason": "no pursuit labels"})
            continue
        events = read_events(aerdat_path)
        for target in targets:
            distance_sq = (
                (pursuit["row"] - target["requested_row"]) ** 2
                + (pursuit["col"] - target["requested_col"]) ** 2
            )
            label_index = int(np.argmin(distance_sq))
            label = pursuit[label_index]
            window = _valid_window_centered_at(events, int(label["t"]), args.window_size)
            if len(window) < args.window_size:
                missing.append({"user": user, "target_index": target["target_index"], "reason": "short event window"})
                continue
            geometry = _event_geometry(window, roi_x0, roi_y0, args.roi_size)
            entries.append(
                {
                    "user": user,
                    "eye": args.eye,
                    "target_index": target["target_index"],
                    "requested_screen_row": target["requested_row"],
                    "requested_screen_col": target["requested_col"],
                    "actual_screen_row": float(label["row"]),
                    "actual_screen_col": float(label["col"]),
                    "screen_target_error_px": float(np.sqrt(distance_sq[label_index])),
                    "label_timestamp": int(label["t"]),
                    "window_event_count": int(len(window)),
                    "t_start": int(window["t"][0]),
                    "t_end": int(window["t"][-1]),
                    **geometry,
                    "_events": window.copy(),
                }
            )
        del events
        print(f"user{user}: collected {sum(e['user'] == user for e in entries)} target-aligned windows")

    for target in targets:
        target_entries = [e for e in entries if e["target_index"] == target["target_index"]]
        file_name = f"target{target['target_index']}_row{int(target['requested_row'])}_col{int(target['requested_col'])}.png"
        _plot_pursuit_montage(target_entries, args.users, args.eye, roi, target, out_dir / file_name)
        target["file"] = file_name
    summary_file = "multiuser_summary.png"
    _plot_pursuit_summary(entries, targets, args.users, args.eye, out_dir / summary_file)

    serializable_entries = [{k: v for k, v in entry.items() if k != "_events"} for entry in entries]
    manifest = {
        "generated_at": time.time(),
        "definition": "target-aligned 500-event full-sensor windows; no pupil/eye-position annotation exists",
        "users": args.users,
        "eye": args.eye,
        "window_size": args.window_size,
        "sensor": {"width": SENSOR_WIDTH, "height": SENSOR_HEIGHT},
        "roi": roi,
        "targets": targets,
        "summary_file": summary_file,
        "entries": serializable_entries,
        "missing": missing,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(entries)} windows, {len(targets)} montages and summary to {out_dir}")
    return 0


# ======================================================================
# Research diagnostics. Outputs are descriptive; no model or cache is fit.
# ======================================================================


def audit_distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return dict(zip(
        ("min", "p05", "median", "p95", "max"),
        np.quantile(values, [0, .05, .5, .95, 1]).tolist(),
    ), n=int(len(values)), mean=float(values.mean()))


def audit_time_control(cache_dir: Path) -> dict:
    """Negative control using ONLY train labels and absolute window times.

    Interpolation may use training observations later than the query; this
    diagnoses within-recording split difficulty, not an online gaze tracker.
    Reading selected NPZ members does not decompress the event arrays.
    """
    with np.load(cache_dir / "train.npz", allow_pickle=False) as data:
        train_time = (data["t_start"].astype(float) + data["t_end"]) / 2
        order = np.argsort(train_time)
        train_time, train_label = train_time[order], data["labels"][order]
        train_segment = data["segment_index"][order]
    if not len(train_time):
        raise ValueError(f"Empty training set: {cache_dir}")
    result = {"cache_dir": str(cache_dir), "train_samples": len(train_time), "splits": {}}
    for split in ("validation", "test"):
        path = cache_dir / f"{split}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            query = (data["t_start"].astype(float) + data["t_end"]) / 2
            labels, segments = data["labels"], data["segment_index"]
            duration = (data["t_end"] - data["t_start"]) / 1000
        right = np.clip(np.searchsorted(train_time, query), 0, len(train_time) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(abs(query - train_time[left]) <= abs(query - train_time[right]), left, right)
        pred_nearest = train_label[nearest]
        pred_linear = np.column_stack([
            np.interp(query, train_time, train_label[:, axis]) for axis in range(2)
        ])
        error_nearest = np.linalg.norm(pred_nearest - labels, axis=1)
        error_linear = np.linalg.norm(pred_linear - labels, axis=1)
        target_counts = np.unique(labels, axis=0, return_counts=True)[1]
        result["splits"][split] = {
            "samples": len(query),
            "duration_ms": audit_distribution(duration),
            "nearest_train_time_gap_ms": audit_distribution(abs(query - train_time[nearest]) / 1000),
            "nearest_time_pixel_l2": audit_distribution(error_nearest),
            "linear_time_pixel_l2": audit_distribution(error_linear),
            "nearest_time_exact_target_fraction": float(np.all(pred_nearest == labels, axis=1).mean()),
            "nearest_time_same_segment_fraction": float((train_segment[nearest] == segments).mean()),
            "nearest_time_segment_macro_pixel_l2": float(np.mean([
                error_nearest[segments == s].mean() for s in np.unique(segments)
            ])),
            "distinct_label_count": len(target_counts),
            "per_label_sample_count": audit_distribution(target_counts),
            "outside_train_time_range_fraction": float(((query < train_time[0]) | (query > train_time[-1])).mean()),
        }
    return result


def _audit_histogram(events: np.ndarray, centered: bool = False) -> np.ndarray:
    rows, cols = events["row"].astype(float), events["col"].astype(float)
    if centered:
        rows, cols = rows - rows.mean(), cols - cols.mean()
        # Wider fixed bounds avoid dropping events during this diagnostic
        # translation. Absolute/centered distances are not directly comparable.
        bounds = [[-SENSOR_HEIGHT, SENSOR_HEIGHT], [-SENSOR_WIDTH, SENSOR_WIDTH]]
    else:
        bounds = [[0, SENSOR_HEIGHT], [0, SENSOR_WIDTH]]
    hist = np.histogram2d(rows, cols, bins=8, range=bounds)[0].ravel()
    return np.sqrt(hist / hist.sum())


def _audit_matched_summary(windows: list[dict], eyes: list[int]) -> dict:
    """Balanced complete anchors, equal user weight; descriptive ANOVA only."""
    result = {}
    for eye in eyes:
        rows = [w for w in windows if w["eye"] == eye]
        users = sorted({w["user"] for w in rows})
        lookup = {(w["user"], w["anchor"]): w for w in rows}
        anchors = [a for a in sorted({w["anchor"] for w in rows})
                   if all((u, a) in lookup for u in users)]
        summary = {"users": users, "complete_anchor_count": len(anchors)}
        if len(users) >= 2 and len(anchors) >= 2:
            for field in ("centroid_row", "centroid_col", "duration_ms", "positive_polarity_fraction"):
                matrix = np.array([[lookup[u, a][field] for a in anchors] for u in users])
                if field == "duration_ms":
                    matrix = np.log10(np.maximum(matrix, .001))
                grand = matrix.mean()
                user_effect = matrix.mean(axis=1, keepdims=True) - grand
                target_effect = matrix.mean(axis=0, keepdims=True) - grand
                total = float(np.square(matrix - grand).sum())
                summary[field + ("_log10" if field == "duration_ms" else "")] = {
                    "user_offset_variance_fraction": float(len(anchors) * np.square(user_effect).sum() / total) if total else 0.,
                    "stimulus_position_variance_fraction": float(len(users) * np.square(target_effect).sum() / total) if total else 0.,
                    "residual_variance_fraction": float(np.square(matrix - grand - user_effect - target_effect).sum() / total) if total else 0.,
                    "across_user_std_at_same_target": audit_distribution(matrix.std(axis=0)),
                }
            for key in ("_hist", "_centered_hist"):
                features = np.array([[lookup[u, a][key] for a in anchors] for u in users])
                between = [np.linalg.norm(features[i] - features[j], axis=1) / np.sqrt(2)
                           for i in range(len(users)) for j in range(i + 1, len(users))]
                within = [np.linalg.norm(features[:, i] - features[:, j], axis=1) / np.sqrt(2)
                          for i in range(len(anchors)) for j in range(i + 1, len(anchors))]
                summary[key.strip("_")] = {
                    "different_users_same_target": audit_distribution(np.concatenate(between)),
                    "same_user_different_targets": audit_distribution(np.concatenate(within)),
                }
        result[str(eye)] = summary
    return result


def _audit_correlation(records: list[dict], field: str) -> float | None:
    from scipy.stats import spearmanr
    pairs = [(r["previous_jump_screen_px"], r[field]) for r in records
             if r["previous_jump_screen_px"] is not None and r.get(field) is not None]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs).T
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def audit_label_alignment(dataset_root: Path, users: list[int]) -> list[dict]:
    """Compare labels for the SAME displayed stimulus across the two eyes.

    This does not measure eye-motion lag or certify hardware synchronization.
    First-frame target boundaries have frame-rate uncertainty. The offset sweep
    is descriptive only: no labels/timestamps in the dataset are corrected.
    """
    results = []
    for user in users:
        paths = [dataset_root / "eye_data" / f"user{user}" / str(eye) / "frames" for eye in (0, 1)]
        if not all(path.is_dir() for path in paths):
            continue
        labels = [parse_frame_labels_full(path) for path in paths]
        segments = [{(s["row"], s["col"]): s for s in build_target_segments(label, "s")} for label in labels]
        common = sorted(set(segments[0]) & set(segments[1]))
        result = {
            "user": user, "common_saccade_targets": len(common),
            "saccade_left_minus_right_onset_ms": audit_distribution([
                (segments[0][target]["t_start"] - segments[1][target]["t_start"]) / 1000 for target in common
            ]),
        }
        pursuit = [label[label["stimulus"] == "p"] for label in labels]
        if all(len(p) > 2 for p in pursuit):
            start = max(int(p["t"][0]) for p in pursuit) + 200_000
            end = min(int(p["t"][-1]) for p in pursuit) - 200_000
            query = np.arange(start, end, 25_000, dtype=np.float64)
            if len(query):
                left = np.column_stack([np.interp(query, pursuit[0]["t"], pursuit[0][axis]) for axis in ("row", "col")])
                right = np.column_stack([np.interp(query, pursuit[1]["t"], pursuit[1][axis]) for axis in ("row", "col")])
                result["pursuit_same_time_label_difference_px"] = audit_distribution(np.linalg.norm(left - right, axis=1))
            # Some recordings have offsets of many seconds. Center the search
            # on matching saccade target identities, never on gaze predictions.
            base_shift = -result["saccade_left_minus_right_onset_ms"].get("median", 0.) * 1000
            shifts = base_shift + np.arange(-200_000, 200_001, 5_000)
            start = max(int(pursuit[0]["t"][0]), int(pursuit[1]["t"][0]) - shifts.min())
            end = min(int(pursuit[0]["t"][-1]), int(pursuit[1]["t"][-1]) - shifts.max())
            query = np.arange(start, end, 25_000, dtype=np.float64)
            if len(query):
                left = np.column_stack([np.interp(query, pursuit[0]["t"], pursuit[0][axis]) for axis in ("row", "col")])
                errors = []
                for shift in shifts:
                    right = np.column_stack([np.interp(query + shift, pursuit[1]["t"], pursuit[1][axis]) for axis in ("row", "col")])
                    errors.append(np.linalg.norm(left - right, axis=1))
                best = int(np.argmin([error.mean() for error in errors]))
                result.update({
                    "offset_search_center_ms": float(base_shift / 1000),
                    "pursuit_right_query_shift_ms_at_minimum_mean_difference": float(shifts[best] / 1000),
                    "best_offset_at_search_boundary": best in (0, len(shifts) - 1),
                    "pursuit_label_difference_after_diagnostic_shift_px": audit_distribution(errors[best]),
                })
        results.append(result)
    return results


def plot_research_audit(recordings: list[dict], output: Path, window_size: int = 500, fixed_window_ms: float = 25.) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    metrics = [
        ("pursuit_duration_ms", "median", f"Median duration of {window_size} events (ms)"),
        ("pursuit_roi_fraction", "mean", "Mean fraction inside centered 256 x 256 crop"),
        ("pursuit_fixed_window_event_count", "median", f"Median event count in {fixed_window_ms:g} ms"),
    ]
    for eye, color in ((0, "#2878b5"), (1, "#d55e00")):
        rows = sorted([r for r in recordings if r["eye"] == eye], key=lambda r: r["user"])
        for axis, (metric, aggregate, label) in zip(axes.flat, metrics):
            valid = [r for r in rows if aggregate in r[metric]]
            axis.plot([r["user"] for r in valid], [r[metric][aggregate] for r in valid],
                      marker="o", markersize=3, color=color, label=f"eye{eye}")
            axis.set_ylabel(label)
        for field, style in (("first_500ms_roi_events", "-"), ("last_500ms_roi_events", "--")):
            valid = [r for r in rows if r["jump_count_spearman"][field] is not None]
            axes[1, 1].plot([r["user"] for r in valid], [r["jump_count_spearman"][field] for r in valid],
                            linestyle=style, color=color, marker="o", markersize=3,
                            label=f"eye{eye}, {'first' if style == '-' else 'last'} 500 ms")
    axes[0, 0].set_yscale("log")
    axes[1, 0].set_yscale("symlog", linthresh=1)
    axes[1, 1].set_ylabel("Saccade jump distance / event count: Spearman r")
    for axis in axes.flat:
        axis.set_xlabel("Participant ID")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle("EBV-Eye: matched pursuit stimuli and saccade event counts\nDescriptive sensor activity, not gaze accuracy")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def audit_event_alignment(dataset_root: Path, label_alignment: list[dict], users: list[int]) -> list[dict]:
    """Corroborate label-derived offsets using raw 5-ms event-count curves.

    Search near zero and near the label-derived offset. Bin counts include all
    raw records; their correlation is a synchrony diagnostic, not eye position.
    """
    results = []
    for user in users:
        labels = next(r for r in label_alignment if r["user"] == user)
        reference_shift = labels["pursuit_right_query_shift_ms_at_minimum_mean_difference"]
        timestamps, segments = [], []
        for eye in (0, 1):
            directory = dataset_root / "eye_data" / f"user{user}" / str(eye)
            raw = directory / "events.aerdat"
            events = np.memmap(raw, dtype=AERDAT_DTYPE, mode="r", shape=(raw.stat().st_size // AERDAT_DTYPE.itemsize,))
            timestamps.append(np.array(events["t"], dtype=np.int64))
            segments.append(build_target_segments(parse_frame_labels_full(directory / "frames"), "s"))
        start = max(segments[0][0]["t_start"], segments[1][0]["t_start"] - reference_shift * 1000) + 1_000_000
        end = min(segments[0][-1]["t_end"], segments[1][-1]["t_end"] - reference_shift * 1000) - 1_000_000
        query = np.arange(start, end, 5000)
        left = np.log1p(np.diff(np.searchsorted(timestamps[0], query)))
        offsets = np.unique(np.r_[reference_shift + np.arange(-300, 301, 5), np.arange(-300, 301, 5)])
        correlations = []
        for shift in offsets:
            right = np.log1p(np.diff(np.searchsorted(timestamps[1], query + shift * 1000)))
            correlations.append(float(np.corrcoef(left, right)[0, 1]) if len(left) > 2 and left.std() > 0 and right.std() > 0 else np.nan)
        if not np.isfinite(correlations).any():
            results.append({"user": user, "reason": "no variable overlapping count curves"})
            continue
        best = int(np.nanargmax(correlations))
        zero = int(np.argmin(abs(offsets)))
        results.append({
            "user": user, "bin_ms": 5, "label_right_query_shift_ms": reference_shift,
            "event_rate_right_query_shift_ms_at_max_correlation": float(offsets[best]),
            "log_count_correlation_at_best": correlations[best],
            "log_count_correlation_without_shift": correlations[zero] if np.isfinite(correlations[zero]) else None,
            "search_definition": "5ms candidates within +/-300ms of zero and label-derived shift; raw full-sensor count, saccade interval",
        })
    return results


def run_research_audit(args: argparse.Namespace) -> int:
    """Inspect all requested recordings without modifying training inputs.

    Pursuit anchors follow equal arc-length positions on reference-user eye0's
    stimulus path. Other recordings match position AND stimulus direction.
    Windows contain the last N valid FULL-SENSOR events before the label time;
    they cannot precede the start of that pursuit run. A separate 25-ms count
    includes zero-event windows. This is not a pupil annotation or accuracy test.
    """
    if args.anchors < 2 or args.window_size <= 0 or args.fixed_window_ms <= 0:
        raise ValueError("Need >=2 anchors and positive window sizes")
    output = args.out_dir.resolve()
    if output == args.dataset_root.resolve() or args.dataset_root.resolve() in output.parents:
        raise ValueError("Write diagnostics outside the raw dataset tree")
    output.mkdir(parents=True, exist_ok=True)
    ref = parse_frame_labels_full(args.dataset_root / "eye_data" / f"user{args.reference_user}" / "0" / "frames")
    ref = ref[ref["stimulus"] == "p"]
    ref_xy = np.column_stack([ref["row"], ref["col"]])
    if len(ref) < 2:
        raise ValueError("Reference recording has no pursuit path")
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ref_xy, axis=0), axis=1))]
    ref_ids = np.searchsorted(length, np.linspace(.02, .98, args.anchors) * length[-1])
    anchor_xy = ref_xy[ref_ids]
    ref_motion = np.gradient(ref_xy, axis=0)[ref_ids]
    windows, counts, recordings, missing = [], [], [], []
    roi_x0, roi_y0 = centered_offset(256)
    for user in args.users:
        for eye in args.eyes:
            directory = args.dataset_root / "eye_data" / f"user{user}" / str(eye)
            raw = directory / "events.aerdat"
            if not raw.is_file() or not (directory / "frames").is_dir():
                missing.append({"user": user, "eye": eye, "reason": "missing recording"})
                continue
            # Match read_events(): only complete 9-byte records are usable.
            # Report trailing bytes instead of interpreting them as events.
            trailing_bytes = raw.stat().st_size % AERDAT_DTYPE.itemsize
            events = np.memmap(raw, dtype=AERDAT_DTYPE, mode="r",
                               shape=(raw.stat().st_size // AERDAT_DTYPE.itemsize,))
            event_times = np.array(events["t"])
            # Verify the binary-search precondition, chunked to bound memory.
            for start in range(0, len(events), 1_000_000):
                times = event_times[max(0, start - 1):start + 1_000_000]
                if np.any(times[1:] < times[:-1]):
                    raise ValueError(f"Nonmonotonic event timestamps: {raw}")
            labels = parse_frame_labels_full(directory / "frames")
            pursuit = labels[labels["stimulus"] == "p"]
            pursuit_rows = np.flatnonzero(labels["stimulus"] == "p")
            if len(pursuit_rows) and np.any(np.diff(pursuit_rows) != 1):
                raise ValueError(f"Multiple pursuit runs require separate anchors: {directory}")
            local_windows = []
            if len(pursuit) > 2:
                xy = np.column_stack([pursuit["row"], pursuit["col"]])
                motion = np.gradient(xy, axis=0)
                first = int(np.searchsorted(event_times, pursuit["t"][0]))
                for anchor, (point, direction) in enumerate(zip(anchor_xy, ref_motion)):
                    distance = np.linalg.norm(xy - point, axis=1)
                    dot = motion @ direction
                    norm = np.linalg.norm(motion, axis=1) * np.linalg.norm(direction)
                    cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
                    distance[cosine < .9] = np.inf
                    index = int(np.argmin(distance))
                    if distance[index] > args.target_tolerance_px:
                        missing.append({"user": user, "eye": eye, "anchor": anchor, "reason": "no position/direction match"})
                        continue
                    timestamp = int(pursuit["t"][index])
                    end = int(np.searchsorted(event_times, timestamp, side="right"))
                    begin = max(first, end - args.window_size * 2)
                    while True:
                        candidate = events[begin:end]
                        valid = candidate[sensor_valid_mask(candidate) & (candidate["p"] <= 1)]
                        if len(valid) >= args.window_size or begin == first:
                            break
                        begin = max(first, begin - args.window_size * 2)
                    if len(valid) < args.window_size:
                        missing.append({"user": user, "eye": eye, "anchor": anchor, "reason": "too few events"})
                        continue
                    window = valid[-args.window_size:]
                    fixed_begin = max(first, int(np.searchsorted(event_times, timestamp - args.fixed_window_ms * 1000)))
                    fixed = events[fixed_begin:end]
                    fixed_count = int(np.count_nonzero(sensor_valid_mask(fixed) & (fixed["p"] <= 1)))
                    pixel_counts = np.unique(window["row"].astype(np.int64) * SENSOR_WIDTH + window["col"], return_counts=True)[1]
                    entry = {
                        "user": user, "eye": eye, "anchor": anchor,
                        "screen_row": float(xy[index, 0]), "screen_col": float(xy[index, 1]),
                        "label_timestamp_us": timestamp, "target_match_error_px": float(distance[index]),
                        "stimulus_direction_cosine": float(cosine[index]),
                        "stimulus_speed_px_s": float(np.linalg.norm(xy[index + 1] - xy[index - 1]) * 1e6 / (int(pursuit["t"][index + 1]) - int(pursuit["t"][index - 1]))) if 0 < index < len(xy) - 1 else None,
                        **_event_geometry(window, roi_x0, roi_y0, 256),
                        "age_of_last_event_ms": (timestamp - int(window["t"][-1])) / 1000,
                        "fixed_window_event_count": fixed_count,
                        "unique_pixel_fraction": len(pixel_counts) / args.window_size,
                        "top_pixel_event_fraction": int(pixel_counts.max()) / args.window_size,
                        "_hist": _audit_histogram(window),
                        "_centered_hist": _audit_histogram(window, centered=True),
                    }
                    windows.append(entry)
                    local_windows.append(entry)
            segments = build_target_segments(labels, "s")
            local_counts = []
            for index, segment in enumerate(segments):
                begin, end = np.searchsorted(event_times, [segment["t_start"], segment["t_end"]])
                current = events[begin:end]
                current = current[sensor_valid_mask(current) & (current["p"] <= 1)]
                cropped = current[roi_mask(current, roi_x0, roi_y0, 256)]
                previous = segments[index - 1] if index and segments[index - 1]["t_end"] == segment["t_start"] else None
                record = {
                    "user": user, "eye": eye, "segment_index": index,
                    "screen_row": segment["row"], "screen_col": segment["col"],
                    "previous_row": previous["row"] if previous else None,
                    "previous_col": previous["col"] if previous else None,
                    "previous_jump_screen_px": float(np.hypot(segment["row"] - previous["row"], segment["col"] - previous["col"])) if previous else None,
                    "duration_ms": (segment["t_end"] - segment["t_start"]) / 1000,
                    "raw_events": len(current), "roi_events": len(cropped),
                    "first_500ms_roi_events": int(np.count_nonzero(cropped["t"] < segment["t_start"] + 500_000)),
                    "last_500ms_roi_events": int(np.count_nonzero(cropped["t"] >= max(segment["t_start"], segment["t_end"] - 500_000))),
                }
                local_counts.append(record)
                counts.append(record)
            summary = {
                "user": user, "eye": eye, "raw_bytes": raw.stat().st_size,
                "incomplete_trailing_bytes": trailing_bytes,
                "raw_timestamps_nondecreasing": True, "matched_windows": len(local_windows),
                "pursuit_label_interval_ms": audit_distribution(np.diff(pursuit["t"].astype(np.int64)) / 1000),
                "saccade_segments": len(segments),
                "saccade_distinct_targets": len({(s["row"], s["col"]) for s in segments}),
                "saccade_segment_duration_ms": audit_distribution([s["duration_ms"] for s in local_counts]),
                "jump_count_spearman": {k: _audit_correlation(local_counts, k) for k in ("roi_events", "first_500ms_roi_events", "last_500ms_roi_events")},
            }
            for key in ("duration_ms", "roi_fraction", "positive_polarity_fraction", "fixed_window_event_count", "top_pixel_event_fraction", "stimulus_speed_px_s"):
                summary["pursuit_" + key] = audit_distribution([w[key] for w in local_windows if w[key] is not None])
            summary["pursuit_fixed_window_below_500_fraction"] = float(np.mean([w["fixed_window_event_count"] < 500 for w in local_windows])) if local_windows else None
            recordings.append(summary)
            print(f"user{user} eye{eye}: {len(local_windows)} pursuit anchors, {len(segments)} saccade segments", flush=True)
            del events, event_times
    predecessor = {}
    for record in counts:
        if record["previous_row"] is not None:
            key = (record["screen_row"], record["screen_col"])
            predecessor.setdefault(key, set()).add((record["previous_row"], record["previous_col"]))
    summary = {
        "schema_version": 1,
        "definition": "Exploratory matched-stimulus diagnostics, not model accuracy or pupil-ground-truth quality",
        "config": {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == "cache_dirs" else v for k, v in vars(args).items()},
        "matched_windows": len(windows), "saccade_segments": len(counts),
        "matched_pursuit": _audit_matched_summary(windows, args.eyes),
        "saccade_predecessors_per_target_across_recordings": audit_distribution([len(p) for p in predecessor.values()]),
        "recordings": recordings,
        "label_alignment": audit_label_alignment(args.dataset_root, args.users),
        "time_only_controls": [audit_time_control(path) for path in args.cache_dirs],
        "missing": missing,
        "limitations": [
            "Screen stimulus labels are not independent measurements of instantaneous gaze.",
            "Full-sensor causal 500-event windows differ from training ROI windows.",
            "Event centroid is not pupil center; lighting, blinks and noise can move it.",
            "User and camera placement are confounded; only one recording per user/eye.",
            "Same stimulus position/direction does not ensure same eye phase or speed.",
            "Variance fractions are descriptive, not causal effects or independent-sample significance tests.",
            "Label-axis offsets use complete recording labels for diagnosis; not a deployment calibration or a label-free test protocol.",
            "Time interpolation uses absolute recording time and may use later training observations; it is not an online deployable eye tracker.",
        ],
    }
    alignment_candidates = [r for r in summary["label_alignment"] if "pursuit_right_query_shift_ms_at_minimum_mean_difference" in r]
    largest_offsets = sorted(alignment_candidates, key=lambda r: abs(r["pursuit_right_query_shift_ms_at_minimum_mean_difference"]), reverse=True)[:2]
    alignment_users = sorted({r["user"] for r in largest_offsets} | {r["user"] for r in alignment_candidates if r["user"] == args.reference_user})
    summary["event_rate_alignment"] = audit_event_alignment(args.dataset_root, summary["label_alignment"], alignment_users)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, records in (("pursuit_windows", windows), ("saccade_segments", counts)):
        if records:
            keys = [key for key in records[0] if not key.startswith("_")]
            with (output / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
    plot_research_audit(recordings, output / "cohort_summary.png", args.window_size, args.fixed_window_ms)
    print(f"Audit saved to {output}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "coverage":
        return run_coverage(args)
    if args.command == "sample-bins":
        return run_sample_bins(args)
    if args.command == "segment-profile":
        return run_segment_profile(args)
    if args.command == "valid-start-coverage":
        return run_valid_start_coverage(args)
    if args.command == "multiuser-timeline":
        return run_multiuser_timeline(args)
    if args.command == "pursuit-montage":
        return run_pursuit_montage(args)
    if args.command == "research-audit":
        return run_research_audit(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
