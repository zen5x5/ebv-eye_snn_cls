#!/usr/bin/env python3
"""Bit-exact integer contract and export flow for the frozen ebv-eye SCNN.

The module deliberately keeps the executable integer model dependent only on
NumPy.  PyTorch, Sinabs and the training code are imported lazily only by the
checkpoint/export and accelerated evaluation entry points.  This lets the RTL
environment run the arithmetic unit tests without also containing the much
larger training stack.

The frozen network is the user15/eye0, corrective-peak, 10-bin, 22-output
classification checkpoint described in ``report/rtl-block1.md``.  BatchNorm is
folded into each preceding convolution before each layer is quantized.  A
separate signed-16 scale is used per layer; weights, folded bias, threshold and
minimum membrane voltage in a layer share that scale.  Spikes are dimensionless
unsigned counts and therefore connect layers without a rescale operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HANDOFF_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DEFAULT = HANDOFF_ROOT / "trained_model" / "best_checkpoint.pt"
DATASET_ROOT_DEFAULT = Path("/data/xuzx/data/ebv-eye")
GENERATED_DEFAULT = HANDOFF_ROOT / "hardware_reference" / "exported"

WEIGHT_BITS = 16
ACTIVATION_BITS = 16
STATE_BITS = 16
ACCUMULATOR_BITS = 48
SPIKE_MAX = (1 << ACTIVATION_BITS) - 1
ACCUMULATOR_MIN = -(1 << (ACCUMULATOR_BITS - 1))
ACCUMULATOR_MAX = (1 << (ACCUMULATOR_BITS - 1)) - 1
TRACE_SEED = 0x6A09E667F3BCC909
TRACE_INDEX_CONSTANT = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class LayerSpec:
    name: str
    in_channels: int
    out_channels: int
    input_hw: int
    output_hw: int
    kernel: int
    stride: int
    padding: int
    pool: int = 1

    @property
    def weight_count(self) -> int:
        return self.out_channels * self.in_channels * self.kernel * self.kernel

    @property
    def neuron_count(self) -> int:
        return self.out_channels * self.output_hw * self.output_hw

    @property
    def output_activation_hw(self) -> int:
        return self.output_hw // self.pool

    @property
    def output_activation_count(self) -> int:
        hw = self.output_activation_hw
        return self.out_channels * hw * hw


LAYER_SPECS = (
    LayerSpec("block1", 2, 8, 128, 64, 3, 2, 1),
    LayerSpec("block2", 8, 16, 64, 32, 3, 2, 1),
    LayerSpec("block3", 16, 32, 32, 32, 3, 1, 1, 2),
    LayerSpec("block4", 32, 64, 16, 16, 3, 1, 1, 2),
    LayerSpec("block5", 64, 64, 8, 8, 3, 1, 1, 2),
    LayerSpec("block6", 64, 64, 4, 4, 3, 1, 1),
    LayerSpec("block7", 64, 32, 4, 4, 3, 1, 1),
    LayerSpec("head", 32, 22, 4, 1, 4, 1, 0),
)


@dataclass(frozen=True)
class LayerParameters:
    spec: LayerSpec
    weight: np.ndarray
    bias: np.ndarray
    threshold: int
    min_v_mem: int
    scale: float
    scale_binding: str


@dataclass(frozen=True)
class ParameterBundle:
    layers: tuple[LayerParameters, ...]
    checkpoint: str
    checkpoint_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _require_range(name: str, value: np.ndarray, low: int, high: int) -> None:
    minimum, maximum = int(value.min()), int(value.max())
    if minimum < low or maximum > high:
        raise OverflowError(f"{name} range [{minimum}, {maximum}] is outside [{low}, {high}]")


def conv2d_integer(inputs: np.ndarray, layer: LayerParameters) -> np.ndarray:
    """Exact signed-int64 NCHW convolution/correlation.

    Every multiplication and addition is explicitly performed in int64.  The
    legal hardware operand ranges prove that a 48-bit accumulator is sufficient;
    this function checks that bound after the convolution.
    """

    spec = layer.spec
    x = np.asarray(inputs, dtype=np.int64)
    expected = (spec.in_channels, spec.input_hw, spec.input_hw)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise ValueError(f"{spec.name}: expected input (N,{expected}), got {x.shape}")
    _require_range(f"{spec.name} activation", x, 0, SPIKE_MAX)
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (spec.padding, spec.padding), (spec.padding, spec.padding)),
        mode="constant",
    )
    out = np.broadcast_to(
        layer.bias.astype(np.int64).reshape(1, spec.out_channels, 1, 1),
        (len(x), spec.out_channels, spec.output_hw, spec.output_hw),
    ).copy()
    for input_channel in range(spec.in_channels):
        for kernel_y in range(spec.kernel):
            y_stop = kernel_y + spec.stride * spec.output_hw
            for kernel_x in range(spec.kernel):
                x_stop = kernel_x + spec.stride * spec.output_hw
                activation = padded[
                    :,
                    input_channel,
                    kernel_y:y_stop:spec.stride,
                    kernel_x:x_stop:spec.stride,
                ]
                coefficient = layer.weight[:, input_channel, kernel_y, kernel_x].astype(np.int64)
                out += activation[:, None, :, :] * coefficient[None, :, None, None]
    _require_range(f"{spec.name} accumulator", out, ACCUMULATOR_MIN, ACCUMULATOR_MAX)
    return out


def lif_integer(
    syn_sum: np.ndarray,
    previous_v_mem: np.ndarray,
    threshold: int,
    min_v_mem: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Integrate, MultiSpike, subtractive reset, then lower clamp.

    ``spike_full`` is used for the reset so the membrane remainder stays exact.
    The inter-layer activation is saturated to unsigned 16 bits.  Saturation is
    part of the hardware contract and the returned counter records whether it
    was exercised; all frozen real vectors are required to have zero saturation.
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    current = np.asarray(syn_sum, dtype=np.int64)
    old = np.asarray(previous_v_mem, dtype=np.int64)
    v_pre = current + old
    _require_range("v_pre", v_pre, ACCUMULATOR_MIN, ACCUMULATOR_MAX)
    spike_full = np.where(v_pre > 0, v_pre // int(threshold), 0)
    v_post = v_pre - spike_full * int(threshold)
    v_post = np.maximum(v_post, int(min_v_mem))
    state_low, state_high = _signed_range(STATE_BITS)
    _require_range("v_post", v_post, state_low, state_high)
    saturated = int(np.count_nonzero(spike_full > SPIKE_MAX))
    spike = np.clip(spike_full, 0, SPIKE_MAX).astype(np.int64)
    return v_pre, spike, v_post, saturated


def sum_pool2_integer(inputs: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(inputs, dtype=np.int64)
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError(f"2x2 sum pool needs even dimensions, got {x.shape[-2:]}")
    pooled = x.reshape(
        x.shape[0], x.shape[1], x.shape[2] // 2, 2, x.shape[3] // 2, 2
    ).sum(axis=(3, 5), dtype=np.int64)
    saturated = int(np.count_nonzero(pooled > SPIKE_MAX))
    return np.clip(pooled, 0, SPIKE_MAX).astype(np.int64), saturated


def _rotl64_array(values: np.ndarray, amount: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    return (values << np.uint64(amount)) | (values >> np.uint64(64 - amount))


def trace_digest(
    syn_sum: np.ndarray,
    v_pre: np.ndarray,
    spike: np.ndarray,
    v_post: np.ndarray,
    timestep: int,
    layer_index: int,
) -> int:
    """Cheap order-sensitive-by-index RTL/debug digest, not a security hash."""

    arrays = [np.asarray(v, dtype=np.int64).reshape(-1).view(np.uint64) for v in (syn_sum, v_pre, spike, v_post)]
    indices = np.arange(arrays[0].size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        term = (
            arrays[0]
            ^ _rotl64_array(arrays[1], 13)
            ^ _rotl64_array(arrays[2], 27)
            ^ _rotl64_array(arrays[3], 41)
            ^ (indices * np.uint64(TRACE_INDEX_CONSTANT))
        )
    reduced = int(np.bitwise_xor.reduce(term, initial=np.uint64(0)))
    return (reduced ^ TRACE_SEED ^ (int(timestep) << 8) ^ int(layer_index)) & MASK64


def run_integer_network(
    frames: np.ndarray,
    bundle: ParameterBundle,
    capture_spikes: bool = False,
) -> dict[str, Any]:
    """Run the complete frozen network with explicit integer arithmetic."""

    x_all = np.asarray(frames, dtype=np.int64)
    if x_all.ndim != 5 or tuple(x_all.shape[1:]) != (10, 2, 128, 128):
        raise ValueError(f"expected frames (N,10,2,128,128), got {x_all.shape}")
    _require_range("external input", x_all, 0, SPIKE_MAX)
    if len(bundle.layers) != len(LAYER_SPECS):
        raise ValueError("parameter bundle does not contain eight layers")

    batch = len(x_all)
    states = [
        np.zeros((batch, spec.out_channels, spec.output_hw, spec.output_hw), dtype=np.int64)
        for spec in LAYER_SPECS
    ]
    class_counts = np.zeros((batch, 22), dtype=np.int64)
    digests = np.zeros((batch, 10, len(LAYER_SPECS)), dtype=np.uint64)
    captured: dict[str, np.ndarray] = {}
    if capture_spikes:
        for index, spec in enumerate(LAYER_SPECS):
            captured[f"layer{index}_spikes"] = np.empty(
                (batch, 10, spec.out_channels, spec.output_hw, spec.output_hw), dtype=np.uint16
            )

    saturation = {"spike": 0, "pool": 0}
    layer_stats = [
        {
            "syn_sum_min": None,
            "syn_sum_max": None,
            "v_pre_min": None,
            "v_pre_max": None,
            "spike_max": 0,
            "v_post_min": None,
            "v_post_max": None,
        }
        for _ in LAYER_SPECS
    ]
    for timestep in range(10):
        activation = x_all[:, timestep]
        for layer_index, layer in enumerate(bundle.layers):
            syn_sum = conv2d_integer(activation, layer)
            v_pre, spike, v_post, spike_saturated = lif_integer(
                syn_sum,
                states[layer_index],
                layer.threshold,
                layer.min_v_mem,
            )
            states[layer_index] = v_post
            saturation["spike"] += spike_saturated
            for sample in range(batch):
                digests[sample, timestep, layer_index] = trace_digest(
                    syn_sum[sample], v_pre[sample], spike[sample], v_post[sample], timestep, layer_index
                )
            if capture_spikes:
                captured[f"layer{layer_index}_spikes"][:, timestep] = spike.astype(np.uint16)
            stats = layer_stats[layer_index]
            values = {
                "syn_sum_min": int(syn_sum.min()),
                "syn_sum_max": int(syn_sum.max()),
                "v_pre_min": int(v_pre.min()),
                "v_pre_max": int(v_pre.max()),
                "spike_max": int(spike.max()),
                "v_post_min": int(v_post.min()),
                "v_post_max": int(v_post.max()),
            }
            for key, value in values.items():
                if stats[key] is None:
                    stats[key] = value
                elif key.endswith("_min"):
                    stats[key] = min(int(stats[key]), value)
                else:
                    stats[key] = max(int(stats[key]), value)
            if layer.spec.pool == 2:
                activation, pool_saturated = sum_pool2_integer(spike)
                saturation["pool"] += pool_saturated
            else:
                activation = spike
        class_counts += activation.reshape(batch, 22)

    predictions = np.stack(
        (class_counts[:, :11].argmax(axis=1), class_counts[:, 11:].argmax(axis=1)), axis=1
    ).astype(np.uint8)
    return {
        "class_counts": class_counts,
        "predictions": predictions,
        "trace_digests": digests,
        "layer_stats": layer_stats,
        "saturation": saturation,
        **captured,
    }


def extract_parameter_bundle(checkpoint: Path) -> ParameterBundle:
    """Load the frozen checkpoint, fold BN and quantize every layer."""

    import torch

    from evaluation import load_checkpoint_into_model
    from hardware_audit import deployment_network, official_scale, threshold_values
    from model import SpikingNet8LayerBN

    checkpoint = Path(checkpoint).resolve()
    model = SpikingNet8LayerBN(
        num_timesteps=10,
        task="classification",
        spike_threshold=1.0,
        min_v_mem=-1.0,
    )
    load_checkpoint_into_model(model, checkpoint, torch.device("cpu"))
    deployed = deployment_network(model, discretize=False)
    deployed_layers = tuple(deployed.dynapcnn_layers.values())
    if len(deployed_layers) != len(LAYER_SPECS):
        raise RuntimeError(f"expected eight deployment layers, got {len(deployed_layers)}")

    low, high = _signed_range(WEIGHT_BITS)
    result = []
    for index, (spec, source) in enumerate(zip(LAYER_SPECS, deployed_layers)):
        conv = source.conv
        if tuple(conv.weight.shape) != (
            spec.out_channels,
            spec.in_channels,
            spec.kernel,
            spec.kernel,
        ):
            raise RuntimeError(f"{spec.name}: unexpected weight shape {tuple(conv.weight.shape)}")
        if tuple(conv.stride) != (spec.stride, spec.stride) or tuple(conv.padding) != (
            spec.padding,
            spec.padding,
        ):
            raise RuntimeError(f"{spec.name}: stride/padding mismatch")
        expected_pool = [[spec.pool, spec.pool]]
        if source.pool not in (expected_pool, None if spec.pool == 1 else object()):
            raise RuntimeError(f"{spec.name}: expected pool {expected_pool}, got {source.pool}")
        if float(source.rescale_weights) != 1.0:
            raise RuntimeError(f"{spec.name}: unsupported rescale_weights={source.rescale_weights}")

        weight = conv.weight.detach().double()
        bias = conv.bias.detach().double()
        thresholds = threshold_values(source.spk).double()
        candidates = {
            "weight": official_scale(weight, WEIGHT_BITS),
            "bias": official_scale(bias, WEIGHT_BITS),
            "threshold": official_scale(thresholds, STATE_BITS),
        }
        binding = min(candidates, key=candidates.get)
        scale = float(candidates[binding])
        weight_integer = np.rint(weight.numpy() * scale).astype(np.int64)
        bias_integer = np.rint(bias.numpy() * scale).astype(np.int64)
        threshold_integer = int(np.rint(float(thresholds[1]) * scale))
        min_v_mem_integer = int(np.rint(float(thresholds[0]) * scale))
        _require_range(f"{spec.name} weight", weight_integer, low, high)
        _require_range(f"{spec.name} bias", bias_integer, low, high)
        _require_range(
            f"{spec.name} threshold/min_v_mem",
            np.asarray((threshold_integer, min_v_mem_integer)),
            low,
            high,
        )
        if threshold_integer <= 0:
            raise RuntimeError(f"{spec.name}: non-positive threshold after quantization")
        result.append(
            LayerParameters(
                spec=spec,
                weight=weight_integer.astype(np.int16),
                bias=bias_integer.astype(np.int16),
                threshold=threshold_integer,
                min_v_mem=min_v_mem_integer,
                scale=scale,
                scale_binding=binding,
            )
        )
    return ParameterBundle(tuple(result), str(checkpoint), file_sha256(checkpoint))


def save_fixed_width_hex(path: Path, values: Iterable[int], bits: int) -> None:
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    with Path(path).open("w") as handle:
        for value in values:
            handle.write(f"{int(value) & mask:0{digits}x}\n")


def _relative_file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def save_parameter_bundle(bundle: ParameterBundle, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = np.concatenate([layer.weight.reshape(-1) for layer in bundle.layers]).astype(np.int16)
    biases = np.concatenate([layer.bias.reshape(-1) for layer in bundle.layers]).astype(np.int16)
    thresholds = np.asarray([layer.threshold for layer in bundle.layers], dtype=np.int16)
    min_v_mem = np.asarray([layer.min_v_mem for layer in bundle.layers], dtype=np.int16)

    paths = {
        "weights": output_dir / "weights.mem",
        "biases": output_dir / "biases.mem",
        "thresholds": output_dir / "thresholds.mem",
        "min_v_mem": output_dir / "min_v_mem.mem",
        "parameters": output_dir / "parameters.npz",
    }
    save_fixed_width_hex(paths["weights"], weights, WEIGHT_BITS)
    save_fixed_width_hex(paths["biases"], biases, WEIGHT_BITS)
    save_fixed_width_hex(paths["thresholds"], thresholds, STATE_BITS)
    save_fixed_width_hex(paths["min_v_mem"], min_v_mem, STATE_BITS)

    arrays: dict[str, np.ndarray] = {
        "thresholds": thresholds,
        "min_v_mem": min_v_mem,
        "scales": np.asarray([layer.scale for layer in bundle.layers], dtype=np.float64),
    }
    for index, layer in enumerate(bundle.layers):
        arrays[f"weight_{index}"] = layer.weight
        arrays[f"bias_{index}"] = layer.bias
    np.savez_compressed(paths["parameters"], **arrays)

    weight_offset = bias_offset = state_offset = 0
    layer_records = []
    for index, layer in enumerate(bundle.layers):
        spec = layer.spec
        layer_records.append(
            {
                "index": index,
                "name": spec.name,
                "input_shape_chw": [spec.in_channels, spec.input_hw, spec.input_hw],
                "convolution_output_shape_chw": [spec.out_channels, spec.output_hw, spec.output_hw],
                "output_activation_shape_chw": [
                    spec.out_channels,
                    spec.output_activation_hw,
                    spec.output_activation_hw,
                ],
                "kernel": spec.kernel,
                "stride": spec.stride,
                "padding": spec.padding,
                "sum_pool": spec.pool,
                "weight_layout": "OC,IC,KY,KX; KX changes fastest",
                "activation_layout": "C,Y,X; X changes fastest",
                "weight_offset": weight_offset,
                "weight_count": spec.weight_count,
                "bias_offset": bias_offset,
                "bias_count": spec.out_channels,
                "state_offset": state_offset,
                "state_count": spec.neuron_count,
                "scale": layer.scale,
                "scale_binding": layer.scale_binding,
                "threshold": layer.threshold,
                "min_v_mem": layer.min_v_mem,
                "weight_range": [int(layer.weight.min()), int(layer.weight.max())],
                "bias_range": [int(layer.bias.min()), int(layer.bias.max())],
            }
        )
        weight_offset += spec.weight_count
        bias_offset += spec.out_channels
        state_offset += spec.neuron_count

    manifest = {
        "schema_version": 1,
        "model": "SpikingNet8LayerBN classification",
        "checkpoint": bundle.checkpoint,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "quantization": {
            "weight_bits": WEIGHT_BITS,
            "bias_bits": WEIGHT_BITS,
            "threshold_bits": STATE_BITS,
            "state_bits": STATE_BITS,
            "activation_bits": ACTIVATION_BITS,
            "accumulator_bits": ACCUMULATOR_BITS,
            "rounding": "nearest, ties to even (numpy.rint / torch.round)",
            "per_layer_common_scale": True,
        },
        "totals": {
            "weights": weight_offset,
            "biases": bias_offset,
            "neuron_states": state_offset,
            "maximum_activation_words": max(spec.output_activation_count for spec in LAYER_SPECS),
        },
        "layers": layer_records,
        "files": [_relative_file_record(path, output_dir) for path in paths.values()],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_parameter_bundle(directory: Path) -> ParameterBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "parameters.npz") as data:
        layers = []
        for index, spec in enumerate(LAYER_SPECS):
            record = manifest["layers"][index]
            layers.append(
                LayerParameters(
                    spec=spec,
                    weight=data[f"weight_{index}"].astype(np.int16),
                    bias=data[f"bias_{index}"].astype(np.int16),
                    threshold=int(data["thresholds"][index]),
                    min_v_mem=int(data["min_v_mem"][index]),
                    scale=float(data["scales"][index]),
                    scale_binding=str(record["scale_binding"]),
                )
            )
    return ParameterBundle(
        tuple(layers),
        str(manifest["checkpoint"]),
        str(manifest["checkpoint_sha256"]),
    )


def fixed_real_sample_indices(dataset_size: int, count: int, seed: int = 2021) -> np.ndarray:
    if count < 1 or count > dataset_size:
        raise ValueError("real sample count is outside dataset")
    rng = np.random.default_rng(seed)
    if count == 1:
        return np.asarray((0,), dtype=np.int64)
    chosen = rng.choice(np.arange(1, dataset_size), size=count - 1, replace=False)
    return np.sort(np.concatenate((np.asarray((0,), dtype=np.int64), chosen.astype(np.int64))))


def save_real_vectors(
    bundle: ParameterBundle,
    output_dir: Path,
    dataset_root: Path,
    count: int,
) -> dict[str, Any]:
    from dataloader import build_dataset

    dataset = build_dataset(
        dataset_root,
        15,
        "test",
        num_steps=10,
        label_mode="classification",
        valid_start_point="corrective_peak",
        downsampled_to=128,
    )
    indices = fixed_real_sample_indices(len(dataset), count)
    frames = np.stack([dataset[int(index)][0].numpy() for index in indices]).astype(np.uint16)
    labels = np.stack([dataset[int(index)][1].numpy() for index in indices]).astype(np.uint8)
    result = run_integer_network(frames, bundle, capture_spikes=True)
    if result["saturation"] != {"spike": 0, "pool": 0}:
        raise RuntimeError(f"real vectors exercised unexpected saturation: {result['saturation']}")

    output_dir = Path(output_dir)
    vector_path = output_dir / "real_vectors.npz"
    vector_arrays = {
        "sample_indices": indices,
        "input_frames": frames,
        "labels": labels,
        "class_counts": result["class_counts"].astype(np.uint32),
        "predictions": result["predictions"],
        "trace_digests": result["trace_digests"],
    }
    for layer_index in range(len(LAYER_SPECS)):
        vector_arrays[f"layer{layer_index}_spikes"] = result[f"layer{layer_index}_spikes"]
    np.savez_compressed(vector_path, **vector_arrays)
    trace = {
        "schema_version": 1,
        "sample_indices": indices.tolist(),
        "labels": labels.tolist(),
        "predictions": result["predictions"].tolist(),
        "saturation": result["saturation"],
        "layer_stats": result["layer_stats"],
        "digest_algorithm": (
            "xor(seed^(t<<8)^layer, for each CHW neuron: syn_sum ^ rol(v_pre,13) ^ "
            "rol(spike,27) ^ rol(v_post,41) ^ flat_index*0x9e3779b97f4a7c15)"
        ),
        "trace_digests_hex": [
            [[f"{int(value):016x}" for value in timestep] for timestep in sample]
            for sample in result["trace_digests"]
        ],
        "files": [_relative_file_record(vector_path, output_dir)],
    }
    trace_path = output_dir / "real_trace_manifest.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def save_tiny_network_trace(output_dir: Path) -> None:
    """Save a complete hand-checkable two-neuron trace for boundary tests."""

    weights = np.asarray((2, -3, -1, 4), dtype=np.int16).reshape(2, 2, 1, 1)
    layer = LayerParameters(
        spec=LayerSpec("tiny", 2, 2, 1, 1, 1, 1, 0),
        weight=weights,
        bias=np.asarray((1, -2), dtype=np.int16),
        threshold=5,
        min_v_mem=-5,
        scale=1.0,
        scale_binding="artificial",
    )
    inputs = np.asarray(
        (
            ((1, 0),),
            ((0, 1),),
            ((3, 2),),
            ((0, 0),),
        ),
        dtype=np.int64,
    ).reshape(4, 2, 1, 1)
    state = np.zeros((1, 2, 1, 1), dtype=np.int64)
    steps = []
    for timestep, activation in enumerate(inputs):
        syn_sum = conv2d_integer(activation[None], layer)
        v_pre, spike, v_post, saturated = lif_integer(
            syn_sum, state, layer.threshold, layer.min_v_mem
        )
        state = v_post
        steps.append(
            {
                "timestep": timestep,
                "input": activation.reshape(-1).tolist(),
                "syn_sum": syn_sum.reshape(-1).tolist(),
                "v_pre": v_pre.reshape(-1).tolist(),
                "spike": spike.reshape(-1).tolist(),
                "v_post": v_post.reshape(-1).tolist(),
                "saturated": saturated,
            }
        )
    path = Path(output_dir) / "tiny_network_trace.json"
    path.write_text(json.dumps({"schema_version": 1, "steps": steps}, indent=2) + "\n")


def save_boundary_vectors(output_dir: Path) -> None:
    """Save arithmetic corner cases independently of the Python test source."""

    lif_inputs = (
        ("zero", 0, 0, 5, -5),
        ("plus_one", 1, 0, 5, -5),
        ("minus_one", -1, 0, 5, -5),
        ("threshold_equal", 5, 0, 5, -5),
        ("subtractive_multispike", 12, 0, 5, -5),
        ("negative_clamp", -20, 1, 5, -5),
        ("maximum_state", 32767, 32767, 1, -1),
        ("spike_saturation_full_reset", 131071, 0, 1, -1),
    )
    lif_vectors = []
    for name, syn_sum, old_v_mem, threshold, minimum in lif_inputs:
        v_pre, spike, v_post, saturated = lif_integer(
            np.asarray((syn_sum,), dtype=np.int64),
            np.asarray((old_v_mem,), dtype=np.int64),
            threshold,
            minimum,
        )
        lif_vectors.append(
            {
                "name": name,
                "input": {
                    "syn_sum": syn_sum,
                    "old_v_mem": old_v_mem,
                    "threshold": threshold,
                    "min_v_mem": minimum,
                },
                "expected": {
                    "v_pre": int(v_pre[0]),
                    "spike": int(spike[0]),
                    "v_post": int(v_post[0]),
                    "spike_saturated": saturated,
                },
            }
        )
    mac_inputs = (
        ("zeros", 0, 0, 0),
        ("positive_one", 0, 1, 1),
        ("negative_weight", 7, 1, -1),
        ("largest_positive_product", 0, 65535, 32767),
        ("most_negative_product", 0, 65535, -32768),
    )
    boundary = {
        "schema_version": 1,
        "lif": lif_vectors,
        "mac": [
            {
                "name": name,
                "accumulator": accumulator,
                "activation": activation,
                "weight": weight,
                "expected": accumulator + activation * weight,
            }
            for name, accumulator, activation, weight in mac_inputs
        ],
        "negative_right_shift": {
            "applicable": False,
            "reason": "the frozen per-layer integer contract performs no rescale or right shift",
        },
    }
    path = Path(output_dir) / "boundary_vectors.json"
    path.write_text(json.dumps(boundary, indent=2) + "\n")


def torch_integer_forward(frames, bundle: ParameterBundle, device: str):
    """Fast exact-integer-valued float64 convolution path for bulk evaluation.

    IEEE float64 represents every integer encountered by this contract exactly.
    The first fixed real samples are compared against ``run_integer_network``
    before this path is allowed to report full-set accuracy.
    """

    import torch
    from torch.nn import functional as functional

    x_all = frames.to(device=device, dtype=torch.float64)
    if bool((x_all < 0).any()) or bool((x_all > SPIKE_MAX).any()):
        raise OverflowError("accelerated external input is outside unsigned 16-bit range")
    if bool((x_all != torch.round(x_all)).any()):
        raise ValueError("accelerated external input contains non-integer values")
    parameters = [
        (
            torch.as_tensor(layer.weight, device=device, dtype=torch.float64),
            torch.as_tensor(layer.bias, device=device, dtype=torch.float64),
            float(layer.threshold),
            float(layer.min_v_mem),
        )
        for layer in bundle.layers
    ]
    states = [
        torch.zeros(
            (len(x_all), spec.out_channels, spec.output_hw, spec.output_hw),
            device=device,
            dtype=torch.float64,
        )
        for spec in LAYER_SPECS
    ]
    counts = torch.zeros((len(x_all), 22), device=device, dtype=torch.float64)
    for timestep in range(10):
        activation = x_all[:, timestep]
        for index, (layer, values) in enumerate(zip(bundle.layers, parameters)):
            weight, bias, threshold, minimum = values
            syn_sum = functional.conv2d(
                activation,
                weight,
                bias,
                stride=layer.spec.stride,
                padding=layer.spec.padding,
            )
            if float(syn_sum.abs().max()) > ACCUMULATOR_MAX:
                raise OverflowError(f"{layer.spec.name}: accelerated accumulator overflow")
            v_pre = syn_sum + states[index]
            spike_full = torch.where(v_pre > 0, torch.floor(v_pre / threshold), 0.0)
            states[index] = torch.maximum(v_pre - spike_full * threshold, v_pre.new_tensor(minimum))
            if bool((spike_full > SPIKE_MAX).any()):
                raise OverflowError(f"{layer.spec.name}: accelerated spike saturation")
            activation = spike_full
            if layer.spec.pool == 2:
                activation = functional.avg_pool2d(activation, 2, 2, divisor_override=1)
                if bool((activation > SPIKE_MAX).any()):
                    raise OverflowError(f"{layer.spec.name}: accelerated pool saturation")
        counts += activation.reshape(len(x_all), 22)
    if bool((counts != torch.round(counts)).any()):
        raise RuntimeError("accelerated path produced non-integer values")
    return counts.to(dtype=torch.int64)


def classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    import torch

    from dataloader import _load_ann_module, ann_dataloader

    row_values = np.asarray(ann_dataloader.ROW_VALUES)
    col_values = np.asarray(ann_dataloader.COL_VALUES)
    pred_row = row_values[predictions[:, 0]]
    pred_col = col_values[predictions[:, 1]]
    true_row = row_values[labels[:, 0]]
    true_col = col_values[labels[:, 1]]
    pixel_l2 = np.sqrt((pred_row - true_row) ** 2 + (pred_col - true_col) ** 2)

    # Reuse the project's documented angular metric rather than re-deriving its
    # physical screen geometry here.
    _load_ann_module("model")
    ann_training = _load_ann_module("training")
    pred_norm = torch.tensor(
        np.stack(ann_dataloader.normalize_label(pred_row, pred_col), axis=1), dtype=torch.float32
    )
    true_norm = torch.tensor(
        np.stack(ann_dataloader.normalize_label(true_row, true_col), axis=1), dtype=torch.float32
    )
    angular = float(ann_training.angular_error_deg(pred_norm, true_norm))
    return {
        "samples": int(len(labels)),
        "row_accuracy": float(np.mean(predictions[:, 0] == labels[:, 0])),
        "col_accuracy": float(np.mean(predictions[:, 1] == labels[:, 1])),
        "joint_accuracy": float(np.mean(np.all(predictions == labels, axis=1))),
        "pixel_l2": float(pixel_l2.mean()),
        "angular_deg": angular,
    }


def evaluate_integer_dataset(
    bundle: ParameterBundle,
    dataset_root: Path,
    device: str,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset

    from dataloader import build_dataset

    dataset = build_dataset(
        dataset_root,
        15,
        "test",
        num_steps=10,
        label_mode="classification",
        valid_start_point="corrective_peak",
        downsampled_to=128,
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, labels = [], []
    for frames, batch_labels in loader:
        counts = torch_integer_forward(frames, bundle, device)
        prediction = torch.stack(
            (counts[:, :11].argmax(1), counts[:, 11:].argmax(1)), dim=1
        )
        predictions.append(prediction.cpu().numpy().astype(np.uint8))
        labels.append(batch_labels.numpy().astype(np.uint8))
    metrics = classification_metrics(np.concatenate(predictions), np.concatenate(labels))
    metrics.update(
        {
            "device": device,
            "batch_size": batch_size,
            "numeric_path": "float64 convolution with exact integer-valued operands",
        }
    )
    return metrics


def verify_accelerated_integer_path(
    bundle: ParameterBundle,
    generated_dir: Path,
    device: str,
) -> int:
    """Prove the bulk torch path agrees with the explicit NumPy model."""

    import torch

    with np.load(Path(generated_dir) / "real_vectors.npz") as vectors:
        frames = vectors["input_frames"].copy()
        expected_counts = vectors["class_counts"].astype(np.int64)
    actual = torch_integer_forward(torch.from_numpy(frames), bundle, device).cpu().numpy()
    if not np.array_equal(actual, expected_counts):
        mismatch = np.argwhere(actual != expected_counts)[0]
        sample, output = (int(value) for value in mismatch)
        raise AssertionError(
            "accelerated integer path disagrees with explicit NumPy model at "
            f"sample={sample}, output={output}: actual={actual[sample, output]}, "
            f"expected={expected_counts[sample, output]}"
        )
    return len(frames)


def evaluate_float_dataset(
    checkpoint: Path,
    dataset_root: Path,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    """Evaluate the trained FP32 equations on CPU with Exodus converted away.

    The repository model uses the CUDA-only Exodus IAF implementation even
    when ``--device cpu`` is requested.  Sinabs' official ``exodus_to_sinabs``
    conversion replaces only that execution backend and preserves the neuron
    equation, allowing an unattended CPU evaluation without privileged GPU
    access.
    """

    import torch
    from sinabs.exodus.conversion import exodus_to_sinabs
    from torch.utils.data import DataLoader, Subset

    from dataloader import build_dataset
    from evaluation import load_checkpoint_into_model
    from model import SpikingNet8LayerBN

    model = SpikingNet8LayerBN(
        num_timesteps=10,
        task="classification",
        spike_threshold=1.0,
        min_v_mem=-1.0,
    )
    load_checkpoint_into_model(model, Path(checkpoint), torch.device("cpu"))
    model = exodus_to_sinabs(model).eval()
    dataset = build_dataset(
        dataset_root,
        15,
        "test",
        num_steps=10,
        label_mode="classification",
        valid_start_point="corrective_peak",
        downsampled_to=128,
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, labels = [], []
    with torch.no_grad():
        for frames, batch_labels in loader:
            model.reset_states()
            counts = model(frames).sum(dim=1)
            prediction = torch.stack(
                (counts[:, :11].argmax(1), counts[:, 11:].argmax(1)), dim=1
            )
            predictions.append(prediction.numpy().astype(np.uint8))
            labels.append(batch_labels.numpy().astype(np.uint8))
    metrics = classification_metrics(np.concatenate(predictions), np.concatenate(labels))
    metrics.update(
        {
            "device": "cpu",
            "batch_size": batch_size,
            "numeric_path": "trained FP32 model after official exodus_to_sinabs backend conversion",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(Path(checkpoint)),
        }
    )
    return metrics


def command_export(args: argparse.Namespace) -> int:
    bundle = extract_parameter_bundle(args.checkpoint)
    manifest = save_parameter_bundle(bundle, args.output_dir)
    save_tiny_network_trace(args.output_dir)
    save_boundary_vectors(args.output_dir)
    trace = save_real_vectors(bundle, args.output_dir, args.dataset_root, args.real_samples)
    # The central manifest covers every generated payload.  It intentionally
    # does not hash itself because a self-hash cannot be embedded recursively.
    for name in (
        "tiny_network_trace.json",
        "boundary_vectors.json",
        "real_vectors.npz",
        "real_trace_manifest.json",
    ):
        manifest["files"].append(_relative_file_record(Path(args.output_dir) / name, args.output_dir))
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "totals": manifest["totals"],
                "real_samples": len(trace["sample_indices"]),
                "predictions": trace["predictions"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    bundle = load_parameter_bundle(args.generated)
    verified_samples = verify_accelerated_integer_path(bundle, args.generated, args.device)
    metrics = evaluate_integer_dataset(
        bundle,
        args.dataset_root,
        args.device,
        args.batch_size,
        args.limit,
    )
    metrics["accelerated_path_bit_exact_samples"] = verified_samples
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


def command_evaluate_float(args: argparse.Namespace) -> int:
    metrics = evaluate_float_dataset(
        args.checkpoint,
        args.dataset_root,
        args.batch_size,
        args.limit,
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="fold BN, quantize, export memories and vectors")
    export.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    export.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    export.add_argument("--output-dir", type=Path, default=GENERATED_DEFAULT)
    export.add_argument("--real-samples", type=int, default=10)
    export.set_defaults(function=command_export)

    evaluate = subparsers.add_parser("evaluate", help="run bulk integer-valued evaluation")
    evaluate.add_argument("--generated", type=Path, default=GENERATED_DEFAULT)
    evaluate.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--output", type=Path, default=None)
    evaluate.set_defaults(function=command_evaluate)

    evaluate_float = subparsers.add_parser(
        "evaluate-float", help="run the trained FP32 equations through the CPU Sinabs backend"
    )
    evaluate_float.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    evaluate_float.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    evaluate_float.add_argument("--batch-size", type=int, default=16)
    evaluate_float.add_argument("--limit", type=int, default=None)
    evaluate_float.add_argument("--output", type=Path, default=None)
    evaluate_float.set_defaults(function=command_evaluate_float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
