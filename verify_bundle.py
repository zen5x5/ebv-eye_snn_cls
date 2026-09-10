#!/usr/bin/env python3
"""Verify the immutable model and principal hardware-reference artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED = {
    "trained_model/best_checkpoint.pt": "a2ca40bc306940aafae81e8bda6eb0c64cf81a4f9696c0773de69f0c054dc678",
    "trained_model/final_checkpoint.pt": "e2e8a6fc1ea16b275d392aec0e93b645b0f277b2a9363a2b6df01f69399e3379",
    "trained_model/config.json": "5cff5ce79fe2f6d63b0e99aa869d683fef0a38d86ffa06b2a7ca3c5a20cb002d",
    "trained_model/history.json": "8298bf76974275ea7c70862f28e625fd7e8f9cbcf5bf437fc092f5b5fcb79553",
    "trained_model/train.log": "bf96c907b3408416a02a82a6de7553852295cda61ef8a7f66f4bf01e8a85db3b",
    "hardware_reference/exported/parameters.npz": "4bb23ac3b56dd6415540f70e07abe73ee52f83a4907659c1e915051b87a53366",
    "hardware_reference/exported/weights.mem": "be2021141093e658e0180ef05000893268f55e8fb797972d738ffeb75311855d",
    "hardware_reference/exported/biases.mem": "0c0990824258787986592372bd942a467889a6a6f585fbced8cba53034662ee5",
    "hardware_reference/exported/thresholds.mem": "1488fbb0bed4797756936c125a366e52f2dbcf14ce779a9abd7bd674aa8c345c",
    "hardware_reference/exported/min_v_mem.mem": "5c1aa298f3ee9816a3be765bbcfa205b67225cfb7aa5de4a09a520cace71a892",
    "hardware_reference/exported/manifest.json": "e3acd724936ccf02b48bf1e638a12947da787b6ba9b7df2da0cdf28692dda478",
    "hardware_reference/exported/real_vectors.npz": "63692cbe6988d8aff2abacfad5c39f469fca32ad8f8af984638b16b3514be15c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    failures = []
    for relative, expected in EXPECTED.items():
        path = root / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        state = "PASS" if actual == expected else "FAIL"
        print(f"{state}  {relative}")
        if state == "FAIL":
            failures.append((relative, expected, actual))
    if failures:
        details = "\n".join(f"{name}: expected {expected}, got {actual}" for name, expected, actual in failures)
        raise SystemExit(f"bundle verification failed:\n{details}")
    print(f"verified {len(EXPECTED)} immutable artifacts")


if __name__ == "__main__":
    main()
