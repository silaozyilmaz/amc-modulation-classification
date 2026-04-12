#!/usr/bin/env python3
"""
RadioML 2018 HDF5'ten daha küçük bir alt küme üretir (dizüstü / düşük RAM için).

Örnek:
  python scripts/extract_radioml2018_subset.py --n-samples 50000 \\
    --output ../data/processed/radioml2018_subset_50k.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Proje kökü: scripts/ -> parent
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.radioml2018 import default_h5_path  # noqa: E402


def _contiguous_runs(sorted_idx: np.ndarray) -> list[tuple[int, int]]:
    if sorted_idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    a = int(sorted_idx[0])
    b = a
    for x in sorted_idx[1:]:
        xi = int(x)
        if xi == b + 1:
            b = xi
        else:
            runs.append((a, b + 1))
            a = b = xi
    runs.append((a, b + 1))
    return runs


def extract_subset(
    src: Path,
    dst: Path,
    n_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as fsrc:
        n = int(fsrc["X"].shape[0])
        k = min(n_samples, n)
        pick = np.sort(rng.choice(n, size=k, replace=False))
        runs = _contiguous_runs(pick)

        with h5py.File(dst, "w") as fdst:
            dx = fdst.create_dataset(
                "X",
                shape=(k, 1024, 2),
                dtype="float32",
                chunks=(min(512, k), 1024, 2),
            )
            dy = fdst.create_dataset(
                "Y",
                shape=(k, 24),
                dtype="int64",
                chunks=(min(512, k), 24),
            )
            dz = fdst.create_dataset(
                "Z",
                shape=(k, 1),
                dtype="int64",
                chunks=(min(512, k), 1),
            )

            out_pos = 0
            for a, b in tqdm(runs, desc="HDF5 blokları kopyalanıyor"):
                block_len = b - a
                dx[out_pos : out_pos + block_len] = fsrc["X"][a:b]
                dy[out_pos : out_pos + block_len] = fsrc["Y"][a:b]
                dz[out_pos : out_pos + block_len] = fsrc["Z"][a:b]
                out_pos += block_len

    assert out_pos == k, "Yazılan örnek sayısı beklenenle uyuşmuyor"


def main() -> None:
    ap = argparse.ArgumentParser(description="RadioML 2018 HDF5 alt kümesi oluştur")
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Kaynak HDF5 (varsayılan: RADIOML2018_H5 veya proje varsayılanı)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "radioml2018_subset.h5",
        help="Çıktı HDF5 dosyası",
    )
    ap.add_argument("--n-samples", type=int, default=50_000, help="Örnek sayısı")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = args.src or default_h5_path()
    if not src.is_file():
        raise SystemExit(f"Kaynak bulunamadı: {src}")

    extract_subset(src, args.output, args.n_samples, args.seed)
    print(f"Tamam: {args.output} ({args.n_samples} örnek)")


if __name__ == "__main__":
    main()
