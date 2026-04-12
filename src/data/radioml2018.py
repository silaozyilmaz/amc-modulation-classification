"""
RadioML 2018.01A (GOLD_XYZ_OSC HDF5) — disk üzerinden dilimli okuma.

Tam `X` tensörü ~2.5M × 1024 × 2 float32 ≈ 20 GiB RAM ister; bu modül
veriyi yalnızca ihtiyaç duyulan dilimlerde belleğe alır.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union, Generator

import h5py
import numpy as np

ArrayPath = Union[str, Path]


def default_h5_path() -> Path:
    """Proje köküne göre varsayılan HDF5 yolu; `RADIOML2018_H5` ile ezilebilir."""
    env = os.environ.get("RADIOML2018_H5")
    if env:
        return Path(env).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "radioml_2018" / "GOLD_XYZ_OSC.0001_1024.hdf5"


def n_samples(h5_path: Optional[ArrayPath] = None) -> int:
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        return int(f["X"].shape[0])


def describe_h5(h5_path: Optional[ArrayPath] = None) -> dict:
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        return {
            k: {"shape": tuple(f[k].shape), "dtype": str(f[k].dtype)}
            for k in f.keys()
        }


def iter_label_snr_chunks(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 8192,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    X'i belleğe almadan Y ve Z dilimleri üretir (EDA: sınıf/SNR dağılımı için).

    Her yield: y (n,) sınıf indeksi, z (n,) SNR (dataset'teki ham tamsayı).
    chunk_rows: tek seferde okunan satır; ~8k satır Y+Z birkaç MB düzeyindedir.
    """
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        n = int(f["X"].shape[0])
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            Y = f["Y"][start:stop]
            z = np.asarray(f["Z"][start:stop, 0], dtype=np.int64)
            y = np.argmax(Y, axis=1).astype(np.int64, copy=False)
            yield y, z


def iter_xyz_chunks(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 4096,
    start: int = 0,
    stop: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, int], None, None]:
    """
    X, y, z dilimleri (EDA: güç, kalite, sınıf bazlı I/Q — tek seferde sınırlı RAM).

    Yield: X (n, 1024, 2) float32, y (n,), z (n,), global_start_index
    """
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        Xd, Yd, Zd = f["X"], f["Y"], f["Z"]
        n = int(Xd.shape[0])
        hi = n if stop is None else min(stop, n)
        lo = max(0, start)
        for s in range(lo, hi, chunk_rows):
            e = min(s + chunk_rows, hi)
            X = np.asarray(Xd[s:e], dtype=np.float32)
            Y = np.asarray(Yd[s:e])
            z = np.asarray(Zd[s:e, 0], dtype=np.int64)
            y = np.argmax(Y, axis=1).astype(np.int64, copy=False)
            yield X, y, z, s


def accumulate_class_snr_heatmap(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 8192,
    n_classes: int = 24,
    snr_offset: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sınıf × SNR örnek sayısı matrisi (2016 ısı haritası ile aynı mantık).
    Sütun j = SNR değeri (j - snr_offset) dB.
    """
    n_snr = 2 * snr_offset + 1
    heat = np.zeros((n_classes, n_snr), dtype=np.int64)
    for y_blk, z_blk in iter_label_snr_chunks(h5_path, chunk_rows=chunk_rows):
        yi = y_blk.astype(np.int64, copy=False)
        zi = z_blk.astype(np.int64, copy=False) + snr_offset
        m = (zi >= 0) & (zi < n_snr) & (yi >= 0) & (yi < n_classes)
        np.add.at(heat, (yi[m], zi[m]), 1)
    snr_axis = np.arange(-snr_offset, snr_offset + 1)
    return heat, snr_axis


def read_rows_at_indices(
    indices: Sequence[int],
    h5_path: Optional[ArrayPath] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Verilen satır indekslerini okur (sıralı aralıklara bölerek HDF5 dostu).

    Dönüş sırası, `indices` ile aynıdır (yinelenen indeksler korunur).
    """
    requested = np.asarray(indices, dtype=np.int64).ravel()
    if requested.size and (requested < 0).any():
        raise ValueError("read_rows_at_indices: tüm indeksler >= 0 olmalı")
    idx = np.unique(requested)
    if idx.size == 0:
        z0 = np.zeros((0, 1024, 2), dtype=np.float32)
        return z0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    runs: List[Tuple[int, int]] = []
    a = int(idx[0])
    b = a + 1
    for t in idx[1:]:
        t = int(t)
        if t == b:
            b += 1
        else:
            runs.append((a, b))
            a, b = t, t + 1
    runs.append((a, b))
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    pieces_x: List[np.ndarray] = []
    pieces_y: List[np.ndarray] = []
    pieces_z: List[np.ndarray] = []
    with h5py.File(path, "r") as f:
        Xd, Yd, Zd = f["X"], f["Y"], f["Z"]
        for lo, hi in runs:
            X = np.asarray(Xd[lo:hi], dtype=np.float32)
            Y = np.asarray(Yd[lo:hi])
            zz = np.asarray(Zd[lo:hi, 0], dtype=np.int64)
            y = np.argmax(Y, axis=1).astype(np.int64, copy=False)
            pieces_x.append(X)
            pieces_y.append(y)
            pieces_z.append(zz)
    X_cat = np.concatenate(pieces_x, axis=0)
    y_cat = np.concatenate(pieces_y, axis=0)
    z_cat = np.concatenate(pieces_z, axis=0)
    row_for_sorted = {int(idx[i]): i for i in range(idx.size)}
    order = np.array([row_for_sorted[int(r)] for r in requested], dtype=np.int64)
    if order.size == 0:
        z0 = np.zeros((0, 1024, 2), dtype=np.float32)
        return z0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return X_cat[order], y_cat[order], z_cat[order]


def first_row_index_per_class_min_snr(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 8192,
    n_classes: int = 24,
    snr_floor: int = 16,
) -> np.ndarray:
    """Her sınıf için SNR >= snr_floor olan ilk global satır indeksi (-1 = bulunamadı)."""
    found = np.full(n_classes, -1, dtype=np.int64)
    pos = 0
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        n = int(f["X"].shape[0])
        while pos < n and (found < 0).any():
            stop = min(pos + chunk_rows, n)
            Y = f["Y"][pos:stop]
            z = np.asarray(f["Z"][pos:stop, 0], dtype=np.int64)
            y = np.argmax(Y, axis=1).astype(np.int64, copy=False)
            for i in range(y.shape[0]):
                c = int(y[i])
                if found[c] >= 0:
                    continue
                if int(z[i]) >= snr_floor:
                    found[c] = pos + i
            pos = stop
    return found


def collect_row_indices_per_class_quota(
    quota: int,
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 4096,
    n_classes: int = 24,
    snr_floor: int = 16,
) -> List[np.ndarray]:
    """
    Her sınıf için en fazla `quota` adet global indeks (SNR >= snr_floor).
    Konstelasyon için 2016’daki gibi çoklu örnek toplamakta kullanılır.
    """
    buckets: List[List[int]] = [[] for _ in range(n_classes)]
    pos = 0
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        n = int(f["X"].shape[0])
        while pos < n and any(len(b) < quota for b in buckets):
            stop = min(pos + chunk_rows, n)
            Y = f["Y"][pos:stop]
            z = np.asarray(f["Z"][pos:stop, 0], dtype=np.int64)
            y = np.argmax(Y, axis=1).astype(np.int64, copy=False)
            for i in range(y.shape[0]):
                c = int(y[i])
                if int(z[i]) < snr_floor or len(buckets[c]) >= quota:
                    continue
                buckets[c].append(pos + i)
            pos = stop
    return [np.asarray(b, dtype=np.int64) for b in buckets]


def read_slice(
    start: int,
    stop: int,
    h5_path: Optional[ArrayPath] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [start, stop) aralığını tek seferde okur.

    Dönüş: X (n, 1024, 2) float32, y sınıf indeksi (n,) int64, Z SNR (n,) int64
    """
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    with h5py.File(path, "r") as f:
        X = np.asarray(f["X"][start:stop], dtype=np.float32)
        Y = np.asarray(f["Y"][start:stop])
        Z = np.asarray(f["Z"][start:stop, 0], dtype=np.int64)
    y = Y.argmax(axis=1).astype(np.int64, copy=False)
    return X, y, Z


def train_val_indices(
    n: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministik karıştırılmış train / val indeksleri (tam dizi ~20 MiB / 2.5M örnek)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val = perm[:n_val]
    train = perm[n_val:]
    return train, val


def iterate_epoch_batches(
    h5_path: Optional[ArrayPath] = None,
    batch_size: int = 128,
    chunk_size: int = 8192,
    shuffle: bool = True,
    seed: Optional[int] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Bir epoch için mini-batch üretir; aynı anda yalnızca bir `chunk_size` dilimi RAM'de.

    - `indices`: Alt küme eğitimi için (ör. `train` indeksleri). Sıralı bloklar halinde
      okunur; blok içi ve blok sırası karıştırılabilir.
    """
    path = Path(h5_path) if h5_path is not None else default_h5_path()
    rng = np.random.default_rng(seed)

    with h5py.File(path, "r") as f:
        Xd, Yd, Zd = f["X"], f["Y"], f["Z"]
        n_total = int(Xd.shape[0])
        if indices is None:
            lo, hi = start_idx, end_idx if end_idx is not None else n_total
            idx = np.arange(lo, hi, dtype=np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64).ravel()

        # chunk_size örneklik bloklarına böl
        n_idx = idx.size
        starts = list(range(0, n_idx, chunk_size))
        if shuffle:
            rng.shuffle(starts)

        for s in starts:
            block = idx[s : s + chunk_size]
            if block.size == 0:
                continue
            # HDF5 için sıralı indeks daha verimli
            order = np.argsort(block, kind="mergesort")
            sorted_block = block[order]
            X = np.asarray(Xd[sorted_block], dtype=np.float32)
            Yraw = np.asarray(Yd[sorted_block])
            Z = np.asarray(Zd[sorted_block, 0], dtype=np.int64)
            y = Yraw.argmax(axis=1).astype(np.int64, copy=False)
            # orijinal batch sırasına dön
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            X, y, Z = X[inv], y[inv], Z[inv]

            inner = np.arange(X.shape[0])
            if shuffle:
                rng.shuffle(inner)
            for i in range(0, inner.size, batch_size):
                sel = inner[i : i + batch_size]
                yield X[sel], y[sel], Z[sel]


def streaming_power_mean_by_snr(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 4096,
    snr_offset: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SNR dilimi başına ortalama sinyal gücü mean(I²+Q²) (2016 power eğrisi ile aynı fikir)."""
    n_snr = 2 * snr_offset + 1
    psum = np.zeros(n_snr, dtype=np.float64)
    pcnt = np.zeros(n_snr, dtype=np.int64)
    for X, _, z, _ in iter_xyz_chunks(h5_path, chunk_rows=chunk_rows):
        p = np.mean(X[:, :, 0] ** 2 + X[:, :, 1] ** 2, axis=1, dtype=np.float64)
        zi = z.astype(np.int64, copy=False) + snr_offset
        m = (zi >= 0) & (zi < n_snr)
        np.add.at(psum, zi[m], p[m])
        np.add.at(pcnt, zi[m], 1)
    mean_p = np.divide(
        psum,
        np.maximum(pcnt, 1),
        out=np.zeros_like(psum),
        where=pcnt > 0,
    )
    snr_axis = np.arange(-snr_offset, snr_offset + 1)
    return mean_p, pcnt, snr_axis


def streaming_per_class_iq_moments(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 4096,
    n_classes: int = 24,
) -> dict:
    """
    Tüm veri üzerinde sınıf bazlı I/Q özetleri (2016 ADIM 5 tablosu ile aynı tür).
    Dönüş: count, I_mean, I_std, Q_mean, Q_std, I_min, I_max, Q_min, Q_max sütunları için dict.
    """
    cnt = np.zeros(n_classes, dtype=np.int64)
    sum_i = np.zeros(n_classes, dtype=np.float64)
    sumsq_i = np.zeros(n_classes, dtype=np.float64)
    sum_q = np.zeros(n_classes, dtype=np.float64)
    sumsq_q = np.zeros(n_classes, dtype=np.float64)
    min_i = np.full(n_classes, np.inf, dtype=np.float64)
    max_i = np.full(n_classes, -np.inf, dtype=np.float64)
    min_q = np.full(n_classes, np.inf, dtype=np.float64)
    max_q = np.full(n_classes, -np.inf, dtype=np.float64)

    for X, y, _, _ in iter_xyz_chunks(h5_path, chunk_rows=chunk_rows):
        for c in range(n_classes):
            m = y == c
            if not np.any(m):
                continue
            ii = X[m, :, 0].ravel().astype(np.float64, copy=False)
            qq = X[m, :, 1].ravel().astype(np.float64, copy=False)
            cnt[c] += ii.size
            sum_i[c] += ii.sum()
            sumsq_i[c] += np.dot(ii, ii)
            sum_q[c] += qq.sum()
            sumsq_q[c] += np.dot(qq, qq)
            min_i[c] = min(min_i[c], float(ii.min()))
            max_i[c] = max(max_i[c], float(ii.max()))
            min_q[c] = min(min_q[c], float(qq.min()))
            max_q[c] = max(max_q[c], float(qq.max()))

    with np.errstate(divide="ignore", invalid="ignore"):
        i_mean = np.divide(sum_i, np.maximum(cnt, 1))
        q_mean = np.divide(sum_q, np.maximum(cnt, 1))
        i_var = np.divide(sumsq_i, np.maximum(cnt, 1)) - i_mean**2
        q_var = np.divide(sumsq_q, np.maximum(cnt, 1)) - q_mean**2
    i_std = np.sqrt(np.maximum(i_var, 0.0))
    q_std = np.sqrt(np.maximum(q_var, 0.0))
    return {
        "n": cnt,
        "I_mean": i_mean,
        "I_std": i_std,
        "I_min": min_i,
        "I_max": max_i,
        "Q_mean": q_mean,
        "Q_std": q_std,
        "Q_min": min_q,
        "Q_max": max_q,
    }


def streaming_x_has_nan_inf(
    h5_path: Optional[ArrayPath] = None,
    chunk_rows: int = 8192,
) -> Tuple[bool, bool]:
    """Tüm X üzerinde NaN veya Inf var mı (tek geçiş)."""
    any_nan = False
    any_inf = False
    for X, _, _, _ in iter_xyz_chunks(h5_path, chunk_rows=chunk_rows):
        any_nan = any_nan or bool(np.isnan(X).any())
        any_inf = any_inf or bool(np.isinf(X).any())
        if any_nan and any_inf:
            break
    return any_nan, any_inf


def estimate_ram_for_slice(n_rows: int) -> float:
    """Yaklaşık RAM (GiB) — X + Y + Z için."""
    bytes_x = n_rows * 1024 * 2 * 4
    bytes_y = n_rows * 24 * 8
    bytes_z = n_rows * 8
    return (bytes_x + bytes_y + bytes_z) / (1024**3)
