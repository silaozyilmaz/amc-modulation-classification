"""RadioML 2016.10a pickle yükleme ve dizi dönüşümleri."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numpy.exceptions import VisibleDeprecationWarning as _NumpyVisibleDeprecationWarning
except ImportError:
    from numpy import VisibleDeprecationWarning as _NumpyVisibleDeprecationWarning


def load_rml2016_dict(pkl_path: str | Path, encoding: str = "latin1") -> dict[Any, Any]:
    """RML2016.10a_dict.pkl dosyasını pickle ile yükler."""
    path = Path(pkl_path)
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=_NumpyVisibleDeprecationWarning)
            return pickle.load(f, encoding=encoding)


def dict_to_arrays(dataset: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sözlükten (mod, SNR) anahtarlarıyla X, y (modülasyon adları), snr dizilerini üretir.

    Returns
    -------
    X : ndarray, shape (N, 2, 128)
    y : ndarray, shape (N,), str modülasyon adları
    snr : ndarray, shape (N,), SNR (dB)
    """
    keys = sorted(dataset.keys(), key=lambda k: (k[0], k[1]))
    x_chunks: list[np.ndarray] = []
    y_list: list[str] = []
    snr_list: list[int] = []

    for mod, snr_db in keys:
        arr = np.asarray(dataset[(mod, snr_db)])
        n = arr.shape[0]
        x_chunks.append(arr)
        y_list.extend([mod] * n)
        snr_list.extend([int(snr_db)] * n)

    X = np.concatenate(x_chunks, axis=0)
    y = np.array(y_list, dtype=object)
    snr = np.array(snr_list, dtype=np.int32)
    return X, y, snr
