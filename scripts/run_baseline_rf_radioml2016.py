#!/usr/bin/env python3
"""
RadioML 2016.10a Random Forest baseline: uçtan uca eğitim ve değerlendirme.
Çıktılar: models/results/
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.radioml2016 import dict_to_arrays, load_rml2016_dict
from src.models.baseline_rf import (
    encode_labels,
    evaluate_baseline,
    flatten_iq,
    stratified_train_val_test_split,
    train_random_forest,
    zscore_per_sample,
)

DEFAULT_PKL = ROOT / "data" / "raw" / "radioml_2016" / "RML2016.10a_dict.pkl"
RESULTS_DIR = ROOT / "models" / "results"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    pkl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PKL
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {pkl_path}")

    t0 = time.perf_counter()
    dataset = load_rml2016_dict(pkl_path)
    X, y_str, snr = dict_to_arrays(dataset)
    print(f"Yüklendi: X {X.shape}, y {y_str.shape}, snr {snr.shape}")

    X_flat = flatten_iq(X)
    X_norm = zscore_per_sample(X_flat)
    y, le = encode_labels(y_str)
    print(f"Sınıf sayısı: {len(le.classes_)} → {list(map(str, le.classes_))}")

    (
        X_tr,
        X_va,
        X_te,
        y_tr,
        y_va,
        y_te,
        _snr_tr,
        _snr_va,
        snr_te,
    ) = stratified_train_val_test_split(X_norm, y, snr)
    print(f"Bölme: train {X_tr.shape[0]}, val {X_va.shape[0]}, test {X_te.shape[0]}")

    clf = train_random_forest(X_tr, y_tr)
    metrics = evaluate_baseline(
        clf, X_va, y_va, X_te, y_te, snr_te, le, RESULTS_DIR
    )

    elapsed = time.perf_counter() - t0
    print(f"Doğrulama doğruluğu: {metrics['val_accuracy']:.4f}")
    print(f"Test doğruluğu:       {metrics['test_accuracy']:.4f}")
    print(f"Makro F1:             {metrics['macro_f1']:.4f}")
    hs = metrics["high_snr_test"]
    if hs.get("n_samples", 0) > 0 and hs.get("accuracy") is not None:
        print(
            f"Test (SNR >= {hs['snr_threshold_db']:.0f} dB): "
            f"n={hs['n_samples']}, doğruluk={hs['accuracy']:.4f}, makro F1={hs['macro_f1']:.4f}"
        )
    print(f"Toplam süre:          {elapsed / 60:.2f} dk")
    print(f"Sonuçlar:             {RESULTS_DIR}")


if __name__ == "__main__":
    main()
