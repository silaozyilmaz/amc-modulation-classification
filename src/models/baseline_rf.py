"""RadioML 2016.10a için Random Forest baseline: ön işleme, eğitim, değerlendirme."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def flatten_iq(X: np.ndarray) -> np.ndarray:
    """(N, 2, 128) -> (N, 256) I ardından Q kanalları birleştirilir."""
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Beklenen (N, 2, L), alınan {X.shape}")
    return X.reshape(X.shape[0], -1)


def zscore_per_sample(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Her satır (sinyal) için sıfır ortalama, birim varyans."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (X - mean) / std


def encode_labels(y_str: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str.astype(str))
    return y_enc, le


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Oranların toplamı 1 olmalı.")

    X_tr, X_temp, y_tr, y_temp, snr_tr, snr_temp = train_test_split(
        X,
        y,
        snr,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=random_state,
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    X_va, X_te, y_va, y_te, snr_va, snr_te = train_test_split(
        X_temp,
        y_temp,
        snr_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te, snr_tr, snr_va, snr_te


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_high_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    threshold_db: float = 0.0,
) -> dict[str, Any]:
    """
    Test setinde SNR >= threshold_db örnekleri için doğruluk ve makro F1 hesaplar.
    Sonuçları loglar; JSON için sözlük döner.
    """
    mask = snr >= threshold_db
    n = int(mask.sum())
    if n == 0:
        out: dict[str, Any] = {
            "snr_threshold_db": threshold_db,
            "n_samples": 0,
            "accuracy": None,
            "macro_f1": None,
        }
        logger.warning(
            "evaluate_high_snr: SNR >= %.1f dB için test örneği yok.",
            threshold_db,
        )
        return out

    yt = y_true[mask]
    yp = y_pred[mask]
    acc = float(accuracy_score(yt, yp))
    f1m = float(f1_score(yt, yp, average="macro", zero_division=0))
    out = {
        "snr_threshold_db": threshold_db,
        "n_samples": n,
        "accuracy": acc,
        "macro_f1": f1m,
    }
    logger.info(
        "evaluate_high_snr (test, SNR >= %.1f dB): n=%d, accuracy=%.4f, macro_f1=%.4f",
        threshold_db,
        n,
        acc,
        f1m,
    )
    return out


def accuracy_by_snr(
    y_true: np.ndarray, y_pred: np.ndarray, snr: np.ndarray
) -> dict[int, float]:
    levels = sorted(int(s) for s in np.unique(snr))
    out: dict[int, float] = {}
    for s in levels:
        m = snr == s
        if m.sum() == 0:
            continue
        out[s] = float(accuracy_score(y_true[m], y_pred[m]))
    return out


def plot_snr_vs_accuracy(
    snr_acc: dict[int, float],
    out_path: Path,
    title: str = "SNR vs Accuracy (test)",
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    snrs = sorted(snr_acc.keys())
    accs = [snr_acc[s] for s in snrs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs, accs, marker="o", linewidth=2, markersize=5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(snrs[::2])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Normalize edilmiş karışıklık matrisi (test)",
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics_json(
    path: Path,
    test_accuracy: float,
    macro_f1: float,
    val_accuracy: float | None,
    snr_accuracy: dict[int, float],
    class_names: list[str],
    high_snr_test: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "test_accuracy": test_accuracy,
        "macro_f1": macro_f1,
        "val_accuracy": val_accuracy,
        "snr_accuracy": {str(k): v for k, v in sorted(snr_accuracy.items())},
        "class_names": class_names,
    }
    if high_snr_test is not None:
        payload["high_snr_test"] = high_snr_test
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_baseline(
    clf: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    snr_test: np.ndarray,
    label_encoder: LabelEncoder,
    results_dir: Path,
    model_basename: str = "baseline_rf_radioml2016",
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_pred_val = clf.predict(X_val)
    y_pred_te = clf.predict(X_test)

    val_acc = float(accuracy_score(y_val, y_pred_val))
    test_acc = float(accuracy_score(y_test, y_pred_te))
    macro_f1 = float(f1_score(y_test, y_pred_te, average="macro", zero_division=0))

    names = list(label_encoder.classes_)
    report = classification_report(
        y_test, y_pred_te, target_names=names, zero_division=0
    )
    (results_dir / f"{model_basename}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    snr_acc = accuracy_by_snr(y_test, y_pred_te, snr_test)
    high_snr = evaluate_high_snr(y_test, y_pred_te, snr_test, threshold_db=0.0)
    plot_snr_vs_accuracy(
        snr_acc,
        results_dir / f"{model_basename}_snr_vs_accuracy.png",
    )
    plot_confusion_matrix_normalized(
        y_test,
        y_pred_te,
        names,
        results_dir / f"{model_basename}_confusion_matrix.png",
    )

    save_metrics_json(
        results_dir / f"{model_basename}_metrics.json",
        test_accuracy=test_acc,
        macro_f1=macro_f1,
        val_accuracy=val_acc,
        snr_accuracy=snr_acc,
        class_names=names,
        high_snr_test=high_snr,
        extra={"n_estimators": clf.n_estimators, "max_depth": clf.max_depth},
    )

    joblib.dump(clf, results_dir / f"{model_basename}.joblib")
    joblib.dump(label_encoder, results_dir / f"{model_basename}_label_encoder.joblib")

    return {
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "snr_accuracy": snr_acc,
        "high_snr_test": high_snr,
    }
