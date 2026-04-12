#!/usr/bin/env python3
"""02_eda_2018.ipynb üretir — 01_eda_2016 ile aynı analiz adımları, chunk/stream okuma."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_OUT = ROOT / "notebooks" / "02_eda_2018.ipynb"


def src(s: str) -> list:
    lines = s.strip().split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(s)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(s),
    }


cells: list = []

cells.append(
    md(
        """# RadioML 2018.01A — EDA (`01_eda_2016` ile aynı adımlar)

Tam HDF5 ~21 GB; **`X` tamamı RAM'e sığdırılmaz.** Bu defter, 2016'daki **ADIM-0 … ADIM-6** analizlerinin 2018 karşılığını üretir:

- Dağılım ve ısı haritası: yalnızca **`Y`/`Z`** veya tek geçiş **`iter_label_snr_chunks`**
- Dalga / güç / sınıf bazlı I/Q / NaN kontrolü: **`iter_xyz_chunks`** (sınırlı `chunk_rows`)
- Konstelasyon (çoklu örnek): **`collect_row_indices_per_class_quota`** + **`read_rows_at_indices`**
- Öznitelik / PCA: bellek için **`read_slice(0, SUBSET_N)`** alt kümesi (2016'daki 12k örnek mantığı)

**Çıktılar:** `../results/results_2018/` — dosya adları 2016 ile aynı önek: `eda_2018_*`."""
    )
)

cells.append(
    md(
        """## Ortam"""
    )
)

cells.append(
    code(
        """import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.radioml2018 import (
    accumulate_class_snr_heatmap,
    collect_row_indices_per_class_quota,
    default_h5_path,
    describe_h5,
    estimate_ram_for_slice,
    first_row_index_per_class_min_snr,
    n_samples,
    read_rows_at_indices,
    read_slice,
    streaming_per_class_iq_moments,
    streaming_power_mean_by_snr,
    streaming_x_has_nan_inf,
)

RESULTS = ROOT / "results" / "results_2018"
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")

# SNR histogram / ısı haritası için eksen (2016 ile uyumlu aralık; gerekirse genişletilir)
SNR_OFF = 40
CHUNK_YZ = 8192
CHUNK_XYZ = 4096

print("ROOT:", ROOT)
print("RESULTS:", RESULTS)"""
    )
)

cells.append(md("""## ADIM-0 — Yapısal inceleme ve veri yükleme (HDF5 / h5py)"""))

cells.append(
    code(
        """H5_PATH = default_h5_path()
print("HDF5:", H5_PATH)
if not H5_PATH.is_file():
    raise FileNotFoundError(
        f"Dosya yok: {H5_PATH}\\nİndirin veya RADIOML2018_H5 ortam değişkeni ile yolu verin."
    )

info = describe_h5(H5_PATH)
for k, v in info.items():
    print(k, v)

N = n_samples(H5_PATH)
print(f"\\nToplam örnek: {N:,}")
for nrows in (1024, 4096, 8192):
    print(f"  X[{nrows}] satır ≈ {estimate_ram_for_slice(nrows):.2f} GiB (yaklaşık)")"""
    )
)

cells.append(
    code(
        """classes_path = ROOT / "data" / "raw" / "radioml_2018" / "classes-fixed.json"
with open(classes_path, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)
n_cls = len(CLASS_NAMES)
name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
print(n_cls, "sınıf")"""
    )
)

cells.append(
    code(
        """print("X üzerinde NaN / Inf taraması (chunk)...")
any_nan, any_inf = streaming_x_has_nan_inf(H5_PATH, chunk_rows=CHUNK_XYZ)
print(f"NaN var: {any_nan}, Inf var: {any_inf}")
shape_err = info["X"]["shape"] != (N, 1024, 2)
print(f"X şekli beklenen (N,1024,2): shape_err={shape_err}")"""
    )
)

cells.append(md("""## ADIM-1 — Dağılım analizi (sınıf, SNR, ısı haritası)"""))

cells.append(
    code(
        """heat, snr_axis = accumulate_class_snr_heatmap(
    H5_PATH, chunk_rows=CHUNK_YZ, n_classes=n_cls, snr_offset=SNR_OFF
)
class_counts = heat.sum(axis=1)
print("Sınıf toplamları:", class_counts.sum(), "N:", N)"""
    )
)

cells.append(
    code(
        """fig, ax = plt.subplots(figsize=(14, 4))
bars = ax.bar(range(n_cls), class_counts, color="steelblue", edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(class_counts) * 0.01,
            f"{int(val):,}", ha="center", va="bottom", fontsize=7, rotation=0)
ax.set_xticks(range(n_cls))
ax.set_xticklabels(CLASS_NAMES, rotation=55, ha="right", fontsize=8)
ax.set_title("RadioML 2018 — Sınıf başına örnek sayısı")
ax.set_ylabel("Örnek sayısı")
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(class_counts)"""
    )
)

cells.append(
    code(
        """snr_counts = heat.sum(axis=0)
mask = snr_counts > 0
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(snr_axis[mask], snr_counts[mask], width=1.6, color="coral", edgecolor="white", linewidth=0.5)
ax.axvline(x=0, color="navy", linestyle="--", linewidth=1.0, label="SNR = 0 dB")
ax.set_title("RadioML 2018 — SNR dağılımı")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Örnek sayısı")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_snr_distribution.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    code(
        """col_mask = heat.sum(axis=0) > 0
Hplot = heat[:, col_mask]
snr_labels = snr_axis[col_mask]
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(
    Hplot.astype(int),
    annot=False,
    fmt="d",
    cmap="YlOrRd",
    linewidths=0.2,
    ax=ax,
    xticklabels=snr_labels,
    yticklabels=CLASS_NAMES,
)
ax.set_title("RadioML 2018 — Sınıf × SNR örnek sayısı")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Modülasyon")
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_class_snr_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Min hücre:", Hplot.min(), "Max hücre:", Hplot.max())"""
    )
)

cells.append(md("""## ADIM-2 — Dalga biçimleri (1024 adım; yüksek SNR'de sınıf başına bir örnek)"""))

cells.append(
    code(
        """SNR_WAVE = 16
idx_per_class = first_row_index_per_class_min_snr(
    H5_PATH, chunk_rows=CHUNK_YZ, n_classes=n_cls, snr_floor=SNR_WAVE
)
missing = [CLASS_NAMES[i] for i in range(n_cls) if idx_per_class[i] < 0]
if missing:
    print("Uyarı: yüksek SNR bulunamayan sınıflar:", missing)
valid = idx_per_class >= 0
read_idx = idx_per_class[valid]
X_hi, y_hi, z_hi = read_rows_at_indices(read_idx, H5_PATH)
print("Okunan satır:", X_hi.shape)"""
    )
)

cells.append(
    code(
        """# 24 panel: 4 satır x 6 sütun (2016: 11 sınıf 3x4)
n_show = int(valid.sum())
ncols = 6
nrows = int(np.ceil(n_show / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
axes = np.atleast_2d(axes)
k = 0
for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]
        if k >= n_show:
            ax.set_visible(False)
            continue
        sig = X_hi[k]
        i_ch = sig[:, 0]
        ax.plot(i_ch, color="steelblue", linewidth=0.7)
        mod = CLASS_NAMES[int(y_hi[k])]
        ax.set_title(f"{mod}  SNR={int(z_hi[k])} dB", fontsize=8)
        ax.set_ylim(-0.25, 0.25)
        k += 1
fig.suptitle(f"2018 — I kanalı (SNR ≥ {SNR_WAVE} dB, sınıf başına 1 örnek)", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_waveforms_I_highsnr.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    code(
        """# 2016'daki gibi 4 temsili sınıf — 2018 isimleri
sel = []
for lab in ("BPSK", "16QAM", "FM", "AM-DSB-SC"):
    if lab in name_to_idx:
        sel.append(name_to_idx[lab])
fig, axes = plt.subplots(len(sel), 2, figsize=(14, 2.8 * len(sel)))
if len(sel) == 1:
    axes = np.array([axes])
for row, ci in enumerate(sel):
    j = np.where(y_hi == ci)[0]
    sig = X_hi[j[0]] if len(j) else X_hi[0]
    axes[row, 0].plot(sig[:, 0], color="steelblue", linewidth=0.8)
    axes[row, 1].plot(sig[:, 1], color="coral", linewidth=0.8)
    axes[row, 0].set_title(f"{CLASS_NAMES[ci]} — I")
    axes[row, 1].set_title(f"{CLASS_NAMES[ci]} — Q")
    for ax in axes[row]:
        ax.set_ylim(-0.25, 0.25)
fig.suptitle("I ve Q karşılaştırması (yüksek SNR diliminden)", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_waveforms_IQ_comparison.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(md("""## Adım 3 — SNR etkisi (örnek: BPSK)"""))

cells.append(
    code(
        """def find_pair_indices(h5_path, pairs, chunk_rows=8192):
    need = set(pairs)
    out = {}
    pos = 0
    with h5py.File(h5_path, "r") as f:
        n = int(f["X"].shape[0])
        while need and pos < n:
            stop = min(pos + chunk_rows, n)
            Y = f["Y"][pos:stop]
            z = np.asarray(f["Z"][pos:stop, 0], dtype=np.int64)
            y = np.argmax(Y, axis=1).astype(np.int64)
            for i in range(len(y)):
                t = (int(y[i]), int(z[i]))
                if t in need:
                    out[t] = pos + i
                    need.remove(t)
            pos = stop
    return out


bi = name_to_idx.get("BPSK", 0)
targets = [(bi, s) for s in (-20, -10, 0, 18)]
found = find_pair_indices(H5_PATH, set(targets))
print("Bulunan (sınıf, SNR) → indeks:", found)"""
    )
)

cells.append(
    code(
        """order_keys = [k for k in targets if k in found]
if len(order_keys) < len(targets):
    print("Eksik SNR seviyeleri — veri setinde tam eşleşme yok; mevcut olanlar çiziliyor")
X_bpsk, _, _ = read_rows_at_indices([found[k] for k in order_keys], H5_PATH)
fig, axes = plt.subplots(1, len(order_keys), figsize=(4 * len(order_keys), 3))
if len(order_keys) == 1:
    axes = [axes]
colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]
for idx, (ax, k, color) in enumerate(zip(axes, order_keys, colors)):
    ax.plot(X_bpsk[idx, :, 0], color=color, linewidth=0.9)
    ax.set_title(f"SNR = {k[1]:+d} dB")
    ax.set_ylim(-0.25, 0.25)
fig.suptitle("BPSK — Gürültü etkisi (I kanalı)", y=1.03)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_snr_effect_BPSK.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    code(
        """# 3 mod × 3 SNR (2016'daki grid benzeri)
mods = []
for lab in ("BPSK", "64QAM", "FM"):
    if lab in name_to_idx:
        mods.append(name_to_idx[lab])
snrs = (-10, 0, 18)
pairs = [(m, s) for m in mods for s in snrs]
found_g = find_pair_indices(H5_PATH, set(pairs))
if not mods:
    print("Grid için modülasyon adı yok")
else:
    fig, axes = plt.subplots(len(mods), len(snrs), figsize=(11, 2.6 * len(mods)))
    axes = np.atleast_2d(axes)
    for r, m in enumerate(mods):
        for c, s in enumerate(snrs):
            k = (m, s)
            ax = axes[r, c]
            if k not in found_g:
                ax.set_visible(False)
                continue
            Xrow, _, _ = read_rows_at_indices([found_g[k]], H5_PATH)
            ax.plot(Xrow[0, :, 0], color="#2471a3", linewidth=0.85)
            ax.set_ylim(-0.25, 0.25)
            if r == 0:
                ax.set_title(f"SNR={s:+d} dB", fontsize=9)
            if c == 0:
                ax.set_ylabel(CLASS_NAMES[m], fontsize=10, fontweight="bold")
    fig.suptitle("Modülasyon × SNR — I kanalı", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS / "eda_2018_waveform_grid.png", dpi=150, bbox_inches="tight")
    plt.show()"""
    )
)

cells.append(
    code(
        """mean_p, pcnt, snr_ax_p = streaming_power_mean_by_snr(
    H5_PATH, chunk_rows=CHUNK_XYZ, snr_offset=SNR_OFF
)
plot_mask = pcnt > 0
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(snr_ax_p[plot_mask], mean_p[plot_mask], marker="o", color="steelblue", linewidth=2, markersize=4)
ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="0 dB")
ax.set_title("Ortalama sinyal gücü vs SNR (tüm sınıflar, streaming)")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Ortalama (I²+Q²)")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_power_vs_snr.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(md("""## Adım 4 — Konstelasyon"""))

cells.append(
    code(
        """QUOTA = 200
SNR_CON = 16
print("Sınıf başına örnek toplanıyor (yüksek SNR, kota=%d)..." % QUOTA)
idx_lists = collect_row_indices_per_class_quota(
    QUOTA, H5_PATH, chunk_rows=CHUNK_YZ, n_classes=n_cls, snr_floor=SNR_CON
)
flat_idx = np.concatenate([x for x in idx_lists if x.size > 0])
X_c, y_c, z_c = read_rows_at_indices(flat_idx, H5_PATH)
print("Konstelasyon için X:", X_c.shape)"""
    )
)

cells.append(
    code(
        """def norm_scatter(sig):
    p = np.sqrt(np.mean(sig[:, 0] ** 2 + sig[:, 1] ** 2) + 1e-12)
    return sig[:, 0] / p, sig[:, 1] / p


ncols = 6
nrows = int(np.ceil(n_cls / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
axes = axes.flatten()
for i in range(n_cls):
    ax = axes[i]
    sel = y_c == i
    if not np.any(sel):
        ax.set_visible(False)
        continue
    Xi = X_c[sel]
    iv, qv = [], []
    for j in range(min(50, Xi.shape[0])):  # çok yoğunsa alt örnekle
        a, b = norm_scatter(Xi[j])
        iv.append(a)
        qv.append(b)
    iv = np.concatenate(iv)
    qv = np.concatenate(qv)
    ax.scatter(iv, qv, s=0.15, alpha=0.25, color="steelblue")
    ax.set_title(CLASS_NAMES[i], fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
for j in range(n_cls, len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f"Konstelasyon — tüm sınıflar (SNR≥{SNR_CON}, kota)", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_constellation_all.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    code(
        """psk_names = [n for n in ("BPSK", "QPSK", "8PSK") if n in name_to_idx]
snr_psk = (18, 6, 0)
fig, axes = plt.subplots(len(psk_names), len(snr_psk), figsize=(10, 3 * len(psk_names)))
if len(psk_names) == 1:
    axes = np.array([axes])
pairs_p = [(name_to_idx[n], s) for n in psk_names for s in snr_psk]
fp = find_pair_indices(H5_PATH, set(pairs_p))
for r, n in enumerate(psk_names):
    m = name_to_idx[n]
    for c, s in enumerate(snr_psk):
        ax = axes[r, c]
        k = (m, s)
        if k not in fp:
            ax.set_visible(False)
            continue
        Xr, _, _ = read_rows_at_indices([fp[k]], H5_PATH)
        iv, qv = norm_scatter(Xr[0])
        ax.scatter(iv, qv, s=0.3, alpha=0.4, color="steelblue")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        if r == 0:
            ax.set_title(f"SNR={s:+d} dB", fontsize=9)
        if c == 0:
            ax.set_ylabel(n, fontsize=10, fontweight="bold")
plt.suptitle("PSK — konstelasyon vs SNR", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_constellation_PSK.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    code(
        """qam_names = [n for n in ("16QAM", "64QAM") if n in name_to_idx]
fig, axes = plt.subplots(len(qam_names), len(snr_psk), figsize=(10, 3.2 * len(qam_names)))
if len(qam_names) == 1:
    axes = np.array([axes])
pairs_q = [(name_to_idx[n], s) for n in qam_names for s in snr_psk]
fq = find_pair_indices(H5_PATH, set(pairs_q))
for r, n in enumerate(qam_names):
    m = name_to_idx[n]
    for c, s in enumerate(snr_psk):
        ax = axes[r, c]
        k = (m, s)
        if k not in fq:
            ax.set_visible(False)
            continue
        Xr, _, _ = read_rows_at_indices([fq[k]], H5_PATH)
        iv, qv = norm_scatter(Xr[0])
        ax.scatter(iv, qv, s=0.3, alpha=0.4, color="coral")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        if r == 0:
            ax.set_title(f"SNR={s:+d} dB", fontsize=9)
        if c == 0:
            ax.set_ylabel(n, fontsize=10, fontweight="bold")
plt.suptitle("QAM — konstelasyon vs SNR", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_constellation_QAM.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(
    code(
        """qi = name_to_idx.get("QPSK", 1)
snr_q = (-20, -10, -4, 0, 6, 18)
pairs_q2 = [(qi, s) for s in snr_q]
fq2 = find_pair_indices(H5_PATH, set(pairs_q2))
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()
for ax, s in zip(axes, snr_q):
    k = (qi, s)
    if k not in fq2:
        ax.set_visible(False)
        continue
    Xr, _, _ = read_rows_at_indices([fq2[k]], H5_PATH)
    iv, qv = norm_scatter(Xr[0])
    ax.scatter(iv, qv, s=0.5, alpha=0.45, color="steelblue")
    ax.set_title(f"SNR = {s:+d} dB")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
plt.suptitle("QPSK — SNR azaldıkça konstelasyon", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_constellation_QPSK_snr.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

cells.append(md("""## ADIM-5 — İstatistiksel özet ve eksik veri (streaming)"""))

cells.append(
    code(
        """def sanitize_iq(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


probe_n = min(2048, N)
Xp, _, _ = read_slice(0, probe_n, H5_PATH)
pr = sanitize_iq(Xp)
print(f"Sanitize (ilk {probe_n}): NaN={np.isnan(pr).any()}, Inf={np.isinf(pr).any()}")"""
    )
)

cells.append(
    code(
        """print("Sınıf bazlı I/Q (tam veri, streaming — birkaç dakika sürebilir)...")
mom = streaming_per_class_iq_moments(H5_PATH, chunk_rows=CHUNK_XYZ, n_classes=n_cls)
rows = []
for i in range(n_cls):
    rows.append(
        {
            "modulation": CLASS_NAMES[i],
            "n_scalar": int(mom["n"][i]),
            "I_mean": mom["I_mean"][i],
            "I_std": mom["I_std"][i],
            "I_min": mom["I_min"][i],
            "I_max": mom["I_max"][i],
            "Q_mean": mom["Q_mean"][i],
            "Q_std": mom["Q_std"][i],
            "Q_min": mom["Q_min"][i],
            "Q_max": mom["Q_max"][i],
        }
    )
stats_by_class = pd.DataFrame(rows).set_index("modulation").sort_index()
print(stats_by_class.round(6).to_string())
out_stats = RESULTS / "eda_2018_stats_by_class.csv"
stats_by_class.to_csv(out_stats)
print("CSV:", out_stats)"""
    )
)

cells.append(
    code(
        """# Global özet + seyrek Pearson (2016 ile aynı fikir)
SUB = min(80_000, N)
Xg, yg, zg = read_slice(0, SUB, H5_PATH)
g_i = Xg[:, :, 0].ravel()
g_q = Xg[:, :, 1].ravel()
global_summary = pd.DataFrame(
    [
        {
            "I_mean": g_i.mean(),
            "I_std": g_i.std(),
            "Q_mean": g_q.mean(),
            "Q_std": g_q.std(),
            "I_Q_pearson_flat_sample": np.corrcoef(g_i[::500], g_q[::500])[0, 1],
        }
    ]
)
print(global_summary.round(6).to_string(index=False))"""
    )
)

cells.append(md("""## ADIM-6 — Öznitelik mühendisliği, korelasyon, PCA (alt küme — RAM)"""))

cells.append(
    code(
        """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SUB_FEAT = min(120_000, N)
Xf, yf, zf = read_slice(0, SUB_FEAT, H5_PATH)
print("Öznitelik alt kümesi:", Xf.shape)"""
    )
)

cells.append(
    code(
        """FEATURE_NAMES = [
    "I_mu", "I_std", "Q_mu", "Q_std",
    "env_mu", "env_std", "zcr_I",
    "spec_low",
]


def handcrafted_features(sig):
    i, q = sig[:, 0].astype(np.float64), sig[:, 1].astype(np.float64)
    env = np.sqrt(i * i + q * q)
    centered = i - i.mean()
    zcr = np.sum(np.diff(np.sign(centered)) != 0) / max(len(i) - 1, 1)
    spec = np.abs(np.fft.rfft(i))
    spec_low = float(spec[: min(32, spec.size)].mean())
    return np.array(
        [i.mean(), i.std(), q.mean(), q.std(), env.mean(), env.std(), zcr, spec_low],
        dtype=np.float64,
    )


rng = np.random.default_rng(42)
N_FEAT = min(12_000, Xf.shape[0])
pick = rng.choice(Xf.shape[0], size=N_FEAT, replace=False)
F = np.stack([handcrafted_features(Xf[i]) for i in pick])
y_sub = yf[pick]
feat_df = pd.DataFrame(F, columns=FEATURE_NAMES)
print(feat_df.describe().T.round(6).to_string())"""
    )
)

cells.append(
    code(
        """corr = feat_df[FEATURE_NAMES].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
ax.set_title("El yapımı öznitelikler — Pearson (2018 alt örneklem)")
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
corr.to_csv(RESULTS / "eda_2018_feature_correlation_matrix.csv")
print("CSV:", RESULTS / "eda_2018_feature_correlation_matrix.csv")"""
    )
)

cells.append(
    code(
        """scaler = StandardScaler()
Fz = scaler.fit_transform(F)
pca = PCA(n_components=2, random_state=42)
Z2 = pca.fit_transform(Fz)
uniq_cls = np.unique(y_sub.astype(int))
colors = plt.cm.tab20(np.linspace(0, 1, max(uniq_cls.size, 1), endpoint=False))
fig, ax = plt.subplots(figsize=(10, 7))
for k, c in enumerate(uniq_cls):
    m = y_sub == c
    ax.scatter(
        Z2[m, 0],
        Z2[m, 1],
        s=10,
        alpha=0.45,
        c=[colors[k % len(colors)]],
        linewidths=0,
        label=CLASS_NAMES[int(c)],
    )
ax.legend(fontsize=7, ncol=2, loc="best")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
ax.set_title("PCA — el yapımı öznitelikler (2018)")
plt.tight_layout()
fig.savefig(RESULTS / "eda_2018_pca_handcrafted.png", dpi=150, bbox_inches="tight")
plt.show()
print("PCA varyans oranı:", pca.explained_variance_ratio_.round(4))"""
    )
)

cells.append(
    md(
        """## Bitiş

- Karşılaştırma: `03_eda_compare_2016_2018.ipynb`
- Çok yavaşsa: `CHUNK_XYZ` artırılabilir veya `streaming_per_class_iq_moments` yalnızca `read_slice` alt kümesinde çalıştırılabilir (raporda belirt)."""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.6"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", NB_OUT)
