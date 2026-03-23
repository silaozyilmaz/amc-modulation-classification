# Ham Veri (Raw Data)

Bu klasör ham veri setlerini içerir. Bu dosyalar `.gitignore` ile hariç tutulur (büyük boyut).

---

## RadioML 2016.10a

| Özellik | Değer |
|---------|-------|
| **Dosya** | `RML2016.10a_dict.pkl` veya `RML2016.10a_dict.dat` |
| **Format** | Python pickle (dict) |
| **Key** | `(modulation, snr)` tuple |
| **Value** | NumPy array `(1000, 2, 128)` — 1000 örnek, 2 kanal (I/Q), 128 zaman adımı |
| **Modülasyon sayısı** | 11 |
| **SNR aralığı** | -20 dB ile +18 dB (2 dB adımlarla) |

**Kaynak:** [DeepSig](https://www.deepsig.ai/datasets), [Zenodo](https://zenodo.org/records/18397070)

---

## RadioML 2018.01A (opsiyonel)

İnceleme veya karşılaştırma için. HDF5 formatında, ~21 GB — tam yükleme yapılmaz, chunk/lazy okuma gerekir.
