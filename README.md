# Gürültülü I/Q Sinyallerinden RF Modülasyon Türlerinin Derin Öğrenme ile Otomatik Sınıflandırılması (AMC)

**English:** Deep Learning-Based Automatic Modulation Classification Using Noisy I/Q Signal Data

Sakarya Uygulamalı Bilimler Üniversitesi — Elektrik-Elektronik Mühendisliği Bölümü  
**Yapay Zekaya Giriş** dersi dönem projesi.

---

## Proje Özeti

RF (radyo frekansı) I/Q sinyal verilerinden modülasyon türlerini otomatik sınıflandıran bir sistem. RadioML 2016.10a ve RadioML 2018.01A veri setleri kullanılarak EDA, baseline model, derin öğrenme modelleri ve Streamlit demo geliştirilecektir.

---

## Proje Yapısı

```
├── data/
│   ├── raw/           # Ham veri (RML2016.10a_dict.dat vb.)
│   └── processed/    # İşlenmiş veri
├── notebooks/        # EDA ve deney notebook'ları
├── src/
│   ├── data/         # Veri yükleme
│   ├── features/     # Özellik çıkarma
│   ├── models/       # Model tanımları
│   ├── evaluation/   # Metrikler ve değerlendirme
│   └── utils/        # Yardımcı fonksiyonlar
├── app/              # Streamlit demo
├── docs/             # Dokümantasyon
├── models/           # Kaydedilmiş model ağırlıkları
└── results/          # Grafikler ve sonuçlar
```

---

## Kurulum

### 1. Sanal ortam oluştur (önerilir)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 3. Veri setini indir

RadioML 2016.10a dosyasını indirip `data/raw/radioml_2016/` altına koyun (ör. `RML2016.10a_dict.pkl`).

RadioML 2018 (GOLD_XYZ, ~21 GiB disk) için `data/raw/radioml_2018/GOLD_XYZ_OSC.0001_1024.hdf5` dosyasını aynı şekilde yerel olarak tutun.

- [DeepSig Datasets](https://www.deepsig.ai/datasets)
- [Zenodo](https://zenodo.org/records/18397070)

### Git ve büyük veri setleri

Ham veri repoya eklenmez: `.gitignore` içinde `data/raw/**/*.pkl`, `*.hdf5` vb. **tüm alt klasörler** kapsanır. Repoda yalnızca kod ve küçük metadatalar kalır; veriyi her ortamda ayrı indirmeniz gerekir.

### RadioML 2018 ve düşük RAM

Tam `X` matrisi bellekte ~20 GiB yer tutar; `pickle` ile tek seferde yüklemeyin. Projede:

- `src/data/radioml2018.py`: HDF5’ten **dilimli** okuma (`read_slice`), epoch için **chunk’lı** batch iterator (`iterate_epoch_batches`), isteğe bağlı `train_val_indices`.
- Ortam değişkeni `RADIOML2018_H5` ile HDF5 yolunu değiştirebilirsiniz.
- Daha küçük bir dosya üretmek için (ör. 50k örnek, bir kez çalıştırın):

```bash
python scripts/extract_radioml2018_subset.py --n-samples 50000 \
  --output data/processed/radioml2018_subset_50k.h5
```

Ardından eğitimde `RADIOML2018_H5` veya kodda bu çıktı yolunu kullanın. `iterate_epoch_batches` içinde `chunk_size` ve `batch_size` ile anlık RAM kullanımını sınırlayın (`estimate_ram_for_slice(chunk_size)` kabaca tahmin verir).

---

## Versiyonlama

İlerleme **haftalık milestone** olarak `v0.H.0` tag’leriyle izlenir (**H** = 1…12); CRISP-DM fazları ve final teslim paketi ile eşleme [VERSIONING.md](VERSIONING.md) içindedir.

```bash
git tag -l 'v0.*' | sort -V
git push origin main --tags
```

---

## Lisans

Eğitim amaçlı dönem projesi.
