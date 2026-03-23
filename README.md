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

RadioML 2016.10a dosyasını indirip `data/raw/` klasörüne koyun:

- [DeepSig Datasets](https://www.deepsig.ai/datasets)
- [Zenodo](https://zenodo.org/records/18397070)

Dosya adı: `RML2016.10a_dict.dat`

---

## Versiyonlama

Proje ilerlemesi **tag** ile takip edilmektedir. Detaylar için [VERSIONING.md](VERSIONING.md) dosyasına bakın.

```bash
git tag -l    # Mevcut versiyonları listele
```

---

## Lisans

Eğitim amaçlı dönem projesi.
