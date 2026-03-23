# İşlenmiş Veri (Processed Data)

Bu klasör ham veriden türetilen, model eğitimi için hazırlanmış verileri içerir.

---

## İçerik (planlanan)

| Dosya / Format | Açıklama |
|----------------|----------|
| `X_train.npy`, `X_val.npy`, `X_test.npy` | Train/val/test bölünmüş I/Q dizileri |
| `y_train.npy`, `y_val.npy`, `y_test.npy` | Sınıf etiketleri |
| `metadata.json` | Modülasyon listesi, SNR aralığı, split oranları |

---

## Not

- İşlenmiş dosyalar `.gitignore` ile hariç tutulur.
- Ham veri `data/raw/` içindedir; preprocessing script'leri buradan okuyup bu klasöre yazar.
