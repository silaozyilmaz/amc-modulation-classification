# Versiyonlama Stratejisi

Bu projede ilerleme **semantic versioning** ve **tag** ile takip edilir.

## Format

- **v0.X.0** = Ana milestone (yeni özellik/faz tamamlandı)
- Her an `git tag -l` ile mevcut versiyonlar görülebilir

## Versiyon Geçmişi

| Tag | Açıklama |
|-----|----------|
| v0.1.0 | Proje yapısı, requirements.txt, .gitignore, src modülleri |

## Gelecek Planlanan Versiyonlar

| Tag | Açıklama |
|-----|----------|
| v0.2.0 | RadioML 2016 veri yükleme utilities |
| v0.3.0 | EDA notebook |
| v0.4.0 | Baseline model |
| v0.5.0 | Model karşılaştırması |
| v0.6.0 | Derin öğrenme modeli |
| v0.7.0 | Değerlendirme metrikleri |
| v0.8.0 | En iyi model optimizasyonu |
| v0.9.0 | Streamlit demo |
| v1.0.0 | Final: Dokümantasyon ve sunum |

## Kullanım

```bash
# Mevcut tag'leri listele
git tag -l

# Belirli bir versiyona geri dön (sadece bakmak için)
git checkout v0.1.0

# Tag ile birlikte push
git push origin main --tags
```
