# Versiyonlama ve Git Tag Stratejisi

Bu doküman, **Yapay Zekaya Giriş** dersi proje yaşam döngüsü (CRISP-DM / MLOps) ve **haftalık takvim** ile uyumludur. Takvimdeki gerçek tarihlerle birebir örtüşme zorunlu değildir.

**Kaynak gerçek:** Versiyon geçmişi için bağlayıcı olan şey **`git log` ve uzak repodaki tag’ler**dir. Bu dosyadaki tablolar, o geçmişin kısa bir özeti ve ders beklentileriyle hizalamak için bir çerçevedir.

---

## Çalışma akışı (push → dokümantasyon)

1. Anlamlı commit’lerle çalışın; hazır olduğunuzda **`main`** (veya kullandığınız dal) üzerine push edin.
2. Bir **milestone**ı işaretlemek istediğinizde (ör. haftalık teslim, büyük bir özellik paketi) ilgili commit’e **annotated tag** atın ve `git push origin main --tags` ile gönderin.
3. Aynı push/tag turunda veya hemen ardından bu dosyada **«Mevcut kayıtlı tag’ler»** tablosuna yeni satır ekleyin: tag adı, kısa not, mümkünse hangi ders haftası / CRISP fazıyla örtüştüğü.
4. Aşağıdaki **Hafta ↔ Tag** tablosu **referanstır**; repoda henüz o kapsam yoksa tabloyu değiştirmeniz gerekmez. Gerçek durum her zaman tag tablosu + commit geçmişindedir.

Böylece önce sabit bir “hedef versiyon planı”na uymaya çalışmak yerine, **ne pushlandıysa dokümantasyon ona göre güncellenir**; ders rubriği için de hangi haftanın hangi tag ile karşılandığını geriye dönük net görebilirsiniz.

---

## Tag formatı

| Bileşen | Anlamı |
|--------|--------|
| **v0.H.0** | **H** = ders takvimindeki hafta (1–12). Bir milestone’ı kilitlemek istediğinizde bu numarayı kullanın; referans tabloyla hizalamak için uygun **H**’yi seçin. |
| **v0.H.P** (isteğe bağlı) | Aynı hafta içinde küçük düzeltmeler: **P** patch (ör. `v0.3.1`). |

İsterseniz anlamlı mesaj için **annotated tag** kullanın:

```bash
git tag -a v0.3.0 -m "Hafta 3: EDA notebook ve görseller"
```

---

## Hafta ↔ Tag ↔ CRISP-DM (referans tablo)

Ders dokümanındaki **6 faz** ile haftalar çakışır; aşağıdaki tablo tipik eşlemeyi gösterir (**hedef şablon**). Gerçek tag sırası ve tarihleri üstteki akışa göre oluşur.

| Hafta | Tipik tag | Ders konusu / teslim (özet) | CRISP-DM fazı |
|------|-------------|-----------------------------|----------------|
| 1 | `v0.1.0` | Grup, konu, repo; problem ifadesi, paydaş/metrik taslağı | 1 — Problem tanımlama |
| 2 | `v0.2.0` | Ham veri, kaynak listesi; veri envanteri | 2 — Veri toplama & etiketleme |
| 3 | `v0.3.0` | Ön işleme, görselleştirme; **EDA raporu (.ipynb)** | 3 — EDA |
| 4 | `v0.4.0` | **Baseline** (scikit-learn) | 4 — Model geliştirme & eğitim |
| 5 | `v0.5.0` | En az **3 algoritma**; karşılaştırma tablosu | 4 — Model geliştirme & eğitim |
| 6 | `v0.6.0` | **DL** (ANN / CNN / RNN) denemesi | 4 — Model geliştirme & eğitim |
| 7 | `v0.7.0` | Metrikler, yorum; **metrik raporu** | 5 — Değerlendirme & yorumlama |
| 8 | `v0.8.0` | Optimizasyon / ince ayar (takvim Hafta 8) | 4–5 |
| 9 | `v0.9.0` | Optimize edilmiş model; transfer learning vb. (Hafta 9) | 4–5 |
| 10 | `v0.10.0` | **Streamlit veya Flask** çalışan demo | 6 — Dağıtım & arayüz |
| 11 | `v0.11.0` | README, teknik doküman, kullanım kılavuzu | 6 + dokümantasyon |
| 12 | `v0.12.0` | Final rapor, sunum, canlı demo için kod kilitlenmesi | Tüm fazlar — kapanış |

İsteğe bağlı: sunum sonrası “resmi özet sürüm” için **`v1.0.0`** (yalnızca `v0.12.0` ile aynı içerik veya küçük düzeltmeler).

---

## Mevcut kayıtlı tag’ler

**Push ve tag attıkça bu tabloyu güncelleyin** (ders tesliminde “versiyon geçmişi özeti” buradan okunabilir).

| Tag | Özet (repoda o anda ne vardı) | Hafta / not (isteğe bağlı) |
|-----|------------------------------|---------------------------|
| `v0.1.0` | Önceki iskelet + `docs/AMC_Faz1_Problem_Tanimlama.pdf` | Hafta 1 — CRISP Faz 1 (problem tanımlama) |
| `v0.2.0` | v0.1.0 + `docs/AMC_Faz2_Veri_Toplama_Raporu.pdf` | Hafta 2 — CRISP Faz 2 (veri toplama raporu) |
| `v0.3.0` | v0.2.0 + EDA notebook’ları (2016 / 2018 / karşılaştırma), RadioML2018 okuma, script’ler, README/VERSIONING güncellemeleri | Hafta 3 — CRISP Faz 3 (EDA) |

---

## GitHub ve ders gereksinimleri (özet)

- En az **10 anlamlı commit**; anlamlı kilitleme noktalarında **`v0.H.0`** (veya patch) tag’i — tag’i push’tan sonra bu dosyadaki tabloya yansıtın.
- Zorunlu klasörler: `data/`, `notebooks/`, `src/`, `api/`, `docs/` — ham veri repoda olmayabilir (`.gitignore` + README’de indirme talimatı).
- `requirements.txt` veya `environment.yml`; kökte **README.md** (kurulum, çalıştırma, proje özeti).

---

## Final teslim paketi ↔ repo içeriği

| Teslim öğesi | Pratik karşılık |
|--------------|-----------------|
| 1. GitHub repo + versiyon geçmişi | `git push origin main --tags` — tag listesi bu dokümandaki şema ile uyumlu olmalı |
| 2. Teknik rapor (≥15 sayfa, IEEE PDF) | `docs/` (ör. `docs/technical_report.pdf`) |
| 3. Çalışan demo (Streamlit / Flask) | `app/` ve/veya `api/` + README’de çalıştırma komutu |
| 4. Sunum (≤15 slayt, canlı demo bölümü) | `docs/` (ör. `presentation.pdf` / `.pptx`) |
| 5. Model dosyası (.pkl / .h5 / .pt) | `models/` — büyük ağırlıklar genelde `.gitignore`’da; **teslimde** Drive/GitHub Release/LFS ile paylaşım, README’de konum/link |
| 6. Kullanım kılavuzu | README + (bonus) video linki README’de |

---

## Komutlar

```bash
# Tüm tag'leri listele
git tag -l 'v0.*' | sort -V

# Belirli bir milestone'a kod olarak bakmak (salt okuma)
git checkout v0.3.0

# Uzak repoya tag'leri gönder
git push origin main --tags

# Annotated tag oluşturma örneği
git tag -a v0.4.0 -m "Hafta 4: sklearn baseline model"
```

---

