# 🎧 Spotify Insight — Dashboard Streamlit

Prototype dashboard analisis Spotify Tracks Dataset (114.000 lagu) untuk mata kuliah Data Mining dan Visualisasi.

## Cara menjalankan

```bash
pip install -r requirements.txt
python prepare.py              # sekali saja, ±4 menit: menghitung model & menyimpan ke artifacts/
streamlit run Beranda.py
```

Data dibaca dari `../data/dataset.csv`. Folder `artifacts/` sudah disertakan, jadi `prepare.py` hanya perlu dijalankan ulang jika data atau parameter diubah.

## Halaman

| Halaman | Metode | Fitur interaktif |
|---|---|---|
| 🏠 Beranda | KPI, treemap genre | filter genre / popularity / explicit, putar lagu #1 |
| 🧹 Kualitas Data | missing, duplikat, nilai tidak valid, imputasi mean/median/KNN, IQR | pilih metode imputasi, boxplot outlier per genre |
| 🎧 Eksplorasi Audio | korelasi, peta mood energy×valence, radar genre, violin, uji Mann-Whitney | scatter bebas (pilih sumbu & warna) |
| 🌌 Clustering Mood | K-Means, K-Prototype, Fuzzy C-Means, UMAP, parallel coordinates | slider k, galaksi lagu, sorot lagu/artis |
| 🧬 Peta Genre & Artis | hierarchical (Ward), jaringan artis + Louvain | genre bersaudara, fokus artis |
| 🔮 Prediksi Hit | Random Forest, ROC, SHAP | ambang probabilitas, jelaskan satu lagu, "coba sendiri", klasifikasi genre pilihan |
| 🔗 Pola Asosiasi | Apriori | filter support/confidence/lift & konsekuen |
| 🚨 Deteksi Anomali | Isolation Forest per genre | pilih genre & sumbu, dengarkan lagu teraneh |
| 🎯 Rekomendasi Lagu | cosine similarity (NearestNeighbors) | cari lagu, filter genre, Spotify player, radar dua lagu |
| 💡 Insight | ringkasan temuan | – |

## Struktur

```
app/
├── Beranda.py          # halaman utama
├── pages/              # 9 halaman analisis
├── utils.py            # tema, loader (cache), filter, komponen
├── prepare.py          # hitung semua model sekali → artifacts/
├── artifacts/          # hasil prepare.py
└── .streamlit/config.toml  # tema gelap ala Spotify
```
