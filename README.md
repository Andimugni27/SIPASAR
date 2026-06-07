# SIPASAR — AI-Powered Customer Segmentation & Market Insight Dashboard

**Tim:** PJK-GM091 (Pijak x IBM SkillsBuild)
**Tema:** AI for Business Intelligence and Market Insights

Web app open-source untuk UMKM & retail Indonesia. Upload CSV transaksi → otomatis dapat segmentasi pelanggan (RFM + K-Means), klasifikasi segmen baru (Random Forest), top product per segmen, dan insight rule-based dalam Bahasa Indonesia.

## Fitur

- Upload CSV transaksi (kolom: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`)
- Feature engineering RFM otomatis
- Segmentasi K-Means (Elbow + Silhouette Score)
- Klasifikasi pelanggan baru via Random Forest
- Top produk per segmen
- Insight bisnis Bahasa Indonesia (rule-based)
- Download hasil segmentasi (CSV)

## Quick Start

```bash
# 1. Buat virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan Streamlit
streamlit run app.py
```

Buka browser ke `http://localhost:8501`.

## Struktur Proyek

```
capstone2026/
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   └── sample_data.csv       # Dummy data untuk testing
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning
│   ├── rfm.py                # Feature engineering RFM
│   ├── clustering.py         # K-Means + Elbow + Silhouette
│   ├── classification.py     # Random Forest
│   ├── top_product.py        # Ranking produk per segmen
│   └── insight.py            # Rule-based insight Bahasa Indonesia
├── models/                   # Output .pkl (gitignored)
└── notebooks/
    └── 01_eda.ipynb          # EDA notebook
```

## Dataset

Default pakai `data/sample_data.csv` (dummy 100 baris). Untuk dataset asli:

1. Download Online Retail UCI: <https://archive.ics.uci.edu/ml/datasets/Online+Retail>
2. Simpan sebagai `data/online_retail.csv`

## Tim

| Role | Anggota |
|------|---------|
| Data Preparation & Documentation Lead | Nida Nurapipah |
| EDA & Business Analytics Lead | Rifki Saputra |
| Clustering Model Lead | Nazly Rafa Oktafian Nuzqu |
| Classification & Insight Generator Lead | Muhammad Yusuf |
| Dashboard & Deployment Lead | Andi Muchamad Mugni Pabilla |

## Lisensi

MIT
