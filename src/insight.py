from __future__ import annotations

import pandas as pd

SEGMENT_TEMPLATES = {
    "Champions": {
        "deskripsi": "Pelanggan terbaik Anda — sering belanja, transaksi besar, dan baru saja membeli.",
        "rekomendasi": [
            "Berikan program loyalitas eksklusif (VIP membership, akses awal produk baru).",
            "Minta testimoni atau referral — mereka paling mungkin merekomendasikan toko Anda.",
            "Jangan beri diskon besar; fokus pada appreciation (gift, kartu ucapan).",
        ],
    },
    "Loyal": {
        "deskripsi": "Pelanggan setia dengan frekuensi pembelian tinggi.",
        "rekomendasi": [
            "Tawarkan bundling atau upsell produk komplementer.",
            "Kirim survey kepuasan untuk menjaga retensi.",
            "Berikan reward poin atau cashback berjenjang.",
        ],
    },
    "Potential": {
        "deskripsi": "Pelanggan dengan potensi tumbuh — pembelian belum sering tapi cukup recent.",
        "rekomendasi": [
            "Kirim rekomendasi produk berdasarkan riwayat pembelian.",
            "Tawarkan diskon kecil untuk mendorong pembelian kedua/ketiga.",
            "Edukasi via konten (newsletter, tips penggunaan produk).",
        ],
    },
    "Need Attention": {
        "deskripsi": "Pelanggan yang mulai jarang belanja — perlu pendekatan ulang.",
        "rekomendasi": [
            "Kirim email re-engagement dengan penawaran personal.",
            "Tampilkan produk baru yang relevan dengan kategori favorit mereka.",
            "Tanya feedback: apa yang membuat mereka berkurang belanja?",
        ],
    },
    "At Risk": {
        "deskripsi": "Pelanggan yang sudah lama tidak transaksi — risiko churn tinggi.",
        "rekomendasi": [
            "Berikan diskon agresif atau voucher comeback bernilai besar.",
            "Kirim pesan personal (WA/email) menanyakan kabar dan menawarkan promo.",
            "Tampilkan produk best-seller terbaru untuk menarik kembali.",
        ],
    },
    "Regular": {
        "deskripsi": "Pelanggan dengan profil pembelian rata-rata.",
        "rekomendasi": [
            "Lakukan A/B testing pada penawaran promo standar.",
            "Pantau perubahan perilaku untuk mendeteksi pergeseran segmen.",
        ],
    },
}


def generate_insight(
    segment: str,
    rfm_stats: dict | None = None,
    top_products: list[str] | None = None,
) -> dict:
    template = SEGMENT_TEMPLATES.get(
        segment,
        {
            "deskripsi": f"Segmen {segment} — perlu analisis lebih lanjut.",
            "rekomendasi": ["Lakukan eksplorasi data lanjutan untuk segmen ini."],
        },
    )

    insight = {
        "segment": segment,
        "deskripsi": template["deskripsi"],
        "rekomendasi": list(template["rekomendasi"]),
    }

    if rfm_stats:
        insight["statistik"] = (
            f"Rata-rata Recency: {rfm_stats.get('Recency', 0):.0f} hari, "
            f"Frequency: {rfm_stats.get('Frequency', 0):.1f}x, "
            f"Monetary: Rp{rfm_stats.get('Monetary', 0):,.0f}"
        )

    if top_products:
        insight["produk_unggulan"] = top_products

    return insight


def generate_all_insights(rfm_labeled: pd.DataFrame, top_products_df: pd.DataFrame) -> list[dict]:
    results = []
    for segment in rfm_labeled["Segment"].unique():
        seg_df = rfm_labeled[rfm_labeled["Segment"] == segment]
        stats = {
            "Recency": seg_df["Recency"].mean(),
            "Frequency": seg_df["Frequency"].mean(),
            "Monetary": seg_df["Monetary"].mean(),
        }
        top = (
            top_products_df[top_products_df["Segment"] == segment]["Description"]
            .head(3)
            .tolist()
        )
        results.append(generate_insight(segment, stats, top))
    return results
