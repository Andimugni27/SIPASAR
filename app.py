from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.classification import predict_segment, train_random_forest
from src.clustering import RFM_FEATURES, find_optimal_k, label_segments, scale_rfm, train_kmeans
from src.insight import generate_all_insights
from src.preprocessing import REQUIRED_COLUMNS, clean_transactions
from src.rfm import build_rfm
from src.top_product import top_products_per_segment

st.set_page_config(
    page_title="SIPASAR — Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
)

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_data.csv"


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame, k: int):
    df_clean = clean_transactions(df_raw)
    rfm = build_rfm(df_clean)
    X, scaler = scale_rfm(rfm)
    kmeans = train_kmeans(X, k)
    rfm["Cluster"] = kmeans.labels_
    rfm_labeled, mapping = label_segments(rfm)
    top_prod = top_products_per_segment(df_clean, rfm_labeled)
    rf_model, rf_report = train_random_forest(rfm_labeled)
    insights = generate_all_insights(rfm_labeled, top_prod)
    return {
        "df_clean": df_clean,
        "rfm": rfm_labeled,
        "top_prod": top_prod,
        "rf_model": rf_model,
        "rf_report": rf_report,
        "scaler": scaler,
        "kmeans": kmeans,
        "mapping": mapping,
        "insights": insights,
    }


def sidebar() -> tuple[pd.DataFrame | None, int]:
    st.sidebar.title("📊 SIPASAR")
    st.sidebar.caption("AI-Powered Customer Segmentation")
    st.sidebar.markdown("---")

    st.sidebar.subheader("1. Upload Data")
    uploaded = st.sidebar.file_uploader(
        "Pilih CSV transaksi",
        type=["csv"],
        help=f"Kolom wajib: {', '.join(REQUIRED_COLUMNS)}",
    )

    use_sample = st.sidebar.checkbox("Pakai sample data", value=uploaded is None)

    df = None
    if uploaded is not None and not use_sample:
        df = load_csv(uploaded)
    elif use_sample and SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Konfigurasi Model")
    k = st.sidebar.slider("Jumlah cluster (K)", 2, 8, 4)

    st.sidebar.markdown("---")
    st.sidebar.caption("Tim PJK-GM091 · Pijak × IBM SkillsBuild")
    return df, k


def page_overview(result):
    df = result["df_clean"]
    rfm = result["rfm"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{df['InvoiceNo'].nunique():,}")
    c2.metric("Total Pelanggan", f"{rfm['CustomerID'].nunique():,}")
    c3.metric("Total Revenue", f"Rp{df['TotalPrice'].sum():,.0f}")
    c4.metric("Jumlah Segmen", rfm["Segment"].nunique())

    st.markdown("### Distribusi Segmen Pelanggan")
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Jumlah"]
    fig = px.bar(seg_counts, x="Segment", y="Jumlah", color="Segment", text="Jumlah")
    st.plotly_chart(fig, use_container_width=True)


def page_segmentation(result):
    rfm = result["rfm"]

    st.markdown("### RFM 3D Scatter per Segmen")
    fig = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment", size="Monetary", size_max=18, opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Segmentasi")
    st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)

    csv = rfm.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download hasil segmentasi (CSV)",
        data=csv, file_name="segmentasi_pelanggan.csv", mime="text/csv",
    )


def page_top_products(result):
    top = result["top_prod"]
    st.markdown("### Top Produk per Segmen")
    for seg in top["Segment"].unique():
        with st.expander(f"🛒 {seg}", expanded=False):
            seg_df = top[top["Segment"] == seg].drop(columns=["Segment"])
            st.dataframe(seg_df, use_container_width=True, hide_index=True)


def page_insight(result):
    st.markdown("### Insight & Rekomendasi per Segmen")
    for ins in result["insights"]:
        with st.container(border=True):
            st.subheader(f"🎯 {ins['segment']}")
            st.write(ins["deskripsi"])
            if "statistik" in ins:
                st.caption(ins["statistik"])
            st.markdown("**Rekomendasi:**")
            for r in ins["rekomendasi"]:
                st.markdown(f"- {r}")
            if ins.get("produk_unggulan"):
                st.markdown("**Produk unggulan:** " + ", ".join(ins["produk_unggulan"]))


def page_predict(result):
    st.markdown("### Prediksi Segmen Pelanggan Baru")
    st.caption("Masukkan nilai RFM untuk memprediksi segmen pelanggan.")

    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (hari)", min_value=0, value=30)
    frequency = c2.number_input("Frequency (transaksi)", min_value=1, value=5)
    monetary = c3.number_input("Monetary (Rp)", min_value=0.0, value=500000.0, step=10000.0)

    if st.button("Prediksi Segmen", type="primary"):
        seg = predict_segment(result["rf_model"], recency, frequency, monetary)
        st.success(f"Pelanggan diprediksi masuk segmen: **{seg}**")

    with st.expander("Performa Model Random Forest"):
        report_df = pd.DataFrame(result["rf_report"]).T
        st.dataframe(report_df, use_container_width=True)


def page_diagnostics(df_raw: pd.DataFrame, k: int):
    st.markdown("### Elbow & Silhouette Score")
    st.caption("Bantu memilih nilai K optimal.")

    df_clean = clean_transactions(df_raw)
    rfm = build_rfm(df_clean)
    X, _ = scale_rfm(rfm)
    scores = find_optimal_k(X, k_range=range(2, 9))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(scores, x="k", y="inertia", markers=True, title="Elbow Method")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(scores, x="k", y="silhouette", markers=True, title="Silhouette Score")
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="Target ≥ 0.50")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(scores, use_container_width=True, hide_index=True)


def main():
    df_raw, k = sidebar()

    st.title("📊 SIPASAR Dashboard")
    st.caption("AI-Powered Customer Segmentation & Market Insight — PJK-GM091")

    if df_raw is None or df_raw.empty:
        st.info("Upload CSV transaksi atau aktifkan 'Pakai sample data' di sidebar untuk mulai.")
        st.markdown(f"**Kolom yang diharapkan:** `{', '.join(REQUIRED_COLUMNS)}`")
        return

    try:
        result = run_pipeline(df_raw, k)
    except ValueError as e:
        st.error(f"Validasi data gagal: {e}")
        return
    except Exception as e:
        st.exception(e)
        return

    tabs = st.tabs([
        "🏠 Ringkasan", "🎯 Segmentasi", "🛒 Top Produk",
        "💡 Insight", "🔮 Prediksi", "🔬 Diagnostik",
    ])
    with tabs[0]:
        page_overview(result)
    with tabs[1]:
        page_segmentation(result)
    with tabs[2]:
        page_top_products(result)
    with tabs[3]:
        page_insight(result)
    with tabs[4]:
        page_predict(result)
    with tabs[5]:
        page_diagnostics(df_raw, k)


if __name__ == "__main__":
    main()
