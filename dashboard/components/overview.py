"""
Tab 1 — Project Overview
KPI cards, dataset info, best model highlight, overall performance chart.
"""

import streamlit as st
import plotly.graph_objects as go
from dashboard.utils.data_loader import (
    load_dashboard_data, load_model_metrics, get_test_samples,
    load_experiment_config, MODEL_LABELS,
)
from dashboard.utils.viz_helpers import radar_chart, bar_chart_metric, MODEL_COLORS


def kpi_card(col, label: str, value: str, delta: str = "", color="#4C72B0"):
    col.markdown(
        f"""
        <div style="background:{color}18;border-left:4px solid {color};
                    border-radius:8px;padding:16px 20px;margin-bottom:8px;">
          <p style="margin:0;font-size:13px;color:#888;">{label}</p>
          <p style="margin:4px 0 0;font-size:26px;font-weight:700;">{value}</p>
          <p style="margin:0;font-size:12px;color:#aaa;">{delta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.title("🛰️ LEVIR-CD Change Detection Dashboard")
    st.caption(
        "Information Visualization Project — Satellite Imagery Change Detection "
        "using CNN & Transformer Models"
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    df      = load_dashboard_data()
    metrics = load_model_metrics()
    config  = load_experiment_config()
    samples = get_test_samples()

    # Normalise model column name
    model_col = "model" if "model" in metrics.columns else metrics.columns[0]

    # Best model by F1
    best_row   = metrics.loc[metrics["f1"].idxmax()]
    best_model = best_row[model_col]
    best_f1    = best_row["f1"]

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.subheader("📊 At a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Models Compared", "4", "CNN → Transformer progression", "#4C72B0")
    kpi_card(c2, "Test Samples",    str(len(samples)), "LEVIR-CD test split", "#DD8452")
    kpi_card(c3, "Image Resolution","224 × 224 px", "Resized for training", "#55A868")
    kpi_card(c4, "Best Model",      best_model, f"F1 = {best_f1:.4f}", "#C44E52")
    kpi_card(c5, "Dataset",         "LEVIR-CD", "Bi-temporal satellite pairs", "#8172B2")

    st.markdown("---")

    # ── Dataset description ────────────────────────────────────────────────────
    st.subheader("📁 Dataset — LEVIR-CD")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        **LEVIR-CD** (Large-scale Remote Sensing Change Detection Dataset) is a
        publicly available bi-temporal satellite imagery dataset designed for
        building change detection.

        | Property | Value |
        |---|---|
        | Image pairs | ~637 (train+val+test) |
        | Split used | Subsampled: 20 train / 5 val / 5 test |
        | Image size | 1024×1024 (resized to 224×224) |
        | Resolution | ~0.5 m/pixel |
        | Change type | Building construction / demolition |
        | Labels | Binary change mask (0 = no-change, 1 = change) |
        """)
    with col_b:
        st.markdown("""
        **Model Progression (CNN → Transformer)**

        | # | Model | Type |
        |---|---|---|
        | 1 | Siamese CNN | Pure CNN baseline |
        | 2 | Siamese U-Net | CNN + skip connections |
        | 3 | Swin Transformer | Shifted-window attention |
        | 4 | ChangeFormer | Hierarchical transformer |
        """)

    st.markdown("---")

    # ── Overall performance charts ─────────────────────────────────────────────
    st.subheader("📈 Overall Model Performance")
    tab_radar, tab_bar = st.tabs(["Radar Chart", "Bar Charts"])

    with tab_radar:
        st.plotly_chart(radar_chart(metrics), width="stretch", key="overview_radar")
        st.caption(
            "Radar chart shows all six metrics simultaneously. "
            "A larger filled area indicates stronger overall performance."
        )

    with tab_bar:
        cols = st.columns(3)
        for idx, m in enumerate(["f1", "iou", "dice"]):
            with cols[idx]:
                st.plotly_chart(bar_chart_metric(metrics, m), width="stretch",
                                 key=f"overview_bar_{m}")

    st.markdown("---")

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.subheader("📋 Aggregate Metrics Table")
    display = metrics.copy()
    for c in ["accuracy", "precision", "recall", "f1", "iou", "dice"]:
        if c in display.columns:
            display[c] = display[c].round(4)
    st.dataframe(display, width="stretch", hide_index=True)

    # ── Training times ─────────────────────────────────────────────────────────
    time_key_map = {
        "siamese_cnn" : "baseline_training_time_sec",
        "siamese_unet": "siamese_unet_training_time_sec",
        "swin"        : "swin_training_time_sec",
        "changeformer": "changeformer_training_time_sec",
    }
    times = {k: config[v] for k, v in time_key_map.items() if config and v in config}
    if times:
        st.markdown("---")
        st.subheader("⏱️ Training Times")
        t_cols = st.columns(len(times))
        palette = list(MODEL_COLORS.values())
        for i, (k, v) in enumerate(times.items()):
            label = MODEL_LABELS.get(k, k)
            kpi_card(t_cols[i], f"{label}", f"{v:.0f}s", f"≈ {v/60:.1f} min", palette[i % len(palette)])

    st.markdown("---")
    st.caption("Built with Streamlit · LEVIR-CD Dataset · InfoVis Project 2025")
