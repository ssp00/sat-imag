"""
Tab 9 — Sample-Level Model Comparison
T1 | T2 | GT | all 4 model predictions in one row per sample.
"""

import streamlit as st

from dashboard.utils.data_loader import (
    get_test_samples, load_sample_images, load_prediction,
    load_dashboard_data, MODEL_KEYS, MODEL_LABELS,
)
from dashboard.utils.viz_helpers import make_error_map, MODEL_COLORS
import plotly.graph_objects as go


def render():
    st.title("🔀 Sample-Level Model Comparison")
    st.markdown(
        "View **all four model predictions side by side** for a single test sample, "
        "including T1, T2 and Ground Truth. Select a sample to begin."
    )

    samples = get_test_samples()
    df      = load_dashboard_data()

    # ── Sample selector ────────────────────────────────────────────────────────
    sample = st.selectbox("📂 Test Sample", samples,
                           format_func=lambda s: s.replace(".png",""), key="samplecmp_sample")

    imgs = load_sample_images(sample)

    # ── Header row: T1, T2, GT ────────────────────────────────────────────────
    st.subheader("🛰️ Input Images & Ground Truth")
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        st.image(imgs["t1"], caption="🕐 T1 (Before)", width="stretch")
    with hc2:
        st.image(imgs["t2"], caption="🕑 T2 (After)",  width="stretch")
    with hc3:
        st.image(imgs["gt"], caption="✅ Ground Truth", width="stretch")

    st.markdown("---")

    # ── Predictions row ────────────────────────────────────────────────────────
    st.subheader("🤖 Model Predictions")
    p_cols = st.columns(4)
    for i, mk in enumerate(MODEL_KEYS):
        pred = load_prediction(mk, sample)
        with p_cols[i]:
            st.image(pred, caption=MODEL_LABELS[mk], width="stretch")

    st.markdown("---")

    # ── Error maps row ─────────────────────────────────────────────────────────
    st.subheader("🗺️ Error Maps (TP/TN/FP/FN)")
    st.caption("🟢 TP · 🔴 FP · 🔵 FN · ⬛ TN")
    e_cols = st.columns(4)
    gt_raw = imgs["gt_raw"]
    for i, mk in enumerate(MODEL_KEYS):
        pred_rgb = load_prediction(mk, sample)
        pred_raw = pred_rgb[:, :, 0]
        emap     = make_error_map(pred_raw, gt_raw)
        with e_cols[i]:
            st.image(emap, caption=f"{MODEL_LABELS[mk]} Error", width="stretch")

    st.markdown("---")

    # ── Metric comparison for this sample ─────────────────────────────────────
    st.subheader("📊 Metric Comparison — This Sample")
    sample_df = df[df["image_name"] == sample][
        ["model", "accuracy", "precision", "recall", "f1", "iou", "dice",
         "true_change_percentage", "predicted_change_percentage"]
    ].copy()
    for c in ["accuracy","precision","recall","f1","iou","dice",
              "true_change_percentage","predicted_change_percentage"]:
        if c in sample_df.columns:
            sample_df[c] = sample_df[c].round(4)
    st.dataframe(sample_df.reset_index(drop=True), width="stretch", hide_index=True)

    # Radar for this sample only
    fig_radar = go.Figure()
    cats       = ["accuracy","precision","recall","f1","iou","dice"]
    cat_labels = ["Accuracy","Precision","Recall","F1","IoU","Dice"]
    for _, row in sample_df.iterrows():
        vals = [row[c] for c in cats]
        vals_closed = vals + [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cat_labels + [cat_labels[0]],
            name=row["model"],
            fill="toself",
            line=dict(color=MODEL_COLORS.get(row["model"])),
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title=f"Per-Model Radar — {sample.replace('.png','')}",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig_radar, width="stretch", key=f"sample_radar_{sample}")

    st.markdown("---")

    # ── Browse all samples ─────────────────────────────────────────────────────
    st.subheader("🖼️ Browse All Samples")
    show_all = st.checkbox("Show all samples (4-model grid)")
    if show_all:
        for s in samples:
            st.markdown(f"#### {s.replace('.png','')}")
            imgs_s = load_sample_images(s)
            # T1 T2 GT + 4 preds = 7 columns
            all_cols = st.columns(7)
            with all_cols[0]:
                st.image(imgs_s["t1"], caption="T1", width="stretch")
            with all_cols[1]:
                st.image(imgs_s["t2"], caption="T2", width="stretch")
            with all_cols[2]:
                st.image(imgs_s["gt"], caption="GT", width="stretch")
            for j, mk in enumerate(MODEL_KEYS):
                pred_s = load_prediction(mk, s)
                with all_cols[3+j]:
                    st.image(pred_s, caption=MODEL_LABELS[mk][:10],
                             width="stretch")
            st.markdown("---")
