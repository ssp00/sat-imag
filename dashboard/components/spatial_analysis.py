"""
Tab 4 — Spatial Change Visualization
Overlay, opacity slider, change %, pixel counts.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from dashboard.utils.data_loader import (
    get_test_samples, load_sample_images, load_prediction,
    load_dashboard_data, MODEL_KEYS, MODEL_LABELS,
)
from dashboard.utils.viz_helpers import overlay_mask_on_image


def render():
    st.title("🗺️ Spatial Change Visualization")
    st.markdown(
        "Explore **where** changes occur in the satellite imagery. "
        "Use the opacity slider to blend prediction masks over the original image."
    )

    samples = get_test_samples()
    df      = load_dashboard_data()

    # ── Controls ───────────────────────────────────────────────────────────────
    col_s, col_m, col_alpha = st.columns([2, 2, 2])
    with col_s:
        sample = st.selectbox("📂 Test Sample", samples,
                               format_func=lambda s: s.replace(".png",""), key="spatial_sample")
    with col_m:
        model_key = st.selectbox("🤖 Model", MODEL_KEYS,
                                  format_func=lambda k: MODEL_LABELS[k], key="spatial_model")
    with col_alpha:
        alpha = st.slider("🔆 Overlay Opacity", min_value=0.0, max_value=1.0,
                           value=0.4, step=0.05)

    imgs     = load_sample_images(sample)
    pred_rgb = load_prediction(model_key, sample)
    pred_raw = pred_rgb[:, :, 0]   # single-channel
    gt_raw   = imgs["gt_raw"]

    # ── Overlays ───────────────────────────────────────────────────────────────
    gt_overlay   = overlay_mask_on_image(imgs["t2"], gt_raw,   color=(0, 220, 0),   alpha=alpha)
    pred_overlay = overlay_mask_on_image(imgs["t2"], pred_raw, color=(220, 100, 0), alpha=alpha)

    st.markdown("---")

    # ── Row 1: original + GT overlay + pred overlay ────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(imgs["t2"],   caption="🕑 T2 (After)", width="stretch")
    with c2:
        st.image(gt_overlay,   caption="🟢 GT Overlay on T2",  width="stretch")
    with c3:
        st.image(pred_overlay, caption=f"🟠 {MODEL_LABELS[model_key]} Overlay on T2",
                 width="stretch")

    st.markdown("---")

    # ── Row 2: binary masks ────────────────────────────────────────────────────
    c4, c5 = st.columns(2)
    with c4:
        st.image(imgs["gt"],   caption="✅ Ground Truth Mask", width="stretch")
    with c5:
        st.image(pred_rgb,     caption=f"🤖 Predicted Mask — {MODEL_LABELS[model_key]}",
                 width="stretch")

    st.markdown("---")

    # ── Statistics ─────────────────────────────────────────────────────────────
    st.subheader("📐 Spatial Statistics")
    row = df[(df["image_name"] == sample) & (df["model"] == MODEL_LABELS[model_key])]

    total_pixels = gt_raw.size

    gt_pix   = int(np.sum(gt_raw > 127))
    pred_pix = int(np.sum(pred_raw > 127))
    gt_pct   = gt_pix   / total_pixels * 100
    pred_pct = pred_pix / total_pixels * 100

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("GT Changed Pixels",   f"{gt_pix:,}", f"{gt_pct:.2f}%")
    s2.metric("Pred Changed Pixels", f"{pred_pix:,}", f"{pred_pct:.2f}%")
    s3.metric("Area Δ (pred − GT)",  f"{pred_pix - gt_pix:+,}",
              f"{pred_pct - gt_pct:+.2f}%")
    s4.metric("Total Pixels",        f"{total_pixels:,}")

    # ── Change area bar comparison ─────────────────────────────────────────────
    fig = go.Figure([
        go.Bar(name="Ground Truth",  x=["Change Area (%)"], y=[round(gt_pct, 3)],
               marker_color="#55A868"),
        go.Bar(name=MODEL_LABELS[model_key], x=["Change Area (%)"], y=[round(pred_pct, 3)],
               marker_color="#C44E52"),
    ])
    fig.update_layout(
        barmode="group",
        title="Ground Truth vs Predicted Change Area",
        yaxis_title="% of Image",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig, width="stretch",
                     key=f"spatial_area_bar_{sample}_{model_key}")

    # ── All-model change area chart ────────────────────────────────────────────
    st.subheader("📊 All Models — Change Area for This Sample")
    sample_df = df[df["image_name"] == sample].copy()
    fig2 = go.Figure()
    # GT bar (same for all rows)
    fig2.add_trace(go.Bar(
        name="Ground Truth",
        x=sample_df["model"],
        y=[gt_pct] * len(sample_df),
        marker_color="#55A868",
    ))
    fig2.add_trace(go.Bar(
        name="Predicted",
        x=sample_df["model"],
        y=sample_df["predicted_change_percentage"],
        marker_color="#C44E52",
    ))
    fig2.update_layout(
        barmode="group",
        title="Predicted vs GT Change Area % — All Models",
        yaxis_title="% of Image",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig2, width="stretch", key=f"spatial_all_models_bar_{sample}")
