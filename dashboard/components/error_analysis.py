"""
Tab 5 — Error Analysis
Colour-coded error maps (TP/TN/FP/FN), per-sample error breakdown, FP/FN scatter.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from dashboard.utils.data_loader import (
    get_test_samples, load_sample_images, load_prediction,
    load_dashboard_data, MODEL_KEYS, MODEL_LABELS,
)
from dashboard.utils.viz_helpers import make_error_map, MODEL_COLORS


def render():
    st.title("🔬 Error Analysis")
    st.markdown(
        "Understand **where** the model makes mistakes using colour-coded error maps. "
        "🟢 TP · 🔴 FP · 🔵 FN · ⬛ TN"
    )

    samples = get_test_samples()
    df      = load_dashboard_data()

    # ── Controls ───────────────────────────────────────────────────────────────
    col_s, col_m = st.columns(2)
    with col_s:
        sample = st.selectbox("📂 Test Sample", samples,
                               format_func=lambda s: s.replace(".png",""), key="error_sample")
    with col_m:
        model_key = st.selectbox("🤖 Model", MODEL_KEYS,
                                  format_func=lambda k: MODEL_LABELS[k], key="error_model")

    imgs     = load_sample_images(sample)
    pred_rgb = load_prediction(model_key, sample)
    pred_raw = pred_rgb[:, :, 0]
    gt_raw   = imgs["gt_raw"]

    error_map = make_error_map(pred_raw, gt_raw)

    # ── Legend ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;gap:16px;margin-bottom:8px;">
          <span>🟩 <b>Green</b> = True Positive (correctly detected change)</span>
          <span>🟥 <b>Red</b> = False Positive (predicted change, no actual change)</span>
          <span>🟦 <b>Blue</b> = False Negative (missed actual change)</span>
          <span>⬛ <b>Dark</b> = True Negative (correct no-change)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(imgs["t2"],  caption="🕑 T2 (After)", width="stretch")
    with c2:
        st.image(imgs["gt"],  caption="✅ Ground Truth", width="stretch")
    with c3:
        st.image(error_map,   caption=f"🗺️ Error Map — {MODEL_LABELS[model_key]}",
                 width="stretch")

    st.markdown("---")

    # ── Pixel counts ───────────────────────────────────────────────────────────
    row = df[(df["image_name"] == sample) & (df["model"] == MODEL_LABELS[model_key])]
    if not row.empty:
        r = row.iloc[0]
        tp, tn, fp, fn = int(r.TP), int(r.TN), int(r.FP), int(r.FN)

        st.subheader("📊 Pixel-Level Error Breakdown")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ True Positives",  f"{tp:,}", help="Correctly predicted as change")
        m2.metric("✅ True Negatives",  f"{tn:,}", help="Correctly predicted as no-change")
        m3.metric("❌ False Positives", f"{fp:,}", help="Predicted change, actual no-change")
        m4.metric("❌ False Negatives", f"{fn:,}", help="Predicted no-change, actual change")

        # Donut chart
        st.subheader("🍩 Error Distribution")
        fig_donut = go.Figure(go.Pie(
            labels=["True Positive", "True Negative", "False Positive", "False Negative"],
            values=[tp, tn, fp, fn],
            hole=0.5,
            marker=dict(colors=["#00C800", "#1E1E1E", "#DC0000", "#0064DC"]),
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            title=f"Pixel Composition — {MODEL_LABELS[model_key]} on {sample}",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
        )
        st.plotly_chart(fig_donut, width="stretch",
                         key=f"error_donut_{sample}_{model_key}")

    st.markdown("---")

    # ── All models error on this sample ───────────────────────────────────────
    st.subheader("🔄 Error Maps — All Models (Same Sample)")
    cols4 = st.columns(4)
    for i, mk in enumerate(MODEL_KEYS):
        p_rgb = load_prediction(mk, sample)
        p_raw = p_rgb[:, :, 0]
        emap  = make_error_map(p_raw, gt_raw)
        with cols4[i]:
            st.image(emap, caption=MODEL_LABELS[mk], width="stretch")

    st.markdown("---")

    # ── FP / FN bar chart across all samples ──────────────────────────────────
    st.subheader("📊 FP & FN Comparison — All Samples × All Models")
    fig_err = go.Figure()
    for mk in MODEL_KEYS:
        m_df = df[df["model"] == MODEL_LABELS[mk]]
        fig_err.add_trace(go.Bar(
            name=f"{MODEL_LABELS[mk]} — FP",
            x=m_df["image_name"],
            y=m_df["FP"],
            marker_color=MODEL_COLORS[MODEL_LABELS[mk]],
            opacity=0.9,
        ))
        fig_err.add_trace(go.Bar(
            name=f"{MODEL_LABELS[mk]} — FN",
            x=m_df["image_name"],
            y=m_df["FN"],
            marker_color=MODEL_COLORS[MODEL_LABELS[mk]],
            opacity=0.5,
            marker_pattern_shape="/",
        ))
    fig_err.update_layout(
        barmode="group",
        title="False Positives (solid) and False Negatives (hatched) per Sample",
        xaxis_title="Sample",
        yaxis_title="Pixel Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", y=-0.35),
    )
    st.plotly_chart(fig_err, width="stretch", key="error_fp_fn_bar")
