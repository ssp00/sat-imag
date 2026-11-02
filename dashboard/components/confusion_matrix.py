"""
Tab 6 — Confusion Matrix
Aggregate 2×2 heatmap per model, sample-level drill-down.
"""

import streamlit as st
import numpy as np

from dashboard.utils.data_loader import (
    load_dashboard_data, MODEL_KEYS, MODEL_LABELS,
)
from dashboard.utils.viz_helpers import confusion_heatmap


def render():
    st.title("🔲 Confusion Matrix")
    st.markdown(
        "Visualise the **aggregate** confusion matrix for each model across all test samples. "
        "Select a model and optionally drill down to a single sample."
    )

    df = load_dashboard_data()

    # ── Model selector ─────────────────────────────────────────────────────────
    model_key = st.selectbox("🤖 Select Model", MODEL_KEYS,
                              format_func=lambda k: MODEL_LABELS[k], key="cm_model")
    model_label = MODEL_LABELS[model_key]
    model_df    = df[df["model"] == model_label]

    # ── Aggregate confusion matrix ─────────────────────────────────────────────
    agg_tp = int(model_df["TP"].sum())
    agg_tn = int(model_df["TN"].sum())
    agg_fp = int(model_df["FP"].sum())
    agg_fn = int(model_df["FN"].sum())

    st.subheader(f"📊 Aggregate Confusion Matrix — {model_label}")
    c_main, c_stats = st.columns([2, 1])
    with c_main:
        fig = confusion_heatmap(agg_tn, agg_fp, agg_fn, agg_tp,
                                title=f"{model_label} — Aggregated over {len(model_df)} Samples")
        st.plotly_chart(fig, width="stretch", key=f"cm_agg_{model_key}")

    with c_stats:
        total = agg_tp + agg_tn + agg_fp + agg_fn
        st.markdown("### 📐 Derived Metrics")
        acc  = (agg_tp + agg_tn) / total if total else 0
        prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0
        rec  = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        iou  = agg_tp / (agg_tp + agg_fp + agg_fn) if (agg_tp + agg_fp + agg_fn) else 0

        st.metric("Accuracy",  f"{acc:.4f}")
        st.metric("Precision", f"{prec:.4f}")
        st.metric("Recall",    f"{rec:.4f}")
        st.metric("F1-Score",  f"{f1:.4f}")
        st.metric("IoU",       f"{iou:.4f}")

        st.markdown("---")
        st.markdown(
            f"**TP:** {agg_tp:,}  \n"
            f"**TN:** {agg_tn:,}  \n"
            f"**FP:** {agg_fp:,}  \n"
            f"**FN:** {agg_fn:,}  \n"
            f"**Total:** {total:,} pixels"
        )

    st.markdown("---")

    # ── Side-by-side all models ────────────────────────────────────────────────
    st.subheader("📊 All Models — Confusion Matrices")
    cols = st.columns(2)
    for i, mk in enumerate(MODEL_KEYS):
        ml = MODEL_LABELS[mk]
        mdf = df[df["model"] == ml]
        tp = int(mdf["TP"].sum())
        tn = int(mdf["TN"].sum())
        fp = int(mdf["FP"].sum())
        fn = int(mdf["FN"].sum())
        with cols[i % 2]:
            fig2 = confusion_heatmap(tn, fp, fn, tp, title=ml)
            st.plotly_chart(fig2, width="stretch", key=f"cm_all_{mk}")

    st.markdown("---")

    # ── Per-sample drill-down ──────────────────────────────────────────────────
    st.subheader("🔍 Per-Sample Confusion Matrix")
    samples = sorted(model_df["image_name"].unique())
    sel_sample = st.selectbox("Select sample", samples,
                               format_func=lambda s: s.replace(".png",""), key="cm_sel_sample")
    s_row = model_df[model_df["image_name"] == sel_sample].iloc[0]
    fig3 = confusion_heatmap(
        int(s_row.TN), int(s_row.FP), int(s_row.FN), int(s_row.TP),
        title=f"{model_label} — {sel_sample}"
    )
    st.plotly_chart(fig3, width="stretch",
                     key=f"cm_sample_{model_key}_{sel_sample}")
