"""
Tab 3 — Change Detection Explorer
Select sample + model → display T1 | T2 | GT | Prediction side by side.
"""

import streamlit as st
import numpy as np
from PIL import Image

from dashboard.utils.data_loader import (
    get_test_samples, load_sample_images, load_prediction,
    load_dashboard_data, MODEL_KEYS, MODEL_LABELS,
)


def render():
    st.title("🔍 Change Detection Explorer")
    st.markdown(
        "Select a test sample and a model to visualise the bi-temporal satellite "
        "images alongside the ground-truth and predicted change maps."
    )

    samples = get_test_samples()
    df      = load_dashboard_data()

    # ── Controls ───────────────────────────────────────────────────────────────
    col_s, col_m = st.columns(2)
    with col_s:
        sample = st.selectbox("📂 Select Test Sample", samples,
                              format_func=lambda s: s.replace(".png",""), key="explorer_sample")
    with col_m:
        model_key = st.selectbox("🤖 Select Model", MODEL_KEYS,
                                  format_func=lambda k: MODEL_LABELS[k], key="explorer_model")

    # ── Load images ────────────────────────────────────────────────────────────
    imgs = load_sample_images(sample)
    pred = load_prediction(model_key, sample)

    # ── Metrics for selection ──────────────────────────────────────────────────
    row = df[(df["image_name"] == sample) & (df["model"] == MODEL_LABELS[model_key])]
    if not row.empty:
        r = row.iloc[0]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy",  f"{r['accuracy']:.4f}")
        mc2.metric("Precision", f"{r['precision']:.4f}")
        mc3.metric("Recall",    f"{r['recall']:.4f}")
        mc4.metric("F1-Score",  f"{r['f1']:.4f}")
        mc5.metric("IoU",       f"{r['iou']:.4f}")

    st.markdown("---")

    # ── 4-panel image display ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(imgs["t1"], caption="🕐 T1 (Before)", width="stretch")
    with c2:
        st.image(imgs["t2"], caption="🕑 T2 (After)", width="stretch")
    with c3:
        st.image(imgs["gt"], caption="✅ Ground Truth", width="stretch")
    with c4:
        st.image(pred, caption=f"🤖 {MODEL_LABELS[model_key]} Prediction",
                 width="stretch")

    st.markdown("---")

    # ── All models for same sample ─────────────────────────────────────────────
    st.subheader("🔄 Compare All 4 Models — Same Sample")
    st.caption("Switch between models using the selector above, or view all predictions below.")

    cols = st.columns(4)
    for i, mk in enumerate(MODEL_KEYS):
        p = load_prediction(mk, sample)
        with cols[i]:
            st.image(p, caption=MODEL_LABELS[mk], width="stretch")

    st.markdown("---")

    # ── Per-model metric comparison for this sample ────────────────────────────
    st.subheader(f"📊 Metrics — Sample: {sample.replace('.png','')}")
    sample_df = df[df["image_name"] == sample][
        ["model","accuracy","precision","recall","f1","iou","dice"]
    ].reset_index(drop=True)
    for c in ["accuracy","precision","recall","f1","iou","dice"]:
        sample_df[c] = sample_df[c].round(4)
    st.dataframe(sample_df, width="stretch", hide_index=True)
