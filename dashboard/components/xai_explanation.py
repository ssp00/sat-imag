"""
Tab 7 — XAI / Model Explanation
Displays pre-generated XAI overlay grids (Attention Rollout + Grad-CAM).
"""

import streamlit as st
import numpy as np

from dashboard.utils.data_loader import (
    get_test_samples, load_xai_image, load_sample_images,
    load_prediction, MODEL_KEYS, MODEL_LABELS,
)


def render():
    st.title("🧠 XAI — Model Explanations")
    st.markdown(
        "Understand **what the model focuses on** when predicting change. "
        "Pre-generated XAI grids show Attention Rollout (Transformer) and "
        "Grad-CAM saliency overlays."
    )

    # ── Explanation note ───────────────────────────────────────────────────────
    with st.expander("ℹ️ About the Explanation Methods", expanded=False):
        st.markdown("""
        ### Attention Rollout (Swin Transformer, ChangeFormer)
        - Propagates attention maps across all transformer layers by matrix
          multiplication, producing a single spatial attention map.
        - Highlights which **spatial regions** the model attends to most when
          processing T1 and T2 images.
        - **Note:** Attention maps show *correlation*, not causation. They indicate
          which regions influenced the output, not definitively *why*.

        ### Grad-CAM (Siamese CNN, Siamese U-Net)
        - Computes gradients of the output w.r.t. the final convolutional feature
          maps to produce a coarse class activation map.
        - Effective for CNN-based architectures; applied to the last decoder block.
        - **Not appropriate** for pure Transformer layers — only used where
          compatible convolutional layers exist.

        ### XAI Overlap Metric
        - Measures the intersection-over-union of the XAI saliency map with the
          ground-truth change mask.
        - Higher overlap → explanation is consistent with actual changes.

        ### XAI Stability Score
        - Cosine similarity of saliency maps between two nearby input perturbations.
        - Higher stability → more consistent explanations.
        """)

    samples = get_test_samples()

    # ── Sample selector ────────────────────────────────────────────────────────
    sample = st.selectbox("📂 Select Test Sample", samples,
                           format_func=lambda s: s.replace(".png",""), key="xai_sample")

    # ── XAI grid image ─────────────────────────────────────────────────────────
    xai_img = load_xai_image(sample)
    st.subheader("🖼️ XAI Explanation Grid")
    st.image(xai_img, caption=f"XAI grid — {sample} (T1 | Pred | GT | XAI overlay)",
             width="stretch")
    st.caption(
        "Each row corresponds to a model. Columns: T1 image · Predicted mask · "
        "Ground truth · Explanation overlay."
    )

    st.markdown("---")

    # ── Side-by-side T2 / GT / Prediction / XAI columns ──────────────────────
    st.subheader("🔍 Detailed View — Select Model")
    model_key   = st.selectbox("🤖 Model", MODEL_KEYS,
                                format_func=lambda k: MODEL_LABELS[k], key="xai_model")
    imgs        = load_sample_images(sample)
    pred        = load_prediction(model_key, sample)

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.image(imgs["t1"], caption="🕐 T1 (Before)", width="stretch")
        st.image(imgs["t2"], caption="🕑 T2 (After)",  width="stretch")
    with cc2:
        st.image(imgs["gt"], caption="✅ Ground Truth",             width="stretch")
        st.image(pred,       caption=f"🤖 {MODEL_LABELS[model_key]} Prediction",
                 width="stretch")
    with cc3:
        st.image(xai_img,   caption="🧠 XAI Grid (all models)",    width="stretch")

    st.markdown("---")

    # ── XAI metric table ───────────────────────────────────────────────────────
    from dashboard.utils.data_loader import load_dashboard_data
    df = load_dashboard_data()
    st.subheader("📊 XAI Metrics")
    xai_cols = ["model", "image_name", "xai_overlap", "xai_stability"]
    avail    = [c for c in xai_cols if c in df.columns]
    xai_df   = df[df["image_name"] == sample][avail].reset_index(drop=True)
    for c in ["xai_overlap", "xai_stability"]:
        if c in xai_df.columns:
            xai_df[c] = xai_df[c].round(4)
    st.dataframe(xai_df, width="stretch", hide_index=True)

    st.info(
        "**XAI Overlap**: IoU between the saliency map and the GT change mask. "
        "Higher = explanation highlights the correct regions.  \n"
        "**XAI Stability**: Cosine similarity across two perturbed passes. "
        "Higher = more consistent explanations."
    )

    st.markdown("---")

    # ── All XAI grids ──────────────────────────────────────────────────────────
    st.subheader("📁 All Sample XAI Grids")
    for s in samples:
        with st.expander(f"XAI — {s.replace('.png','')}"):
            img = load_xai_image(s)
            st.image(img, width="stretch")
