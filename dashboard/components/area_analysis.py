"""
Tab 8 — Change Area Analysis
Scatter (GT% vs Pred%), distribution of area errors, per-model stats.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils.data_loader import load_dashboard_data, MODEL_LABELS, MODEL_KEYS
from dashboard.utils.viz_helpers import scatter_change_area, diff_distribution, MODEL_COLORS


def render():
    st.title("📐 Change Area Analysis")
    st.markdown(
        "Analyse how accurately each model estimates the **extent** of change, "
        "not just *where* it occurs. Compare predicted change area % against ground truth."
    )

    df = load_dashboard_data()

    # ── Scatter: GT% vs Pred% ──────────────────────────────────────────────────
    st.subheader("🎯 Ground Truth vs Predicted Change Area")
    st.plotly_chart(scatter_change_area(df), width="stretch", key="area_scatter")
    st.caption(
        "Points on the diagonal (dashed line) indicate perfect area estimation. "
        "Points above = over-prediction; below = under-prediction."
    )

    st.markdown("---")

    # ── Distribution of errors ─────────────────────────────────────────────────
    st.subheader("📊 Area Prediction Error Distribution")
    st.plotly_chart(diff_distribution(df), width="stretch", key="area_diff_dist")
    st.caption(
        "Histogram of (Predicted % − GT %). "
        "Values clustered around 0 indicate better calibration."
    )

    st.markdown("---")

    # ── Per-model area summary ─────────────────────────────────────────────────
    st.subheader("📋 Per-Model Area Statistics")
    stats = []
    for mk in MODEL_KEYS:
        ml  = MODEL_LABELS[mk]
        mdf = df[df["model"] == ml]
        diff = mdf["predicted_change_percentage"] - mdf["true_change_percentage"]
        stats.append({
            "Model"         : ml,
            "Mean GT %"     : round(mdf["true_change_percentage"].mean(), 3),
            "Mean Pred %"   : round(mdf["predicted_change_percentage"].mean(), 3),
            "Mean Error %"  : round(diff.mean(), 3),
            "Abs Error %"   : round(diff.abs().mean(), 3),
            "Max Over-pred" : round(diff.max(), 3),
            "Max Under-pred": round(diff.min(), 3),
        })
    import pandas as pd
    stats_df = pd.DataFrame(stats)
    st.dataframe(stats_df, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Line chart: GT vs Pred per sample per model ────────────────────────────
    st.subheader("📈 Sample-wise Change Area — GT vs All Models")
    samples = sorted(df["image_name"].unique())
    fig = go.Figure()
    # GT (take from any model — same for all)
    gt_vals = df[df["model"] == MODEL_LABELS[MODEL_KEYS[0]]].sort_values("image_name")
    fig.add_trace(go.Scatter(
        x=gt_vals["image_name"],
        y=gt_vals["true_change_percentage"],
        mode="lines+markers",
        name="Ground Truth",
        line=dict(color="gray", dash="dash", width=2),
        marker=dict(symbol="diamond", size=10),
    ))
    for mk in MODEL_KEYS:
        ml  = MODEL_LABELS[mk]
        mdf = df[df["model"] == ml].sort_values("image_name")
        fig.add_trace(go.Scatter(
            x=mdf["image_name"],
            y=mdf["predicted_change_percentage"],
            mode="lines+markers",
            name=ml,
            line=dict(color=MODEL_COLORS[ml]),
            marker=dict(size=8),
        ))
    fig.update_layout(
        xaxis_title="Sample",
        yaxis_title="Change Area (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch", key="area_samplewise_line")

    st.markdown("---")

    # ── Area vs F1 scatter ────────────────────────────────────────────────────
    st.subheader("🔗 Relationship: Change Area vs F1-Score")
    fig2 = px.scatter(
        df,
        x="true_change_percentage",
        y="f1",
        color="model",
        size="iou",
        color_discrete_map=MODEL_COLORS,
        hover_data=["image_name", "precision", "recall"],
        title="GT Change % vs F1-Score (bubble size = IoU)",
        labels={"true_change_percentage": "GT Change %", "f1": "F1-Score"},
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig2, width="stretch", key="area_vs_f1_scatter")
    st.caption("Samples with larger change areas do not always produce higher F1 scores.")
