"""
Tab 2 — Model Comparison
Bar charts (F1, IoU, Dice), grouped Precision/Recall/F1, ranked table.
"""

import streamlit as st
import plotly.express as px
from dashboard.utils.data_loader import load_model_metrics
from dashboard.utils.viz_helpers import (
    bar_chart_metric, grouped_bar_chart, radar_chart, training_curve_figure,
    sample_metric_heatmap, MODEL_COLORS,
)
from dashboard.utils.data_loader import load_training_history, load_dashboard_data, MODEL_KEYS


def render():
    st.title("📊 Model Comparison")
    st.markdown(
        "Compare all four models across accuracy, precision, recall, F1, IoU and Dice. "
        "All values are derived from actual evaluation on the LEVIR-CD test split."
    )

    metrics = load_model_metrics()
    df_all  = load_dashboard_data()

    # ── Rank table ─────────────────────────────────────────────────────────────
    st.subheader("🏆 Performance Ranking")
    rank = metrics.copy()
    for c in ["accuracy", "precision", "recall", "f1", "iou", "dice"]:
        if c in rank.columns:
            rank[c] = rank[c].round(4)
    rank = rank.sort_values("f1", ascending=False).reset_index(drop=True)
    rank.insert(0, "Rank", range(1, len(rank)+1))
    st.dataframe(rank, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Individual metric bars ─────────────────────────────────────────────────
    st.subheader("📊 Individual Metric Bar Charts")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.plotly_chart(bar_chart_metric(metrics, "f1", "F1-Score by Model"),
                        width="stretch", key="mc_bar_f1")
    with r1c2:
        st.plotly_chart(bar_chart_metric(metrics, "iou", "IoU by Model"),
                        width="stretch", key="mc_bar_iou")
    with r1c3:
        st.plotly_chart(bar_chart_metric(metrics, "dice", "Dice Coefficient by Model"),
                        width="stretch", key="mc_bar_dice")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.plotly_chart(bar_chart_metric(metrics, "accuracy", "Accuracy by Model"),
                        width="stretch", key="mc_bar_accuracy")
    with r2c2:
        st.plotly_chart(bar_chart_metric(metrics, "precision", "Precision by Model"),
                        width="stretch", key="mc_bar_precision")
    with r2c3:
        st.plotly_chart(bar_chart_metric(metrics, "recall", "Recall by Model"),
                        width="stretch", key="mc_bar_recall")

    st.markdown("---")

    # ── Grouped Precision / Recall / F1 ───────────────────────────────────────
    st.subheader("📊 Grouped Precision / Recall / F1")
    st.plotly_chart(grouped_bar_chart(metrics, ["precision", "recall", "f1"]),
                    width="stretch", key="mc_grouped_bar")

    st.markdown("---")

    # ── Radar chart ────────────────────────────────────────────────────────────
    st.subheader("🕸️ Radar Chart — All Metrics")
    st.plotly_chart(radar_chart(metrics), width="stretch", key="mc_radar")

    st.markdown("---")

    # ── Per-sample F1 heatmap ──────────────────────────────────────────────────
    st.subheader("🗂️ Per-Sample F1 Heatmap")
    st.plotly_chart(sample_metric_heatmap(df_all), width="stretch", key="mc_sample_heatmap")
    st.caption(
        "Each cell shows the F1-score for a specific model × sample combination. "
        "Green = high F1, Red = low F1."
    )

    st.markdown("---")

    # ── Training curves ────────────────────────────────────────────────────────
    st.subheader("📉 Training Curves")
    histories = {k: load_training_history(k) for k in MODEL_KEYS}
    has_data  = any(h for h in histories.values())

    if has_data:
        curve_metric = st.selectbox(
            "Select metric to plot",
            ["train_loss", "val_loss", "val_accuracy", "val_precision",
             "val_recall", "val_f1", "val_iou", "val_dice"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="mc_curve_metric",
        )
        st.plotly_chart(training_curve_figure(histories, curve_metric),
                        width="stretch", key=f"mc_training_curve_{curve_metric}")
    else:
        st.info("Training history files not found. Run evaluation to generate them.")

    st.markdown("---")

    # ── Insight callouts ───────────────────────────────────────────────────────
    st.subheader("💡 Key Observations")
    best_f1_row = metrics.loc[metrics["f1"].idxmax()]
    best_rec_row = metrics.loc[metrics["recall"].idxmax()]
    st.info(
        f"**Best F1:** {best_f1_row['model']} ({best_f1_row['f1']:.4f})  \n"
        f"**Best Recall:** {best_rec_row['model']} ({best_rec_row['recall']:.4f})  \n"
        "Higher recall is important for change detection to avoid missed changes (False Negatives)."
    )
