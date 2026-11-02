"""
viz_helpers.py — Visualization helper functions for the dashboard.
Pure numpy/matplotlib/plotly utilities; no Streamlit calls here.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour constants ──────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Siamese CNN"     : "#4C72B0",
    "Siamese U-Net"   : "#DD8452",
    "Swin Transformer": "#55A868",
    "ChangeFormer"    : "#C44E52",
}

METRIC_LABELS = {
    "accuracy" : "Accuracy",
    "precision": "Precision",
    "recall"   : "Recall",
    "f1"       : "F1-Score",
    "iou"      : "IoU",
    "dice"     : "Dice",
}

# ── Image utilities ───────────────────────────────────────────────────────────

def make_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Create a colour-coded error map:
      TP → Green   (0,200,0)
      TN → Dark    (30,30,30)
      FP → Red     (220,0,0)
      FN → Blue    (0,100,220)

    pred_mask, gt_mask: H×W uint8 (0 or 255).
    Returns H×W×3 uint8 RGB.
    """
    h, w = gt_mask.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)

    p = pred_mask > 127
    g = gt_mask   > 127

    out[ g &  p] = [0, 200, 0]    # TP green
    out[~g & ~p] = [30, 30, 30]   # TN dark
    out[~g &  p] = [220, 0, 0]    # FP red
    out[ g & ~p] = [0, 100, 220]  # FN blue
    return out


def overlay_mask_on_image(img_rgb: np.ndarray, mask: np.ndarray,
                           color=(255, 0, 0), alpha=0.4) -> np.ndarray:
    """Blend a binary mask over an RGB image."""
    out = img_rgb.astype(float).copy()
    c   = np.array(color, dtype=float)
    m   = (mask > 127)
    out[m] = (1 - alpha) * out[m] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def masks_to_rgba(mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    """Convert binary mask H×W to RGBA H×W×4, coloured where mask=255."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    m    = mask > 127
    rgba[m, 0] = color[0]
    rgba[m, 1] = color[1]
    rgba[m, 2] = color[2]
    rgba[m, 3] = 180
    return rgba

# ── Plotly chart builders ─────────────────────────────────────────────────────

def bar_chart_metric(model_metrics_df, metric: str, title: str = None) -> go.Figure:
    """Vertical bar chart for one metric across all models."""
    df  = model_metrics_df.copy()
    col = MODEL_COLORS
    fig = px.bar(
        df,
        x="model",
        y=metric,
        color="model",
        color_discrete_map=col,
        text_auto=".3f",
        title=title or METRIC_LABELS.get(metric, metric),
        labels={"model": "Model", metric: METRIC_LABELS.get(metric, metric)},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 1.05],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=50, b=40),
    )
    return fig


def grouped_bar_chart(model_metrics_df, metrics=("precision", "recall", "f1")) -> go.Figure:
    """Grouped bar chart: metrics side-by-side per model."""
    df  = model_metrics_df.copy()
    fig = go.Figure()
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
               "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=METRIC_LABELS.get(metric, metric),
            x=df["model"],
            y=df[metric],
            marker_color=palette[i],
            text=df[metric].round(3),
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Precision / Recall / F1 Comparison",
        yaxis_range=[0, 1.15],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=80, b=40),
    )
    return fig


def radar_chart(model_metrics_df) -> go.Figure:
    """Radar / spider chart comparing all metrics per model."""
    cats = ["accuracy", "precision", "recall", "f1", "iou", "dice"]
    cat_labels = [METRIC_LABELS[c] for c in cats]
    fig = go.Figure()
    for _, row in model_metrics_df.iterrows():
        vals = [row[c] for c in cats]
        vals_closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cat_labels + [cat_labels[0]],
            name=row["model"],
            fill="toself",
            line=dict(color=MODEL_COLORS.get(row["model"])),
            opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Radar — All Metrics",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(t=70),
    )
    return fig


def confusion_heatmap(tn: int, fp: int, fn: int, tp: int, title="Confusion Matrix") -> go.Figure:
    """2×2 heatmap styled as confusion matrix."""
    z      = [[tn, fp], [fn, tp]]
    text   = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
              [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted No-Change", "Predicted Change"],
        y=["Actual No-Change", "Actual Change"],
        colorscale="Blues",
        showscale=True,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=16),
    ))
    fig.update_layout(
        title=title,
        xaxis_side="bottom",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=60, b=60),
    )
    return fig


def scatter_change_area(df) -> go.Figure:
    """Scatter: GT change % vs Predicted change % per sample & model."""
    fig = px.scatter(
        df,
        x="true_change_percentage",
        y="predicted_change_percentage",
        color="model",
        symbol="image_name",
        color_discrete_map=MODEL_COLORS,
        hover_data=["image_name", "f1", "iou"],
        title="Ground-Truth vs Predicted Change Area (%)",
        labels={
            "true_change_percentage": "GT Change %",
            "predicted_change_percentage": "Predicted Change %",
        },
        size_max=14,
    )
    # Perfect prediction diagonal
    max_val = max(df["true_change_percentage"].max(),
                  df["predicted_change_percentage"].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect",
        showlegend=True,
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def diff_distribution(df) -> go.Figure:
    """Histogram of (predicted - GT) change area % per model."""
    df = df.copy()
    df["area_diff"] = df["predicted_change_percentage"] - df["true_change_percentage"]
    fig = px.histogram(
        df,
        x="area_diff",
        color="model",
        color_discrete_map=MODEL_COLORS,
        barmode="overlay",
        nbins=20,
        title="Distribution of Area Prediction Error (Predicted − GT %)",
        labels={"area_diff": "Change Area Error (%)"},
        opacity=0.7,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def training_curve_figure(histories: dict, metric="train_loss") -> go.Figure:
    """Line chart of training curves for all models."""
    fig = go.Figure()
    label_map = {
        "train_loss"   : "Training Loss",
        "val_loss"     : "Validation Loss",
        "val_accuracy" : "Validation Accuracy",
        "val_precision": "Validation Precision",
        "val_recall"   : "Validation Recall",
        "val_f1"       : "Validation F1",
        "val_iou"      : "Validation IoU",
        "val_dice"     : "Validation Dice",
    }
    key_map = {
        "siamese_cnn" : "Siamese CNN",
        "siamese_unet": "Siamese U-Net",
        "swin"        : "Swin Transformer",
        "changeformer": "ChangeFormer",
    }
    for mkey, hist in histories.items():
        if metric not in hist:
            continue
        vals = hist[metric]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vals)+1)),
            y=vals,
            mode="lines+markers",
            name=key_map.get(mkey, mkey),
            line=dict(color=MODEL_COLORS.get(key_map.get(mkey, mkey))),
        ))
    fig.update_layout(
        title=label_map.get(metric, metric),
        xaxis_title="Epoch",
        yaxis_title=label_map.get(metric, metric),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def sample_metric_heatmap(df) -> go.Figure:
    """Heatmap: samples × models coloured by F1."""
    pivot = df.pivot_table(index="image_name", columns="model", values="f1")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=pivot.values.round(3),
        texttemplate="%{text}",
        textfont=dict(size=13),
        colorbar=dict(title="F1"),
    ))
    fig.update_layout(
        title="Per-Sample F1 Heatmap (Models × Samples)",
        xaxis_title="Model",
        yaxis_title="Sample",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(t=60, b=60),
    )
    return fig
