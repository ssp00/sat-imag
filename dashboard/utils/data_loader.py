"""
data_loader.py — Cached data-loading utilities for the Streamlit dashboard.
All paths are resolved relative to the project root (parent of dashboard/).
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project root (one level above this file) ──────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET_ROOT = os.path.join(ROOT, "LEVIR CD", "LEVIR CD")
OUTPUTS_ROOT = os.path.join(ROOT, "outputs")

MODEL_KEYS   = ["siamese_cnn", "siamese_unet", "swin", "changeformer"]
MODEL_LABELS = {
    "siamese_cnn" : "Siamese CNN",
    "siamese_unet": "Siamese U-Net",
    "swin"        : "Swin Transformer",
    "changeformer": "ChangeFormer",
}

# Models were trained/evaluated at 224×224 (see src/evaluate.py img_size), and
# TP/TN/FP/FN in dashboard_data.csv are pixel counts at that resolution. Dataset
# images (LEVIR-CD) are natively 1024×1024, so every loaded array is resized to
# this size to stay aligned with the predictions and the precomputed metrics.
TARGET_SIZE = (224, 224)  # (width, height), PIL convention

# ── Helpers ───────────────────────────────────────────────────────────────────

def _img(path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array (resized to TARGET_SIZE), or grey placeholder."""
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
        return np.array(img)
    w, h = TARGET_SIZE
    return np.full((h, w, 3), 180, dtype=np.uint8)


def _mask(path: str) -> np.ndarray:
    """Load binary mask as 2-D uint8 array (0/255, resized to TARGET_SIZE), or zero placeholder."""
    if path and os.path.exists(path):
        img = Image.open(path).convert("L")
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, Image.NEAREST)
        arr = np.array(img)
        return (arr > 127).astype(np.uint8) * 255
    w, h = TARGET_SIZE
    return np.zeros((h, w), dtype=np.uint8)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    """Load master per-sample dashboard CSV."""
    csv_path = os.path.join(OUTPUTS_ROOT, "dashboard_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(OUTPUTS_ROOT, "metrics", "dashboard_data.csv")
    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    """Load aggregate per-model metrics (test_metrics.csv).
    Normalises column names to lowercase and ensures a 'model' column exists.
    """
    csv_path = os.path.join(OUTPUTS_ROOT, "metrics", "test_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Lowercase all column names
        df.columns = [c.lower().strip() for c in df.columns]
        # Rename index-like column to 'model' if needed
        for candidate in ["unnamed: 0", "model_name", "name", "index"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "model"})
                break
        # If still no 'model' column, reset index as model
        if "model" not in df.columns:
            df = df.reset_index().rename(columns={"index": "model"})
        return df
    # Derive from dashboard_data if test_metrics missing
    df = load_dashboard_data()
    agg = (df.groupby("model")[["accuracy", "precision", "recall", "f1", "iou", "dice"]]
             .mean().reset_index())
    return agg


@st.cache_data
def load_training_history(model_key: str) -> dict:
    """Load per-epoch training history JSON for a given model key."""
    key_map = {
        "siamese_cnn" : "baseline",
        "siamese_unet": "siamese_unet",
        "swin"        : "swin",
        "changeformer": "changeformer",
    }
    fname = f"history_{key_map.get(model_key, model_key)}.json"
    path  = os.path.join(OUTPUTS_ROOT, "metrics", fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_sample_images(sample_name: str) -> dict:
    """
    Return dict with keys: t1, t2, gt
    Values are RGB numpy arrays (H,W,3); gt is replicated to 3 channels.
    """
    t1_path = os.path.join(DATASET_ROOT, "test", "A", sample_name)
    t2_path = os.path.join(DATASET_ROOT, "test", "B", sample_name)
    gt_path = os.path.join(DATASET_ROOT, "test", "label", sample_name)

    t1 = _img(t1_path)
    t2 = _img(t2_path)
    gt_raw = _mask(gt_path)
    gt = np.stack([gt_raw, gt_raw, gt_raw], axis=-1)  # H,W,3
    return {"t1": t1, "t2": t2, "gt": gt, "gt_raw": gt_raw}


@st.cache_data
def load_prediction(model_key: str, sample_name: str) -> np.ndarray:
    """Load binary prediction mask for a model/sample (H,W,3)."""
    pred_dir = os.path.join(OUTPUTS_ROOT, "predictions", model_key)
    path = os.path.join(pred_dir, sample_name)
    raw = _mask(path)
    return np.stack([raw, raw, raw], axis=-1)


@st.cache_data
def load_xai_image(sample_name: str) -> np.ndarray:
    """Load the pre-generated XAI grid PNG (multi-model comparison)."""
    stem = os.path.splitext(sample_name)[0]   # e.g. "test_17"
    path = os.path.join(OUTPUTS_ROOT, "xai", f"xai_{stem}.png")
    return _img(path)


@st.cache_data
def get_test_samples() -> list:
    """Return sorted list of test sample filenames present in dashboard data."""
    df = load_dashboard_data()
    return sorted(df["image_name"].unique().tolist())


@st.cache_data
def load_experiment_config() -> dict:
    path = os.path.join(OUTPUTS_ROOT, "experiment_config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
