"""
dashboard/app.py — Main entry point for the LEVIR-CD Streamlit Dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Tabs:
  1. Project Overview
  2. Model Comparison
  3. Change Detection Explorer
  4. Spatial Change Visualization
  5. Error Analysis
  6. Confusion Matrix
  7. XAI / Model Explanation
  8. Change Area Analysis
  9. Sample-Level Model Comparison
"""

import sys
import os

# ── Make sure the project root is on the Python path ──────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Also add dashboard dir so intra-package imports work when invoked via CLI ──
DASH = os.path.abspath(os.path.dirname(__file__))
if DASH not in sys.path:
    sys.path.insert(0, DASH)

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LEVIR-CD Change Detection Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Sidebar title */
      .sidebar .sidebar-content { padding-top: 1rem; }

      /* Metric card delta colour */
      [data-testid="stMetricDelta"] > div { color: #888 !important; }

      /* ── Tab bar: wrap onto multiple rows instead of squeezing/overlapping ── */
      .stTabs [data-baseweb="tab-list"],
      div[data-baseweb="tab-list"],
      [role="tablist"] {
          display: flex !important;
          flex-wrap: wrap !important;
          row-gap: 6px !important;
          column-gap: 4px !important;
          overflow: visible !important;
          height: auto !important;
          border-bottom: 1px solid rgba(255,255,255,0.12) !important;
      }

      /* Individual tab: fixed size + no shrink so labels never overlap */
      .stTabs [data-baseweb="tab"],
      button[data-baseweb="tab"],
      [role="tab"] {
          flex: 0 0 auto !important;
          font-size: 14px !important;
          font-weight: 600 !important;
          white-space: nowrap !important;
          padding: 8px 16px !important;
          margin: 0 !important;
          border-radius: 8px 8px 0 0 !important;
      }
      .stTabs [data-baseweb="tab"]:hover {
          background-color: rgba(255,255,255,0.06) !important;
      }
      .stTabs [data-baseweb="tab-highlight"] {
          background-color: #FF4B4B !important;
      }
      .stTabs [data-baseweb="tab-border"] {
          display: none !important;
      }

      /* Dataframe header */
      .stDataFrame th { background-color: #1f4068 !important; color: white !important; }

      /* Remove top padding */
      .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/220px-SNice.svg.png",
        width=60,
    )
    st.title("🛰️ LEVIR-CD")
    st.caption("Change Detection Dashboard")
    st.markdown("---")
    st.markdown("""
    **Project:** Information Visualization  
    **Dataset:** LEVIR-CD (Bi-temporal Satellite)  
    **Models:** 4 (CNN → Transformer)  
    **Environment:** CPU · PyTorch
    """)
    st.markdown("---")
    st.markdown("""
    **Navigation:**
    - Tab 1 → Project Overview
    - Tab 2 → Model Comparison
    - Tab 3 → Change Detection Explorer
    - Tab 4 → Spatial Change Visualization
    - Tab 5 → Error Analysis
    - Tab 6 → Confusion Matrix
    - Tab 7 → XAI Explanations
    - Tab 8 → Change Area Analysis
    - Tab 9 → Sample Comparison
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & Plotly")

# ── Tab layout ─────────────────────────────────────────────────────────────────
TABS = [
    "🏠 Overview",
    "📊 Model Comparison",
    "🔍 Explorer",
    "🗺️ Spatial",
    "🔬 Error Analysis",
    "🔲 Confusion Matrix",
    "🧠 XAI",
    "📐 Area Analysis",
    "🔀 Sample Compare",
]

tabs = st.tabs(TABS)

# Lazy imports to avoid circular dependencies
with tabs[0]:
    from dashboard.components import overview
    overview.render()

with tabs[1]:
    from dashboard.components import model_comparison
    model_comparison.render()

with tabs[2]:
    from dashboard.components import explorer
    explorer.render()

with tabs[3]:
    from dashboard.components import spatial_analysis
    spatial_analysis.render()

with tabs[4]:
    from dashboard.components import error_analysis
    error_analysis.render()

with tabs[5]:
    from dashboard.components import confusion_matrix
    confusion_matrix.render()

with tabs[6]:
    from dashboard.components import xai_explanation
    xai_explanation.render()

with tabs[7]:
    from dashboard.components import area_analysis
    area_analysis.render()

with tabs[8]:
    from dashboard.components import sample_comparison
    sample_comparison.render()
