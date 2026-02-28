"""
Phase 6 — Interactive "Glass-Box" Web UI
==========================================

Production-ready Streamlit dashboard for the DR-ISI Traffic Risk
Prediction System.  Acts as a risk-monitoring and model-explainability
interface, exposing all Phase 5 XAI artifacts through an interactive UI.

Layout:
    ┌──────────────────────────────────────────────────────────┐
    │ SIDEBAR                                                  │
    │  - Title: DR-ISI Traffic Risk Monitor                    │
    │  - Node selector (Top-5 from summary.json)               │
    │  - Selected node statistics                              │
    ├──────────────────────────────────────────────────────────┤
    │ MAIN PAGE                                                │
    │  Section A: PyDeck Geospatial Risk Map                   │
    │  Section B: XAI Tabs                                     │
    │    Tab 1 — Feature Drivers (Local SHAP)                  │
    │    Tab 2 — Temporal Evolution (Captum IG)                │
    │    Tab 3 — Global Model Insights (Global SHAP)           │
    └──────────────────────────────────────────────────────────┘

Data Sources (all under data/processed/xai_artifacts/):
    - summary.json              → Top-5 high-risk node list
    - global_feature_importance.json → Global SHAP values
    - subgraph_{id}.json        → GNNExplainer edge importance
    - temporal_profile_{id}.json → Captum 24-hour attribution
    - local_shap_{id}.json      → Per-node SHAP feature values

Coordinate System:
    - Source data is in UTM Zone 14N (EPSG:32614)
    - PyDeck requires WGS84 (EPSG:4326) geographic coordinates
    - Conversion handled by pyproj.Transformer at load time

Usage:
    streamlit run app.py

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-28
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from pyproj import Transformer

# =====================================================================
# Constants & Paths
# =====================================================================

# Directory containing all XAI JSON artifacts from Phase 5
XAI_DIR = Path("data/processed/xai_artifacts")

# Processed graph data for node position lookup
GRAPH_DATA_PATH = Path("data/processed/processed_graph_data.pt")

# Node mapping from Phase 2 (OSMnx node-id → contiguous idx)
NODE_MAPPING_PATH = Path("data/processed/node_mapping.pkl")

# Coordinate Reference Systems
CRS_UTM = "EPSG:32614"    # UTM Zone 14N (source — Dallas)
CRS_WGS84 = "EPSG:4326"   # WGS84 geographic (target — PyDeck)

# Default map centre (Dallas, TX)
DALLAS_LAT = 32.78
DALLAS_LON = -96.80

# Feature display names (matches Phase 3 dynamic feature order)
FEATURE_NAMES = ["crash_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

# Human-readable labels for feature names
FEATURE_LABELS = {
    "crash_count": "Crash Count",
    "hour_sin":    "Hour (sin)",
    "hour_cos":    "Hour (cos)",
    "dow_sin":     "Day-of-Week (sin)",
    "dow_cos":     "Day-of-Week (cos)",
}


# =====================================================================
# CRS Transformer (UTM → WGS84)
# =====================================================================

# Build a reusable pyproj transformer for converting UTM Zone 14N
# coordinates (metres) to WGS84 geographic (degrees).
# always_xy=True ensures the input/output order is (easting, northing)
# → (longitude, latitude).
_transformer = Transformer.from_crs(CRS_UTM, CRS_WGS84, always_xy=True)


def utm_to_latlon(easting: float, northing: float) -> tuple[float, float]:
    """
    Convert a single UTM Zone 14N coordinate to WGS84 (lat, lon).

    Parameters
    ----------
    easting : float
        UTM easting in metres (the "x" coordinate).
    northing : float
        UTM northing in metres (the "y" coordinate).

    Returns
    -------
    tuple[float, float]
        (latitude, longitude) in decimal degrees.
    """
    lon, lat = _transformer.transform(easting, northing)
    return lat, lon


# =====================================================================
# Cached Data Loaders
# =====================================================================
# All data loaders use @st.cache_data so that JSON files are read from
# disk only once per session.  This keeps the dashboard snappy even
# when switching between nodes.


@st.cache_data(show_spinner=False)
def load_summary() -> dict | None:
    """
    Load the summary.json containing Top-5 high-risk node information.

    Returns
    -------
    dict or None
        Parsed JSON with key "top_risk_nodes", each entry containing
        node_idx, mean_risk, max_risk, longitude (UTM), latitude (UTM).
        Returns None if the file is not found.
    """
    path = XAI_DIR / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_global_shap() -> dict | None:
    """
    Load global SHAP feature importance from Phase 5.

    Returns
    -------
    dict or None
        Contains "global_importance" mapping feature names to mean
        |SHAP value|.  Returns None if file is missing.
    """
    path = XAI_DIR / "global_feature_importance.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_subgraph(node_id: int) -> dict | None:
    """
    Load GNNExplainer subgraph for a specific node.

    Parameters
    ----------
    node_id : int
        Contiguous node index (from summary.json).

    Returns
    -------
    dict or None
        Contains "top_k_edges" (list of edge dicts with src_node,
        dst_node, weight), "subgraph_nodes" (list of node indices),
        and "edge_mask_stats".  Returns None if file is missing.
    """
    path = XAI_DIR / f"subgraph_{node_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_temporal_profile(node_id: int) -> dict | None:
    """
    Load Captum IntegratedGradients temporal profile for a node.

    Parameters
    ----------
    node_id : int
        Contiguous node index.

    Returns
    -------
    dict or None
        Contains "temporal_importance_raw" (24 floats),
        "temporal_importance_normalized" (24 floats summing to 1),
        "peak_hour_offset" (int), and "window_size" (int).
    """
    path = XAI_DIR / f"temporal_profile_{node_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_local_shap(node_id: int) -> dict | None:
    """
    Load local SHAP values for a specific node.

    Parameters
    ----------
    node_id : int
        Contiguous node index.

    Returns
    -------
    dict or None
        Contains "local_shap" mapping feature names to SHAP values,
        and "node_idx".  Returns None if file is missing.
    """
    path = XAI_DIR / f"local_shap_{node_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_node_positions() -> dict[int, tuple[float, float]] | None:
    """
    Build a lookup table: contiguous node index → (lat, lon) in WGS84.

    Strategy:
        1. Load the PyG Data object from processed_graph_data.pt
           (only ~3.2 MB; the 'pos' tensor is 24,697 × 2 floats).
        2. Convert each (easting, northing) to (lat, lon) via pyproj.
        3. Return a dict keyed by contiguous node index.

    This is cached with @st.cache_resource so it runs only once per
    Streamlit session, keeping the dashboard responsive.

    Returns
    -------
    dict[int, tuple[float, float]] or None
        Mapping of node_idx → (latitude, longitude).
        Returns None if the data file is not found.
    """
    if not GRAPH_DATA_PATH.exists():
        return None

    try:
        import torch
        data = torch.load(GRAPH_DATA_PATH, map_location="cpu", weights_only=False)

        if not hasattr(data, "pos") or data.pos is None:
            return None

        # data.pos is (N, 2) with columns [easting, northing] in UTM
        pos_np = data.pos.numpy()  # → numpy for fast iteration

        positions: dict[int, tuple[float, float]] = {}
        for idx in range(pos_np.shape[0]):
            easting, northing = pos_np[idx]
            lat, lon = utm_to_latlon(easting, northing)
            positions[idx] = (lat, lon)

        return positions

    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_all_node_risks() -> list[dict] | None:
    """
    Load every node's static DR-ISI score and WGS84 position from
    processed_graph_data.pt.

    Returns
    -------
    list[dict] or None
        Each dict contains node_idx, dr_isi, lat, lon.
        Only non-zero DR-ISI nodes are included (~3,184 of 24,697).
        Sorted by dr_isi descending so slice [:N] gives the top-N.
        Returns None if the data file is not found or cannot be read.
    """
    if not GRAPH_DATA_PATH.exists():
        return None

    try:
        import torch
        data = torch.load(GRAPH_DATA_PATH, map_location="cpu", weights_only=False)

        if not hasattr(data, "pos") or data.pos is None:
            return None
        if not hasattr(data, "y") or data.y is None:
            return None

        pos_np = data.pos.numpy()   # (N, 2) — UTM easting/northing
        y_np   = data.y.numpy()     # (N,)   — DR-ISI scores

        nodes = []
        for idx in range(pos_np.shape[0]):
            dr_isi = float(y_np[idx])
            if dr_isi <= 0.0:       # skip zero-risk nodes
                continue
            easting, northing = pos_np[idx]
            lat, lon = utm_to_latlon(easting, northing)
            nodes.append({
                "node_idx": int(idx),
                "dr_isi":   dr_isi,
                "lat":      lat,
                "lon":      lon,
            })

        nodes.sort(key=lambda n: n["dr_isi"], reverse=True)
        return nodes

    except Exception:
        return None


# =====================================================================
# Visualization Builders
# =====================================================================


def build_risk_map(
    selected_node: dict,
    all_nodes: list[dict],
    subgraph_data: dict | None,
    node_positions: dict[int, tuple[float, float]] | None,
    extended_nodes: list[dict] | None = None,
) -> pdk.Deck:
    """
    Construct the PyDeck geospatial risk map for Dallas, TX.

    Layers:
        1. ScatterplotLayer — Red dot for selected intersection,
           orange dots for other XAI top-K nodes, small yellow dots
           for any additional high-risk nodes in extended_nodes.
        2. ArcLayer — Edges from GNNExplainer subgraph, where arc
           colour intensity is proportional to edge weight.

    Parameters
    ----------
    selected_node : dict
        The currently selected node (from summary.json).
    all_nodes : list[dict]
        All XAI top-K nodes (from summary.json).
    subgraph_data : dict or None
        GNNExplainer output for the selected node.
    node_positions : dict or None
        Node index → (lat, lon) lookup table.
    extended_nodes : list[dict] or None
        Additional high-risk nodes beyond the XAI top-K, each with
        keys node_idx, dr_isi, lat, lon (already in WGS84).

    Returns
    -------
    pdk.Deck
        Configured PyDeck map object.
    """
    # --- Convert selected node's UTM coordinates to lat/lon ---
    sel_lat, sel_lon = utm_to_latlon(
        selected_node["longitude"], selected_node["latitude"]
    )

    # --- Build scatter data ---
    # Extended (background) nodes first so XAI nodes render on top
    scatter_data = []

    xai_node_idxs = {n["node_idx"] for n in all_nodes}

    if extended_nodes:
        for node in extended_nodes:
            if node["node_idx"] not in xai_node_idxs:
                scatter_data.append({
                    "lat":       node["lat"],
                    "lon":       node["lon"],
                    "node_idx":  node["node_idx"],
                    "mean_risk": node["dr_isi"],
                    "color":     [255, 220, 0, 100],  # semi-transparent yellow
                    "radius":    40,
                })

    # XAI top-K nodes rendered on top of extended nodes
    for node in all_nodes:
        lat, lon = utm_to_latlon(node["longitude"], node["latitude"])
        is_selected = node["node_idx"] == selected_node["node_idx"]
        scatter_data.append({
            "lat": lat,
            "lon": lon,
            "node_idx": node["node_idx"],
            "mean_risk": node["mean_risk"],
            # Selected node = red (large), others = orange (smaller)
            "color": [220, 30, 30, 220] if is_selected else [255, 140, 0, 180],
            "radius": 150 if is_selected else 80,
        })

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    # --- Build arc data from GNNExplainer subgraph edges ---
    layers = [scatter_layer]

    if subgraph_data and node_positions:
        edges = subgraph_data.get("top_k_edges", [])
        max_weight = max((e["weight"] for e in edges), default=0.0)

        arc_data = []
        for edge in edges:
            src_idx = edge["src_node"]
            dst_idx = edge["dst_node"]

            # Look up lat/lon for source and destination nodes
            if src_idx in node_positions and dst_idx in node_positions:
                src_lat, src_lon = node_positions[src_idx]
                dst_lat, dst_lon = node_positions[dst_idx]

                # Map edge weight to colour intensity (0-255)
                # If all weights are 0.0, use uniform orange fallback
                if max_weight > 0:
                    intensity = int((edge["weight"] / max_weight) * 255)
                else:
                    intensity = 140  # Uniform fallback

                arc_data.append({
                    "src_lon": src_lon,
                    "src_lat": src_lat,
                    "dst_lon": dst_lon,
                    "dst_lat": dst_lat,
                    "weight": edge["weight"],
                    "color": [255, intensity, 0, 200],
                })

        if arc_data:
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["dst_lon", "dst_lat"],
                get_source_color="color",
                get_target_color="color",
                get_width=3,
                pickable=True,
                auto_highlight=True,
            )
            layers.append(arc_layer)

    # --- Assemble the PyDeck map ---
    view_state = pdk.ViewState(
        latitude=sel_lat,
        longitude=sel_lon,
        zoom=13,
        pitch=45,
        bearing=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "html": "<b>Node {node_idx}</b><br/>Mean Risk: {mean_risk:.4f}",
            "style": {"color": "white"},
        },
    )


def build_local_shap_chart(shap_data: dict) -> go.Figure:
    """
    Create a Plotly horizontal bar chart of local SHAP feature importance.

    Answers the question: "What conditions caused this specific node's
    risk prediction?"

    Parameters
    ----------
    shap_data : dict
        Contains "local_shap" mapping feature names → SHAP values.

    Returns
    -------
    go.Figure
        Plotly figure with horizontal bars sorted by importance.
    """
    local_shap = shap_data["local_shap"]

    # Sort features by importance (ascending for horizontal bar)
    sorted_items = sorted(local_shap.items(), key=lambda x: x[1])
    features = [FEATURE_LABELS.get(f, f) for f, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Colour gradient: less important → light blue, more → dark blue
    max_val = max(values) if values else 1
    colors = [
        f"rgba(31, 119, 180, {0.3 + 0.7 * (v / max_val if max_val > 0 else 0)})"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.6f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text=f"Local SHAP — Node {shap_data.get('node_idx', '?')}",
            font_size=16,
        ),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )

    return fig


def build_temporal_chart(temporal_data: dict) -> go.Figure:
    """
    Create a Plotly area+line chart of Captum temporal attribution.

    Answers the question: "When did the risk start building up in the
    24-hour lookback window?"

    Parameters
    ----------
    temporal_data : dict
        Contains "temporal_importance_normalized" (24 values summing
        to 1) and "peak_hour_offset" (int).

    Returns
    -------
    go.Figure
        Plotly figure with filled area, line, and peak annotation.
    """
    normalized = temporal_data["temporal_importance_normalized"]
    peak_hour = temporal_data["peak_hour_offset"]
    hours = list(range(len(normalized)))

    # Create descriptive hour labels: "t-23 (oldest)" ... "t-0 (newest)"
    hour_labels = [f"t-{23 - h}" for h in hours]

    fig = go.Figure()

    # Filled area for visual impact
    fig.add_trace(go.Scatter(
        x=hours,
        y=normalized,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255, 165, 0, 0.2)",
        line=dict(color="rgba(255, 165, 0, 0.8)", width=2),
        name="Attribution",
        hovertemplate="Hour offset: %{x}<br>Importance: %{y:.4f}<extra></extra>",
    ))

    # Marker dots at each timestep
    fig.add_trace(go.Scatter(
        x=hours,
        y=normalized,
        mode="markers",
        marker=dict(color="darkorange", size=6),
        name="Timesteps",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Vertical line + annotation at peak hour
    peak_val = normalized[peak_hour]
    fig.add_vline(
        x=peak_hour,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Peak: t-{23 - peak_hour}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    fig.update_layout(
        title=dict(
            text=f"Temporal Attribution — Node {temporal_data['target_node_idx']}",
            font_size=16,
        ),
        xaxis=dict(
            title="Hour Offset (0 = oldest, 23 = most recent)",
            tickmode="array",
            tickvals=list(range(0, 24, 3)),
            ticktext=[hour_labels[i] for i in range(0, 24, 3)],
        ),
        yaxis_title="Normalised Importance",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )

    return fig


def build_global_shap_chart(global_data: dict) -> go.Figure:
    """
    Create a Plotly horizontal bar chart of global SHAP feature importance.

    Provides city-wide context: "How does the AI model make decisions
    across all of Dallas?"

    Parameters
    ----------
    global_data : dict
        Contains "global_importance" mapping feature names → mean
        |SHAP value| (averaged across all explained nodes and samples).

    Returns
    -------
    go.Figure
        Plotly figure with horizontal bars sorted by importance.
    """
    importance = global_data["global_importance"]

    # Sort by importance (ascending for horizontal bar display)
    sorted_items = sorted(importance.items(), key=lambda x: x[1])
    features = [FEATURE_LABELS.get(f, f) for f, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Colour gradient using Blues colourmap
    max_val = max(values) if values else 1
    colors = [
        f"rgba(31, 119, 180, {0.3 + 0.7 * (v / max_val if max_val > 0 else 0)})"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.6f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text="Global Feature Importance (SHAP)",
            font_size=16,
        ),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )

    return fig


# =====================================================================
# Streamlit Application
# =====================================================================

def main() -> None:
    """
    Entry point for the Streamlit dashboard.

    Configures the page, builds the sidebar with node selection,
    renders the geospatial map, and displays XAI explanation tabs.
    """

    # --- Page Configuration ---
    st.set_page_config(
        page_title="DR-ISI Traffic Risk Monitor",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom CSS for dark-themed styling ---
    st.markdown("""
    <style>
        /* Metric cards in sidebar */
        [data-testid="stMetric"] {
            background: rgba(28, 131, 225, 0.08);
            border: 1px solid rgba(28, 131, 225, 0.2);
            border-radius: 8px;
            padding: 10px 14px;
        }
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # =================================================================
    # Load core data
    # =================================================================

    summary = load_summary()

    if summary is None:
        st.error(
            "❌ **summary.json not found.**\n\n"
            "Run Phase 5 first:\n"
            "```bash\n"
            "python src/generate_insights.py\n"
            "```"
        )
        st.stop()

    top_nodes = summary["top_risk_nodes"]

    # Load the node position lookup table (cached, runs once)
    node_positions = load_node_positions()

    # =================================================================
    # Sidebar — Node Selection
    # =================================================================

    with st.sidebar:
        st.markdown("# 🚦 DR-ISI Traffic Risk Monitor")
        st.markdown("---")
        st.markdown("### 🔍 Intersection Inspector")

        # Build descriptive labels for the dropdown
        node_labels = [
            f"Node {n['node_idx']} — Risk: {n['mean_risk']:.4f}"
            for n in top_nodes
        ]

        # Dropdown populated from summary.json
        selected_idx = st.selectbox(
            "Select a High-Risk Intersection",
            range(len(top_nodes)),
            format_func=lambda i: node_labels[i],
            help="Choose one of the Top-5 riskiest intersections "
                 "identified by the model.",
        )

        selected_node = top_nodes[selected_idx]
        node_id = selected_node["node_idx"]

        # Display key statistics for the selected node
        st.markdown("---")
        st.markdown(f"### 📊 Node {node_id} Stats")

        col1, col2 = st.columns(2)
        col1.metric("Mean Risk", f"{selected_node['mean_risk']:.4f}")
        col2.metric("Max Risk", f"{selected_node['max_risk']:.4f}")

        # Show geographic coordinates (converted from UTM)
        lat, lon = utm_to_latlon(
            selected_node["longitude"], selected_node["latitude"]
        )
        st.markdown(f"📍 **Location:** `{lat:.5f}°N, {lon:.5f}°W`")

        st.markdown("---")
        st.markdown(
            "**Rank:** #{rank} of {total} intersections".format(
                rank=selected_idx + 1, total=len(top_nodes)
            )
        )

        st.markdown("---")
        st.markdown("### 🗺️ Map Display")
        n_show = st.slider(
            "High-risk intersections on map",
            min_value=len(top_nodes),
            max_value=500,
            value=max(len(top_nodes), 20),
            step=10,
            help=(
                "Slide to control how many of the highest DR-ISI "
                "intersections are shown on the map. "
                "XAI-highlighted nodes (orange/red) are always visible; "
                "additional nodes appear as small yellow dots."
            ),
        )

        # Attribution information
        st.markdown("---")
        st.caption(
            "DR-ISI Traffic Risk Prediction System\n\n"
            "Master's Thesis — ITS\n\n"
            "Phase 6: Interactive Glass-Box Dashboard"
        )

    # =================================================================
    # Main Page — Header
    # =================================================================

    st.markdown(
        "# 🚦 DR-ISI Traffic Risk Monitor — Dallas, TX\n"
        "*Spatio-Temporal Graph Neural Network · Explainable AI Dashboard*"
    )

    # =================================================================
    # Section A: Geospatial Risk Map (PyDeck)
    # =================================================================

    st.markdown("## 🗺️ Section A: Geospatial Risk Map")
    st.markdown(
        "The map shows the **selected intersection** (🔴 red), "
        "**other XAI-highlighted nodes** (🟠 orange), and "
        "**additional high-risk intersections** (🟡 yellow) up to the "
        "count set by the sidebar slider. "
        "Arc lines represent the GNNExplainer influence graph — "
        "road segments the model considers important."
    )

    # Load subgraph for the selected node
    subgraph_data = load_subgraph(node_id)
    if subgraph_data is None:
        st.warning(
            f"⚠️ Subgraph data not found for Node {node_id}. "
            "The influence graph will not be displayed."
        )

    if node_positions is None:
        st.warning(
            "⚠️ Node position data not found. "
            "Subgraph edges cannot be plotted."
        )

    # Load all node risks and compute the extended node list for the map
    all_node_risks = load_all_node_risks()
    extended_nodes = all_node_risks[:n_show] if all_node_risks is not None else None

    # Build and render the map
    try:
        deck = build_risk_map(
            selected_node, top_nodes, subgraph_data, node_positions,
            extended_nodes=extended_nodes,
        )
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Map rendering failed: {e}")

    # =================================================================
    # Section B: Glass-Box Explanations (Tabs)
    # =================================================================

    st.markdown("---")
    st.markdown("## 🔍 Section B: Glass-Box Explanations")
    st.markdown(
        f"XAI insights for **Node {node_id}** — explaining *why* the "
        "model predicts high risk at this intersection."
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Feature Drivers (SHAP)",
        "⏱️ Temporal Evolution (Captum)",
        "🌐 Global Model Insights",
    ])

    # ------------------------------------------------------------------
    # Tab 1: Local SHAP Feature Drivers
    # ------------------------------------------------------------------
    with tab1:
        st.markdown(
            "### What conditions caused this risk?\n\n"
            "Local SHAP values show which input features drove the "
            "model's prediction for this specific intersection."
        )

        local_shap = load_local_shap(node_id)
        if local_shap is not None:
            try:
                fig = build_local_shap_chart(local_shap)
                st.plotly_chart(fig, use_container_width=True)

                # Interpretive note
                shap_vals = local_shap["local_shap"]
                top_feature = max(shap_vals, key=shap_vals.get)
                st.info(
                    f"💡 **Insight:** The strongest driver for Node "
                    f"{node_id} is **{FEATURE_LABELS.get(top_feature, top_feature)}** "
                    f"(SHAP = {shap_vals[top_feature]:.6f})."
                )
            except Exception as e:
                st.error(f"Chart rendering failed: {e}")
        else:
            st.warning(
                f"⚠️ Local SHAP data not found for Node {node_id}. "
                "Run `python src/generate_insights.py` to generate."
            )

    # ------------------------------------------------------------------
    # Tab 2: Temporal Evolution (Captum)
    # ------------------------------------------------------------------
    with tab2:
        st.markdown(
            "### When did the risk start building up?\n\n"
            "Captum IntegratedGradients shows the importance of each "
            "of the 24 hourly timesteps in the lookback window. "
            "Higher values near t-0 (most recent) indicate that "
            "recent activity drives the prediction."
        )

        temporal = load_temporal_profile(node_id)
        if temporal is not None:
            try:
                fig = build_temporal_chart(temporal)
                st.plotly_chart(fig, use_container_width=True)

                # Interpretive note about peak hour
                peak = temporal["peak_hour_offset"]
                hours_ago = 23 - peak
                if hours_ago <= 2:
                    interpretation = (
                        "The model relies primarily on the **most recent "
                        "1–2 hours** of activity — recency is critical."
                    )
                elif hours_ago <= 6:
                    interpretation = (
                        f"The model focuses on activity from "
                        f"**~{hours_ago} hours ago**, suggesting a "
                        "short-term accumulation effect."
                    )
                else:
                    interpretation = (
                        f"The model looks back **{hours_ago} hours** "
                        "— risk at this node is driven by sustained, "
                        "longer-term patterns."
                    )

                st.info(f"💡 **Insight:** Peak at offset {peak} "
                        f"(~{hours_ago}h ago). {interpretation}")

            except Exception as e:
                st.error(f"Chart rendering failed: {e}")
        else:
            st.warning(
                f"⚠️ Temporal profile not found for Node {node_id}. "
                "Run `python src/generate_insights.py` to generate."
            )

    # ------------------------------------------------------------------
    # Tab 3: Global Model Insights
    # ------------------------------------------------------------------
    with tab3:
        st.markdown(
            "### How does the AI make decisions city-wide?\n\n"
            "Global SHAP feature importance (averaged across all "
            "explained nodes and samples) shows which input features "
            "the model relies on most when predicting crash risk "
            "across the entire Dallas road network."
        )

        global_shap = load_global_shap()
        if global_shap is not None:
            try:
                fig = build_global_shap_chart(global_shap)
                st.plotly_chart(fig, use_container_width=True)

                # Interpretive note
                gi = global_shap["global_importance"]
                top_feat = max(gi, key=gi.get)
                st.info(
                    f"💡 **Insight:** `{FEATURE_LABELS.get(top_feat, top_feat)}` "
                    f"is the globally dominant feature (SHAP = "
                    f"{gi[top_feat]:.6f}), meaning the model relies "
                    "primarily on historical crash frequency to assess "
                    "risk. Temporal features (hour/day cycles) play a "
                    "secondary role."
                )
            except Exception as e:
                st.error(f"Chart rendering failed: {e}")
        else:
            st.warning(
                "⚠️ Global SHAP data not found. "
                "Run `python src/generate_insights.py` to generate."
            )

    # =================================================================
    # Footer
    # =================================================================

    st.markdown("---")
    st.caption(
        "🚦 DR-ISI Traffic Risk Prediction System · "
        "Phase 6: Interactive Glass-Box Dashboard · "
        "Spatio-Temporal GNN (HybridSTGNN) · "
        "XAI: SHAP + GNNExplainer + Captum"
    )


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    main()
