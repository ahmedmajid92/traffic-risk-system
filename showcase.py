"""
Pipeline Showcase — Gradio UI
===============================

Interactive dashboard showing inputs/outputs of all 5 project phases.

Usage:
    python showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed")
RAW_DATA_PATH = Path("data/raw/US_Accidents_March23.csv")
MODEL_DIR = Path("models")
XAI_DIR = DATA_DIR / "xai_artifacts"
FIG_DIR = Path("reports/figures")

sys.path.insert(0, "src")


# ── Helpers ──────────────────────────────────────────────────────────

def file_size_str(path: Path) -> str:
    if not path.exists():
        return "not found"
    size = path.stat().st_size
    if size > 1e9:
        return f"{size / 1e9:.2f} GB"
    if size > 1e6:
        return f"{size / 1e6:.1f} MB"
    return f"{size / 1e3:.0f} KB"

def plot_graph_nodes(x, y):
    """Simple scatter plot of node coordinates."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, s=1, c='blue', alpha=0.3)
    ax.set_title("Dallas Road Network (Nodes)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis("equal")
    return fig

# =====================================================================
# Phase 1
# =====================================================================

def build_phase1() -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    csv_path = DATA_DIR / "dallas_crashes_annotated.csv"
    md = "## Phase 1: Data Engineering & NLP Annotation\n\n"
    md += "**File:** `src/data_loader.py`\n\n"
    md += "**Pipeline:** Chunk read (~3 GB) → Dallas filter → Clean → NLP distraction annotation\n\n"

    raw_sample = pd.DataFrame()
    annotated_sample = pd.DataFrame()

    # Try to load raw sample if available
    if RAW_DATA_PATH.exists():
        try:
             raw_sample = pd.read_csv(RAW_DATA_PATH, nrows=5)
             raw_cols = ["ID", "Severity", "Start_Time", "Start_Lat", "Start_Lng", "Description"]
             raw_sample = raw_sample[[c for c in raw_cols if c in raw_sample.columns]]
        except Exception:
            pass
    elif csv_path.exists():
        # Fallback: describe what raw data would look like
        pass

    if not csv_path.exists():
        return md + "> ⚠️ Output file not found.\n",  "", raw_sample, annotated_sample

    df = pd.read_csv(csv_path, nrows=5)
    total = pd.read_csv(csv_path, usecols=[0]).shape[0]
    dates = pd.read_csv(csv_path, usecols=["Start_Time"], nrows=5000)

    md += f"**Output:** `{csv_path.name}` ({file_size_str(csv_path)})\n\n"
    md += "### Key Statistics\n\n"
    md += f"| Metric | Value |\n|--------|-------|\n"
    md += f"| Total crashes | {total:,} |\n"
    md += f"| Columns | {len(df.columns)} |\n"
    md += f"| Date range | {dates['Start_Time'].min()[:10]} → {dates['Start_Time'].max()[:10]} |\n"

    if "Is_Distracted" in df.columns:
        full_dist = pd.read_csv(csv_path, usecols=["Is_Distracted"])
        dist_pct = full_dist["Is_Distracted"].mean() * 100
        md += f"| Distracted crashes | {dist_pct:.1f}% |\n"

    md += "\n### Severity Distribution\n\n"
    if "Severity" in df.columns:
        sev = pd.read_csv(csv_path, usecols=["Severity"])["Severity"].value_counts().sort_index()
        md += "| Severity | Count | Pct |\n|----------|-------|-----|\n"
        for s, c in sev.items():
            md += f"| {s} | {c:,} | {c / sev.sum() * 100:.1f}% |\n"

    sample_cols = ["ID", "Severity", "Start_Time", "Start_Lat", "Start_Lng", "Is_Distracted"]
    annotated_sample = df[[c for c in sample_cols if c in df.columns]]

    return md, "### Raw Data Sample", raw_sample, annotated_sample


# =====================================================================
# Phase 2
# =====================================================================

def build_phase2() -> tuple[str, plt.Figure | None]:
    md = "## Phase 2: Spatial Graph Construction & DR-ISI\n\n"
    md += "**Files:** `src/graph_builder.py`, `src/crash_mapper.py`, `src/drisi_calculator.py`, `src/pyg_converter.py`, `src/build_graph_pipeline.py`\n\n"

    graph_path = DATA_DIR / "processed_graph_data.pt"
    graphml_path = DATA_DIR / "dallas_drive_net.graphml"
    mapped_path = DATA_DIR / "crashes_mapped.csv"

    md += "### Output Files\n\n"
    md += "| File | Size |\n|------|------|\n"
    md += f"| `{graph_path.name}` | {file_size_str(graph_path)} |\n"
    md += f"| `{graphml_path.name}` | {file_size_str(graphml_path)} |\n"
    md += f"| `{mapped_path.name}` | {file_size_str(mapped_path)} |\n"

    fig = None
    if graph_path.exists():
        import torch
        data = torch.load(graph_path, map_location="cpu", weights_only=False)

        md += "\n### Graph Statistics\n\n"
        md += "| Metric | Value |\n|--------|-------|\n"
        if hasattr(data, "x"):
            md += f"| Nodes (intersections) | {data.x.shape[0]:,} |\n"
            md += f"| Node features | {data.x.shape[1]} |\n"
        if hasattr(data, "edge_index"):
            md += f"| Edges (road segments) | {data.edge_index.shape[1]:,} |\n"
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            md += f"| Edge features | {data.edge_attr.shape[1]} |\n"

        md += "\n### Node Feature Statistics\n\n"
        features = ["degree", "betweenness_centrality", "avg_speed", "bearing_entropy", "traffic_signal"]
        md += "| Feature | Mean | Std | Min | Max |\n|---------|------|-----|-----|-----|\n"
        for i, f in enumerate(features):
            s = data.x[:, i]
            md += f"| {f} | {s.mean():.3f} | {s.std():.3f} | {s.min():.3f} | {s.max():.3f} |\n"
        
        # Plot graph if pos exists
        if hasattr(data, "pos"):
             pos = data.pos.numpy()
             # pos is likely metric (UTM), checking range
             # if huge values, assume valid metric, otherwise lat/lon
             x, y = pos[:, 0], pos[:, 1]
             fig = plot_graph_nodes(x, y)
        else:
             md += "\n> ⚠️ No node position data found to plot.\n"

    else:
        md += "\n> ⚠️ Graph data not found.\n"

    return md, fig


# =====================================================================
# Phase 3
# =====================================================================

def build_phase3() -> str:
    md = "## Phase 3: Spatio-Temporal Sequence Generation\n\n"
    md += "**File:** `src/temporal_processor.py`\n\n"
    md += "**Config:** Window = 24 hours, Features = (crash_count, hour_sin, hour_cos, dow_sin, dow_cos), Target = log(1 + Σ EPDO×crashes)\n\n"
    
    md += "### Model Input Architecture (Mermaid Diagram)\n\n"
    md += "```mermaid\n"
    md += "graph TD\n"
    md += "    A(\"Raw Crash Data\") --> B(\"Sparse Hourly Aggregation\")\n"
    md += "    B --> C(\"Sliding Window Generator\")\n"
    md += "    C --> D(\"Model Input Tensor\")\n"
    md += "    subgraph InputShape\n"
    md += "        D -- \"(B, 24, N, 5)\" --> E(\"Batch: B <br/> Window: 24h <br/> Nodes: 24,697 <br/> Feats: 5\")\n"
    md += "    end\n"
    md += "```\n\n"

    signal_path = DATA_DIR / "temporal_signal.pt"
    train_path = DATA_DIR / "train_dataset.pt"
    test_path = DATA_DIR / "test_dataset.pt"

    md += "### Output Files\n\n"
    md += "| File | Size |\n|------|------|\n"
    md += f"| `{signal_path.name}` | {file_size_str(signal_path)} |\n"
    md += f"| `{train_path.name}` | {file_size_str(train_path)} |\n"
    md += f"| `{test_path.name}` | {file_size_str(test_path)} |\n"

    try:
        from temporal_processor import load_dataset
        ds, meta = load_dataset(DATA_DIR, split="train")
        x_sample, y_sample = ds[0]

        md += "\n### Sample Window Stats\n\n"
        md += "| Property | Value |\n|----------|-------|\n"
        md += f"| X shape (1 window) | `{list(x_sample.shape)}` → (W=24, N, F=5) |\n"
        md += f"| Y shape (1 target) | `{list(y_sample.shape)}` → (N,) |\n"
        md += f"| Total train windows | {len(ds):,} |\n"
        nz = (y_sample > 0).sum().item()
        md += f"| Non-zero targets (sample) | {nz} / {y_sample.shape[0]} ({nz / y_sample.shape[0] * 100:.2f}%) |\n"
    except Exception as e:
        md += f"\n> ⚠️ Could not load dataset: {e}\n"

    return md


# =====================================================================
# Phase 4
# =====================================================================

def build_phase4() -> str:
    md = "## Phase 4: Model Architecture & Training\n\n"
    md += "**Files:** `src/model_architecture.py`, `src/train_model.py`, `src/evaluate_model.py`\n\n"

    md += "### Architecture: HybridSTGNN (Mermaid Diagram)\n\n"
    md += "```mermaid\n"
    md += "graph TD\n"
    md += "    IN(\"Input Tensor <br/> (B, 24, N, 5)\") --> GCN(\"Spatial Encoder <br/> 2x GCNConv\")\n"
    md += "    GCN --> EMB(\"Spatial Embeddings <br/> (B, N, 24, 128)\")\n"
    md += "    EMB --> LSTM(\"Temporal Core <br/> LSTM\")\n"
    md += "    LSTM --> HID(\"Hidden State <br/> (B, N, 128)\")\n"
    md += "    HID --> MLP(\"Risk Decoder <br/> MLP\")\n"
    md += "    MLP --> OUT(\"Risk Scores <br/> (B, N)\")\n"
    md += "```\n\n"

    model_path = MODEL_DIR / "best_stgnn_model.pth"
    ckpt_path = MODEL_DIR / "checkpoint.pt"

    md += "### Saved Files\n\n"
    md += "| File | Size |\n|------|------|\n"
    md += f"| `{model_path.name}` | {file_size_str(model_path)} |\n"
    md += f"| `{ckpt_path.name}` | {file_size_str(ckpt_path)} |\n"

    if model_path.exists():
        import torch
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        total_params = sum(v.numel() for v in state.values())
        md += f"\n**Total parameters:** {total_params:,}\n"

    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        md += "\n### Training Progress\n\n"
        md += "| Metric | Value |\n|--------|-------|\n"
        md += f"| Completed epochs | {ckpt.get('epoch', '?')} |\n"
        bvl = ckpt.get("best_val_loss", None)
        if bvl is not None:
            md += f"| Best val loss | {bvl:.6f} |\n"

    md += "\n### Commands\n\n"
    md += "```bash\n"
    md += "# Train for 5 epochs\n"
    md += "python src/train_model.py --epochs 5\n\n"
    md += "# Resume training\n"
    md += "python src/train_model.py --resume --epochs 15\n\n"
    md += "# Evaluate trained model\n"
    md += "python src/evaluate_model.py\n"
    md += "```\n"

    return md


# =====================================================================
# Phase 5
# =====================================================================

def build_phase5_md() -> str:
    md = "## Phase 5: Explainability & Insights\n\n"
    md += "**Files:** `src/xai/adapters.py`, `src/xai/explainers.py`, `src/generate_insights.py`, `src/visualization/static_plots.py`\n\n"
    md += "**XAI Methods:** SHAP (GradientExplainer), GNNExplainer, Captum (IntegratedGradients)\n\n"

    # --- Summary: top risk nodes ---
    summary_path = XAI_DIR / "summary.json"
    if not summary_path.exists():
        return md + "> ⚠️ XAI artifacts not found. Run `python src/generate_insights.py` first.\n"

    with open(summary_path) as f:
        summary = json.load(f)

    top_nodes = summary["top_risk_nodes"]
    md += "### Top-5 High-Risk Intersections\n\n"
    md += "| Rank | Node ID | Mean Risk | Max Risk |\n"
    md += "|------|---------|-----------|----------|\n"
    for i, node in enumerate(top_nodes):
        md += f"| {i+1} | {node['node_idx']} | {node['mean_risk']:.4f} | {node['max_risk']:.4f} |\n"

    # --- SHAP global importance ---
    shap_path = XAI_DIR / "global_feature_importance.json"
    if shap_path.exists():
        with open(shap_path) as f:
            shap_data = json.load(f)
        gi = shap_data["global_importance"]
        md += "\n### SHAP Global Feature Importance\n\n"
        md += "| Feature | Mean |SHAP| |\n|---------|------------|\n"
        for name, val in sorted(gi.items(), key=lambda x: -x[1]):
            bar_len = int(val / max(gi.values()) * 20)
            bar = "█" * bar_len
            md += f"| {name} | {val:.6f} {bar} |\n"
        md += f"\n> **Insight:** `crash_count` dominates — the model relies primarily on historical crash frequency.\n"

    # --- Temporal profiles ---
    md += "\n### Temporal Attribution (Captum IntegratedGradients)\n\n"
    md += "Shows which of the 24 hourly timesteps contribute most to each node's prediction.\n\n"
    for node in top_nodes[:3]:
        nid = node["node_idx"]
        tp_path = XAI_DIR / f"temporal_profile_{nid}.json"
        if tp_path.exists():
            with open(tp_path) as f:
                tp = json.load(f)
            peak = tp["peak_hour_offset"]
            md += f"- **Node {nid}**: Peak at hour offset **{peak}** "
            if peak >= 22:
                md += "(most recent hours dominate)\n"
            else:
                md += f"(hour {peak} of 24 is most important)\n"

    # --- GNNExplainer ---
    md += "\n### GNNExplainer — Spatial Structure\n\n"
    md += "Identifies which road segments (edges) are most influential for each node's prediction.\n\n"
    for node in top_nodes[:3]:
        nid = node["node_idx"]
        sg_path = XAI_DIR / f"subgraph_{nid}.json"
        if sg_path.exists():
            with open(sg_path) as f:
                sg = json.load(f)
            n_sub = len(sg["subgraph_nodes"])
            n_edges = len(sg["top_k_edges"])
            top_weight = sg["top_k_edges"][0]["weight"] if sg["top_k_edges"] else 0
            md += f"- **Node {nid}**: {n_sub} connected nodes, top edge weight = {top_weight:.4f}\n"

    # --- Output files ---
    md += "\n### Generated Artifacts\n\n"
    md += "| File | Size |\n|------|------|\n"
    if XAI_DIR.exists():
        for f in sorted(XAI_DIR.glob("*.json")):
            md += f"| `{f.name}` | {file_size_str(f)} |\n"
    if FIG_DIR.exists():
        for f in sorted(FIG_DIR.glob("*.png")):
            md += f"| `{f.name}` | {file_size_str(f)} |\n"

    md += "\n### Commands\n\n"
    md += "```bash\n"
    md += "# Generate all insights (100 samples, GPU, ~10 min)\n"
    md += "python src/generate_insights.py\n\n"
    md += "# Skip SHAP (faster, ~2 min)\n"
    md += "python src/generate_insights.py --skip-shap\n\n"
    md += "# More samples for higher confidence\n"
    md += "python src/generate_insights.py --n-samples 500 --stride 1\n\n"
    md += "# Regenerate plots only\n"
    md += "python src/visualization/static_plots.py\n"
    md += "```\n"

    return md


def build_phase5_plots() -> list[plt.Figure]:
    """Load SHAP bar chart and temporal plots as matplotlib figures."""
    figs = []

    # SHAP summary
    shap_path = XAI_DIR / "global_feature_importance.json"
    if shap_path.exists():
        with open(shap_path) as f:
            data = json.load(f)
        imp = data["global_importance"]
        features = list(imp.keys())
        values = list(imp.values())
        sorted_idx = np.argsort(values)
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))
        ax.barh(features, values, color=colors)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Global Feature Importance (SHAP)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        figs.append(("SHAP Feature Importance", fig))

    # Temporal profiles
    summary_path = XAI_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            nodes = json.load(f)["top_risk_nodes"]
        for node in nodes[:3]:
            nid = node["node_idx"]
            tp_path = XAI_DIR / f"temporal_profile_{nid}.json"
            if tp_path.exists():
                with open(tp_path) as f:
                    tp = json.load(f)
                norm = np.array(tp["temporal_importance_normalized"])
                hours = np.arange(len(norm))
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(hours, norm, color="steelblue", alpha=0.8)
                ax.axvline(tp["peak_hour_offset"], color="red",
                           linestyle="--", alpha=0.7,
                           label=f"Peak: t-{tp['peak_hour_offset']}")
                ax.set_xlabel("Hour Offset (0 = oldest, 23 = most recent)")
                ax.set_ylabel("Normalized Importance")
                ax.set_title(f"Temporal Attribution — Node {nid}")
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                figs.append((f"Temporal — Node {nid}", fig))

    return figs


# =====================================================================
# Phase 6
# =====================================================================

def build_phase6() -> str:
    md = "## Phase 6: Interactive Glass-Box Dashboard\n\n"
    md += "**File:** `app.py`\n\n"
    md += "**Framework:** Streamlit + PyDeck + Plotly\n\n"

    md += "### Dashboard Layout\n\n"
    md += "```mermaid\ngraph TD\n"
    md += "    A(\"Sidebar\") --> B(\"Node Selector<br/>Top-5 from summary.json\")\n"
    md += "    B --> C(\"Node Stats<br/>Mean/Max Risk + Coordinates\")\n"
    md += "    D(\"Section A\") --> E(\"PyDeck Map<br/>ScatterplotLayer + ArcLayer\")\n"
    md += "    F(\"Section B\") --> G(\"Tab 1: Local SHAP\")\n"
    md += "    F --> H(\"Tab 2: Temporal Captum\")\n"
    md += "    F --> I(\"Tab 3: Global SHAP\")\n"
    md += "```\n\n"

    md += "### Technology Stack\n\n"
    md += "| Component | Technology | Purpose |\n"
    md += "|-----------|-----------|---------|\n"
    md += "| Web Framework | Streamlit | Reactive dashboard UI |\n"
    md += "| Geospatial Map | PyDeck (Deck.gl) | 3D map with scatter + arc layers |\n"
    md += "| Charts | Plotly | Interactive SHAP & temporal charts |\n"
    md += "| CRS Conversion | pyproj | UTM Zone 14N → WGS84 for map pins |\n"

    md += "\n### Coordinate Conversion\n\n"
    md += "All node positions from Phase 2 are in **UTM Zone 14N (EPSG:32614)** — metric coordinates.\n"
    md += "PyDeck requires **WGS84 (EPSG:4326)** — geographic degrees. "
    md += "The dashboard converts at load time using `pyproj.Transformer`.\n\n"

    md += "### Input Artifacts\n\n"
    md += "| Artifact | XAI Source | Dashboard Section |\n"
    md += "|----------|-----------|-------------------|\n"
    md += "| `summary.json` | Model inference | Sidebar + Map |\n"
    md += "| `local_shap_{id}.json` | SHAP | Tab 1: Feature Drivers |\n"
    md += "| `temporal_profile_{id}.json` | Captum IG | Tab 2: Temporal |\n"
    md += "| `global_feature_importance.json` | SHAP | Tab 3: Global Insights |\n"
    md += "| `subgraph_{id}.json` | GNNExplainer | Map ArcLayer |\n"

    md += "\n### Commands\n\n"
    md += "```bash\n"
    md += "# Launch the interactive dashboard\n"
    md += "streamlit run app.py\n\n"
    md += "# Launch on a custom port\n"
    md += "streamlit run app.py --server.port 8502\n\n"
    md += "# Install Phase 6 dependencies\n"
    md += "pip install streamlit pydeck plotly pyproj\n"
    md += "```\n"

    return md


# =====================================================================
# Phase 7
# =====================================================================

def build_phase7() -> tuple[str, plt.Figure | None]:
    md = "## Phase 7: Baseline Benchmarking & Evaluation\n\n"
    md += "**File:** `src/evaluate_baselines.py`\n\n"
    md += "**Goal:** Prove that the ST-GNN architecture outperforms standard baselines.\n\n"

    md += "### Theoretical Design\n\n"
    md += "| Model | Temporal? | Graph? | What It Tests |\n"
    md += "|-------|-----------|--------|---------------|\n"
    md += "| XGBoost | \u2717 (flat) | \u2717 (per-node) | Tabular ML upper bound |\n"
    md += "| LSTM-Only | \u2713 (24-h seq) | \u2717 (no graph) | Value of temporal modelling |\n"
    md += "| HybridSTGNN | \u2713 (LSTM) | \u2713 (GCN) | Full spatio-temporal |\n"

    md += "\n### Data Flattening\n\n"
    md += "```mermaid\ngraph LR\n"
    md += "    A(\"Original: B, 24, N, 5\") --> B(\"XGBoost: samples\u00d7N, 120\")\n"
    md += "    A --> C(\"LSTM: samples\u00d7N, 24, 5\")\n"
    md += "    A --> D(\"ST-GNN: B, 24, N, 5 + edges\")\n"
    md += "```\n\n"

    # Load results if available
    metrics_path = Path("reports/metrics.json")
    fig = None

    if metrics_path.exists():
        with open(metrics_path) as f:
            results = json.load(f)

        md += "### Results\n\n"
        md += "| Model | RMSE | MAE |\n"
        md += "|-------|------|-----|\n"
        for name, m in results.items():
            md += f"| {name} | {m['rmse']:.6f} | {m['mae']:.6f} |\n"

        best = min(results, key=lambda k: results[k]["rmse"])
        md += f"\n> \U0001f3c6 **Winner:** {best} (lowest RMSE)\n"
    else:
        md += "> \u26a0\ufe0f Results not yet generated. Run `python src/evaluate_baselines.py` first.\n"

    # Load comparison chart if available
    chart_path = Path("reports/figures/model_comparison.png")
    if chart_path.exists():
        try:
            img = plt.imread(str(chart_path))
            fig_obj, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(img)
            ax.axis("off")
            fig = fig_obj
        except Exception:
            pass

    md += "\n### Commands\n\n"
    md += "```bash\n"
    md += "# Run the full benchmark\n"
    md += "python src/evaluate_baselines.py\n\n"
    md += "# Quick test (100 nodes)\n"
    md += "python src/evaluate_baselines.py --n-nodes 100 --stride 12\n"
    md += "```\n"

    return md, fig


# =====================================================================
# Gradio App
# =====================================================================

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="Traffic Risk System — Pipeline Showcase",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            .phase-tab { min-height: 400px; }
            .main-title { text-align: center; margin-bottom: 0; }
            .sub-title { text-align: center; color: #666; margin-top: 0; }
        """,
    ) as app:
        gr.Markdown(
            "# 🚦 Traffic Risk Prediction System\n"
            "### Pipeline Showcase — Inputs & Outputs of Each Phase",
            elem_classes=["main-title"],
        )

        with gr.Tabs():
            # ── Phase 1 ──
            data_md, raw_title, raw_df, ann_df = build_phase1()
            with gr.Tab("📊 Phase 1 — Data & NLP", elem_classes=["phase-tab"]):
                 gr.Markdown(data_md)
                 if not raw_df.empty:
                     gr.Markdown("### Raw Data Sample (Before)")
                     gr.DataFrame(raw_df, interactive=False)
                 gr.Markdown("### Annotated Data Sample (After)")
                 gr.DataFrame(ann_df, interactive=False)

            # ── Phase 2 ──
            graph_md, graph_fig = build_phase2()
            with gr.Tab("🗺️ Phase 2 — Graph & DR-ISI", elem_classes=["phase-tab"]):
                gr.Markdown(graph_md)
                if graph_fig is not None:
                    gr.Plot(graph_fig, label="Dallas Road Network Graph")

            # ── Phase 3 ──
            with gr.Tab("⏱️ Phase 3 — Temporal", elem_classes=["phase-tab"]):
                gr.Markdown(build_phase3())

            # ── Phase 4 ──
            with gr.Tab("🧠 Phase 4 — Model", elem_classes=["phase-tab"]):
                gr.Markdown(build_phase4())

            # ── Phase 5 ──
            phase5_plots = build_phase5_plots()
            with gr.Tab("🔍 Phase 5 — XAI Insights", elem_classes=["phase-tab"]):
                gr.Markdown(build_phase5_md())
                for label, fig in phase5_plots:
                    gr.Plot(fig, label=label)

            # ── Phase 6 ──
            with gr.Tab("🖥️ Phase 6 — Dashboard", elem_classes=["phase-tab"]):
                gr.Markdown(build_phase6())

            # ── Phase 7 ──
            phase7_md, phase7_fig = build_phase7()
            with gr.Tab("📊 Phase 7 — Baselines", elem_classes=["phase-tab"]):
                gr.Markdown(phase7_md)
                if phase7_fig is not None:
                    gr.Plot(phase7_fig, label="Model Comparison")

            # ── Overview ──
            with gr.Tab("📋 Overview", elem_classes=["phase-tab"]):
                md = "## Project Overview\n\n"
                md += "| Phase | Description | Key Output | File(s) |\n"
                md += "|-------|-------------|------------|--------|\n"
                md += "| 1 | Data Engineering & NLP | 115k annotated crashes | `data_loader.py` |\n"
                md += "| 2 | Graph Construction | 24,697-node road graph | `build_graph_pipeline.py` |\n"
                md += "| 3 | Temporal Sequences | 49k sliding windows | `temporal_processor.py` |\n"
                md += "| 4 | Model & Training | HybridSTGNN (157k params) | `model_architecture.py`, `train_model.py` |\n"
                md += "| 5 | Explainability & Insights | SHAP, GNNExplainer, Captum | `generate_insights.py` |\n"
                md += "| 6 | Interactive Dashboard | Streamlit + PyDeck + Plotly | `app.py` |\n"
                md += "| 7 | Baseline Benchmarking | XGBoost + LSTM vs ST-GNN | `evaluate_baselines.py` |\n"
                md += "\n### Data Flow\n\n"
                md += "```\n"
                md += "US_Accidents_March23.csv (3 GB)\n"
                md += "    │  Phase 1: filter + NLP\n"
                md += "    ▼\n"
                md += "dallas_crashes_annotated.csv (115k crashes)\n"
                md += "    │  Phase 2: graph + snapping + DR-ISI\n"
                md += "    ▼\n"
                md += "processed_graph_data.pt (24,697 nodes × 71,392 edges)\n"
                md += "    │  Phase 3: temporal sliding windows\n"
                md += "    ▼\n"
                md += "train/test_dataset.pt (49k windows of 24h × 24,697 nodes × 5 features)\n"
                md += "    │  Phase 4: GCN + LSTM + MLP\n"
                md += "    ▼\n"
                md += "best_stgnn_model.pth (157,697 parameters → per-node risk scores)\n"
                md += "    │  Phase 5: SHAP + GNNExplainer + Captum\n"
                md += "    ▼\n"
                md += "xai_artifacts/ (feature importance, subgraphs, temporal profiles)\n"
                md += "    │  Phase 6: Streamlit + PyDeck + Plotly\n"
                md += "    ▼\n"
                md += "app.py → Interactive Glass-Box Dashboard (localhost:8501)\n"
                md += "    │  Phase 7: XGBoost + LSTM baselines\n"
                md += "    ▼\n"
                md += "reports/metrics.json + model_comparison.png\n"
                md += "```\n"
                gr.Markdown(md)

    return app


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(inbrowser=True)
