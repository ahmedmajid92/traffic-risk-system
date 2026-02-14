"""
Pipeline Showcase — Gradio UI
===============================

Interactive dashboard showing inputs/outputs of all 4 project phases.

Usage:
    python showcase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed")
RAW_DATA_PATH = Path("data/raw/US_Accidents_March23.csv")
MODEL_DIR = Path("models")

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

            # ── Overview ──
            with gr.Tab("📋 Overview", elem_classes=["phase-tab"]):
                md = "## Project Overview\n\n"
                md += "| Phase | Description | Key Output | File(s) |\n"
                md += "|-------|-------------|------------|--------|\n"
                md += "| 1 | Data Engineering & NLP | 115k annotated crashes | `data_loader.py` |\n"
                md += "| 2 | Graph Construction | 24,697-node road graph | `build_graph_pipeline.py` |\n"
                md += "| 3 | Temporal Sequences | 49k sliding windows | `temporal_processor.py` |\n"
                md += "| 4 | Model & Training | HybridSTGNN (157k params) | `model_architecture.py`, `train_model.py` |\n"
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
                md += "```\n"
                gr.Markdown(md)

    return app


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(inbrowser=True)
