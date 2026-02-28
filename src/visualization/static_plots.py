"""
Phase 5 — Static XAI Plots
==============================

Helper functions to generate PNG visualizations of XAI artifacts
for immediate validation and reporting.

Usage (standalone):
    python src/visualization/static_plots.py

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-28
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_shap_summary(
    importance_path: Path,
    save_path: Path,
) -> None:
    """
    Horizontal bar chart of global SHAP feature importance.

    Parameters
    ----------
    importance_path : Path to ``global_feature_importance.json``
    save_path : Path, e.g. ``reports/figures/shap_summary.png``
    """
    with open(importance_path) as f:
        data = json.load(f)

    importance = data["global_importance"]
    features = list(importance.keys())
    values = list(importance.values())

    # Sort by importance
    sorted_idx = np.argsort(values)
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, values, color=colors)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global Feature Importance (SHAP)")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → Saved SHAP summary plot: %s", save_path)


def plot_temporal_importance(
    temporal_path: Path,
    node_idx: int,
    save_path: Path,
) -> None:
    """
    Line chart of temporal attribution across the 24-hour window.

    Parameters
    ----------
    temporal_path : Path to ``temporal_profile_{node_id}.json``
    node_idx : int, for title labelling
    save_path : Path, e.g. ``reports/figures/temporal_importance_123.png``
    """
    with open(temporal_path) as f:
        data = json.load(f)

    raw = np.array(data["temporal_importance_raw"])
    normalized = np.array(data["temporal_importance_normalized"])
    peak = data["peak_hour_offset"]
    hours = np.arange(len(raw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw attribution
    ax1.bar(hours, raw, color="steelblue", alpha=0.8)
    ax1.axvline(peak, color="red", linestyle="--", alpha=0.7, label=f"Peak: t-{peak}")
    ax1.set_xlabel("Hour Offset (0 = oldest)")
    ax1.set_ylabel("Mean |Attribution|")
    ax1.set_title(f"Temporal Attribution — Node {node_idx} (Raw)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Normalized attribution
    ax2.plot(hours, normalized, marker="o", color="darkorange", linewidth=2)
    ax2.fill_between(hours, normalized, alpha=0.2, color="orange")
    ax2.axvline(peak, color="red", linestyle="--", alpha=0.7, label=f"Peak: t-{peak}")
    ax2.set_xlabel("Hour Offset (0 = oldest)")
    ax2.set_ylabel("Normalized Importance")
    ax2.set_title(f"Temporal Attribution — Node {node_idx} (Normalized)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → Saved temporal plot: %s", save_path)


# =====================================================================
# Standalone runner
# =====================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    artifact_dir = Path("data/processed/xai_artifacts")
    fig_dir = Path("reports/figures")

    # SHAP summary
    shap_path = artifact_dir / "global_feature_importance.json"
    if shap_path.exists():
        plot_shap_summary(shap_path, fig_dir / "shap_summary.png")
    else:
        print(f"⚠️  {shap_path} not found. Run generate_insights.py first.")

    # Temporal profiles
    for tp in sorted(artifact_dir.glob("temporal_profile_*.json")):
        node_id = tp.stem.replace("temporal_profile_", "")
        plot_temporal_importance(tp, int(node_id), fig_dir / f"temporal_importance_{node_id}.png")
        print(f"  ✓ Plotted temporal importance for node {node_id}")

    print("Done.")
