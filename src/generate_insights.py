"""
Phase 5 — Generate Insights (Orchestrator)
=============================================

End-to-end XAI pipeline:
    1. Load trained model + test data
    2. Identify Top 5 High-Risk Intersections
    3. Run SHAP (Global feature importance)
    4. For each Top-5 node:
       a. GNNExplainer → spatial subgraph
       b. Captum IntegratedGradients → temporal profile
       c. Local SHAP values

All artifacts are saved to ``data/processed/xai_artifacts/``.

Usage:
    python src/generate_insights.py
    python src/generate_insights.py --n-samples 50 --top-k 3

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-28
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_architecture import HybridSTGNN
from temporal_processor import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Step 1: Identify High-Risk Nodes
# =====================================================================

@torch.no_grad()
def find_high_risk_nodes(
    model: HybridSTGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> list[dict]:
    """
    Run inference on the test set and find the top-K highest predicted
    risk nodes (averaged across samples).

    Returns
    -------
    list of dicts with node_idx, mean_risk, max_risk
    """
    model.eval()
    all_preds = []

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch, edge_index)
        all_preds.append(pred.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()    # (n_samples, N)

    # Mean risk across samples per node
    mean_risk = preds.mean(axis=0)                  # (N,)
    max_risk = preds.max(axis=0)                    # (N,)

    # Top-K nodes by mean risk
    top_indices = np.argsort(mean_risk)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "node_idx": int(idx),
            "mean_risk": float(mean_risk[idx]),
            "max_risk": float(max_risk[idx]),
        })

    return results


# =====================================================================
# Step 2: Save JSON helper
# =====================================================================

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("  → Saved %s", path)


# =====================================================================
# Main Pipeline
# =====================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5: Generate XAI insights for HybridSTGNN",
    )
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/best_stgnn_model.pth"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/xai_artifacts"))
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Max test samples for explainers")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of high-risk nodes to explain")
    parser.add_argument("--stride", type=int, default=6,
                        help="Stride for loading test data")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP (slow, run separately)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    # --- Load model ---
    if not args.model_path.exists():
        logger.error("Model not found: %s", args.model_path)
        sys.exit(1)

    model = HybridSTGNN(in_features=5, hidden_dim=args.hidden_dim).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("✓ Model loaded: %s (%d params)",
                args.model_path, sum(p.numel() for p in model.parameters()))

    # --- Load test data ---
    test_dataset, test_meta = load_dataset(args.data_dir, split="test")
    edge_index = test_meta["edge_index"].to(device)
    node_positions = test_meta.get("pos", None)

    # Stride + cap
    indices = list(range(0, len(test_dataset), args.stride))
    if len(indices) > args.n_samples:
        indices = indices[:args.n_samples]
    subset = Subset(test_dataset, indices)

    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info("✓ Test data: %d samples (stride=%d)", len(subset), args.stride)

    num_nodes = test_dataset.num_nodes
    window_size = test_dataset.window_size

    # =================================================================
    # Step 1: Find Top-K High-Risk Intersections
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 1: Identifying Top-%d High-Risk Nodes", args.top_k)
    logger.info("=" * 60)

    top_nodes = find_high_risk_nodes(model, loader, edge_index, device, args.top_k)

    for i, node in enumerate(top_nodes):
        coords = ""
        if node_positions is not None:
            pos = node_positions[node["node_idx"]]
            node["longitude"] = float(pos[0])
            node["latitude"] = float(pos[1])
            coords = f" @ ({pos[0]:.5f}, {pos[1]:.5f})"
        logger.info("  #%d Node %d: mean_risk=%.4f, max_risk=%.4f%s",
                     i + 1, node["node_idx"], node["mean_risk"], node["max_risk"], coords)

    save_json({"top_risk_nodes": top_nodes}, output_dir / "summary.json")

    # Collect some test samples as tensors for explainers
    test_samples_x = []
    for i in range(min(args.n_samples, len(subset))):
        x, _ = subset[i]
        test_samples_x.append(x)

    # =================================================================
    # Step 2: SHAP Global Feature Importance
    # =================================================================
    if not args.skip_shap:
        logger.info("\n" + "=" * 60)
        logger.info("  STEP 2: SHAP Global Feature Importance")
        logger.info("=" * 60)

        from xai.explainers import explain_features_shap

        train_dataset, _ = load_dataset(args.data_dir, split="train")

        # Use first top node as the primary target for global SHAP
        primary_node = top_nodes[0]["node_idx"]

        shap_results = explain_features_shap(
            model=model,
            edge_index=edge_index,
            train_dataset=train_dataset,
            test_samples=test_samples_x[:10],  # small count for CPU-based SHAP
            target_node_idx=primary_node,
            device=device,
            window_size=window_size,
            num_nodes=num_nodes,
            num_features=5,
            background_k=25,
        )

        save_json(shap_results, output_dir / "global_feature_importance.json")

    # =================================================================
    # Step 3: Per-node Explanations
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 3: Per-Node Explanations (GNNExplainer + Captum)")
    logger.info("=" * 60)

    from xai.explainers import explain_structure_gnnexplainer, explain_temporal_captum

    # Use the first test sample for per-node explanations
    sample_x = test_samples_x[0]

    for i, node_info in enumerate(top_nodes):
        node_idx = node_info["node_idx"]
        logger.info("\n  ── Node %d/%d (idx=%d) ──",
                     i + 1, len(top_nodes), node_idx)

        # --- GNNExplainer ---
        try:
            gnn_result = explain_structure_gnnexplainer(
                model=model,
                x_input=sample_x,
                edge_index=edge_index,
                target_node_idx=node_idx,
                device=device,
            )
            save_json(gnn_result, output_dir / f"subgraph_{node_idx}.json")
        except Exception as e:
            logger.warning("  ⚠️ GNNExplainer failed for node %d: %s", node_idx, e)

        # --- Captum Temporal ---
        try:
            captum_result = explain_temporal_captum(
                model=model,
                x_input=sample_x,
                edge_index=edge_index,
                target_node_idx=node_idx,
                device=device,
                window_size=window_size,
                num_nodes=num_nodes,
                num_features=5,
            )
            save_json(captum_result, output_dir / f"temporal_profile_{node_idx}.json")
        except Exception as e:
            logger.warning("  ⚠️ Captum failed for node %d: %s", node_idx, e)

        # --- Local SHAP ---
        if not args.skip_shap:
            try:
                from xai.explainers import explain_features_shap

                local_result = explain_features_shap(
                    model=model,
                    edge_index=edge_index,
                    train_dataset=train_dataset,
                    test_samples=test_samples_x[:5],  # small for local
                    target_node_idx=node_idx,
                    device=device,
                    window_size=window_size,
                    num_nodes=num_nodes,
                    num_features=5,
                    background_k=15,
                )
                save_json(
                    {"local_shap": local_result["local_shap"], "node_idx": node_idx},
                    output_dir / f"local_shap_{node_idx}.json",
                )
            except Exception as e:
                logger.warning("  ⚠️ Local SHAP failed for node %d: %s", node_idx, e)

    # =================================================================
    # Step 4: Generate Static Plots
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 4: Generating Static Plots")
    logger.info("=" * 60)

    try:
        from visualization.static_plots import plot_shap_summary, plot_temporal_importance

        # SHAP bar chart
        shap_path = output_dir / "global_feature_importance.json"
        if shap_path.exists():
            plot_shap_summary(shap_path, Path("reports/figures/shap_summary.png"))

        # Temporal importance per top node
        for node_info in top_nodes:
            node_idx = node_info["node_idx"]
            tp_path = output_dir / f"temporal_profile_{node_idx}.json"
            if tp_path.exists():
                plot_temporal_importance(
                    tp_path, node_idx,
                    Path(f"reports/figures/temporal_importance_{node_idx}.png"),
                )
    except Exception as e:
        logger.warning("  ⚠️ Plot generation failed: %s", e)

    # =================================================================
    # Done
    # =================================================================
    elapsed = time.perf_counter() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("  ✓ PHASE 5 COMPLETE — %.0fs elapsed", elapsed)
    logger.info("  Artifacts: %s", output_dir)
    logger.info("  Plots: reports/figures/")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
