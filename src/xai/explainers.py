"""
Phase 5 — XAI Explainer Implementations
==========================================

Three independent explainer functions:
    1. ``explain_features_shap``     — Global feature importance via SHAP
    2. ``explain_structure_gnnexplainer`` — Spatial edge importance per node
    3. ``explain_temporal_captum``    — Temporal lookback attribution per node

All functions return Python dicts ready for JSON serialization.

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-28
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


# =====================================================================
# 1. SHAP — Feature Importance via Subgraph Extraction
# =====================================================================
# Strategy: SHAP on the full 24k-node graph is infeasible (OOM on GPU
# AND CPU). Instead we extract a k-hop neighbourhood around the target
# node (~100-300 nodes), build a mini-model wrapper that only processes
# that subgraph, and run SHAP on the small graph.  Fast & GPU-friendly.
# =====================================================================

def _extract_khop_subgraph(
    target_node_idx: int,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Extract a k-hop subgraph around the target node.

    Returns
    -------
    sub_nodes : 1-D tensor of original node indices in the subgraph
    sub_edge_index : (2, E_sub) — remapped edge index
    target_local_idx : int — index of the target node within sub_nodes
    """
    from torch_geometric.utils import k_hop_subgraph

    sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=target_node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    target_local_idx = (sub_nodes == target_node_idx).nonzero(as_tuple=True)[0].item()

    return sub_nodes, sub_edge_index, target_local_idx


class _SubgraphWrapper(torch.nn.Module):
    """
    Tiny adapter: takes (B, W*N_sub*F) flat → reshapes → GCN+LSTM+MLP
    on the subgraph only → returns (B,) for the target node.
    """

    def __init__(
        self, model, sub_edge_index, target_local_idx,
        window_size, num_sub_nodes, num_features,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.sub_edge_index = sub_edge_index
        self.target_local_idx = target_local_idx
        self.W = window_size
        self.N_sub = num_sub_nodes
        self.F = num_features

    def forward(self, x_flat):
        B = x_flat.shape[0]
        x_4d = x_flat.reshape(B, self.W, self.N_sub, self.F)
        pred = self.model(x_4d, self.sub_edge_index)   # → (B, N_sub)
        return pred[:, self.target_local_idx]           # → (B,)


def explain_features_shap(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    train_dataset,
    test_samples: list[torch.Tensor],
    target_node_idx: int,
    device: torch.device,
    window_size: int = 24,
    num_nodes: int = 24697,
    num_features: int = 5,
    background_k: int = 25,
    num_hops: int = 2,
) -> dict[str, Any]:
    """
    Compute SHAP feature importance using GradientExplainer on a
    **k-hop subgraph** around the target node.

    Runs on GPU — the subgraph is small enough (~100-300 nodes) to
    avoid OOM.

    Parameters
    ----------
    model : HybridSTGNN (trained)
    edge_index : (2, E) — full graph
    train_dataset : TemporalCrashDataset
    test_samples : list of (W, N, F) tensors (full graph)
    target_node_idx : int — node to explain
    device : torch.device
    background_k : int — random background samples
    num_hops : int — k-hop neighbourhood radius
    """
    import shap

    torch.backends.cudnn.enabled = False

    # --- Extract subgraph ---
    sub_nodes, sub_edge_index, target_local = _extract_khop_subgraph(
        target_node_idx, edge_index.cpu(), num_nodes, num_hops,
    )
    n_sub = len(sub_nodes)
    logger.info("  Subgraph: %d nodes (%d-hop around node %d)",
                n_sub, num_hops, target_node_idx)

    sub_edge_index = sub_edge_index.to(device)

    # --- Build wrapper on the small subgraph ---
    wrapper = _SubgraphWrapper(
        model.to(device), sub_edge_index, target_local,
        window_size, n_sub, num_features,
    ).to(device)
    wrapper.eval()

    # --- Background: random train samples, sliced to subgraph nodes ---
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(train_dataset), size=min(background_k, len(train_dataset)), replace=False)
    bg_list = []
    for idx in bg_indices:
        x, _ = train_dataset[int(idx)]           # (W, N, F)
        x_sub = x[:, sub_nodes, :]                # (W, N_sub, F)
        bg_list.append(x_sub.flatten())
    background = torch.stack(bg_list).to(device)   # (k, W*N_sub*F)
    logger.info("  Background: %s on %s", list(background.shape), device)

    # --- Test data sliced to subgraph ---
    test_flat = torch.stack([
        s[:, sub_nodes, :].flatten() for s in test_samples
    ]).to(device)
    logger.info("  Running SHAP GradientExplainer on %d samples (GPU, %d-node subgraph)...",
                len(test_flat), n_sub)

    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_flat)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_array = np.array(shap_values)  # (n_samples, W*N_sub*F)
    shap_4d = shap_array.reshape(-1, window_size, n_sub, num_features)

    # --- Feature importance: mean |SHAP| across samples, time, subgraph nodes ---
    feature_names = ["crash_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    global_importance = np.mean(np.abs(shap_4d), axis=(0, 1, 2))  # → (F,)
    global_dict = {name: float(val) for name, val in zip(feature_names, global_importance)}

    # --- Local SHAP for target node specifically ---
    local_shap = np.mean(np.abs(shap_4d[:, :, target_local, :]), axis=(0, 1))
    local_dict = {name: float(val) for name, val in zip(feature_names, local_shap)}

    torch.backends.cudnn.enabled = True
    logger.info("  ✓ SHAP complete. Top feature: %s", max(global_dict, key=global_dict.get))

    return {
        "global_importance": global_dict,
        "local_shap": local_dict,
        "target_node_idx": target_node_idx,
        "subgraph_size": n_sub,
        "num_hops": num_hops,
        "n_samples": len(test_samples),
        "background_k": len(bg_list),
    }


# =====================================================================
# 2. GNNExplainer — Spatial Structure Importance
# =====================================================================

def explain_structure_gnnexplainer(
    model: torch.nn.Module,
    x_input: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int,
    device: torch.device,
    top_k_edges: int = 20,
) -> dict[str, Any]:
    """
    Explain which edges (road segments) are most important for a target
    node's prediction using GNNExplainer.

    Uses the spatial-only GCN wrapper (last timestep of input window).

    Parameters
    ----------
    model : HybridSTGNN (trained)
    x_input : (W, N, F) — a single sample's features
    edge_index : (2, E)
    target_node_idx : int
    device : torch.device
    top_k_edges : int — return top-K most important edges

    Returns
    -------
    dict with edge mask, top-K edges, and weights
    """
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    from xai.adapters import GCNExplainerWrapper

    # Use last timestep features for spatial explanation
    x_last = x_input[-1, :, :].to(device)       # → (N, F)
    ei = edge_index.to(device)

    # Wrap model for GNNExplainer (spatial GCN + MLP only)
    wrapper = GCNExplainerWrapper(model).to(device)
    wrapper.eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        model_config=ModelConfig(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type="object",
    )

    logger.info("  Running GNNExplainer for node %d...", target_node_idx)
    explanation = explainer(x_last, ei, target=torch.tensor(target_node_idx))

    edge_mask = explanation.edge_mask.detach().cpu().numpy()  # → (E,)

    # Top-K edges by importance
    top_k_idx = np.argsort(edge_mask)[-top_k_edges:][::-1]
    edge_src = edge_index[0].cpu().numpy()
    edge_dst = edge_index[1].cpu().numpy()

    top_edges = []
    for idx in top_k_idx:
        top_edges.append({
            "edge_idx": int(idx),
            "src_node": int(edge_src[idx]),
            "dst_node": int(edge_dst[idx]),
            "weight": float(edge_mask[idx]),
        })

    # Identify unique nodes in the explanation subgraph
    subgraph_nodes = set()
    for e in top_edges:
        subgraph_nodes.add(e["src_node"])
        subgraph_nodes.add(e["dst_node"])

    logger.info("  ✓ GNNExplainer complete. Subgraph: %d nodes, %d edges",
                len(subgraph_nodes), len(top_edges))

    return {
        "target_node_idx": target_node_idx,
        "top_k_edges": top_edges,
        "subgraph_nodes": sorted(subgraph_nodes),
        "total_edges": len(edge_mask),
        "edge_mask_stats": {
            "mean": float(edge_mask.mean()),
            "std": float(edge_mask.std()),
            "max": float(edge_mask.max()),
            "min": float(edge_mask.min()),
        },
    }


# =====================================================================
# 3. Captum — Temporal Attribution
# =====================================================================

def explain_temporal_captum(
    model: torch.nn.Module,
    x_input: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int,
    device: torch.device,
    window_size: int = 24,
    num_nodes: int = 24697,
    num_features: int = 5,
    n_steps: int = 50,
) -> dict[str, Any]:
    """
    Compute temporal attribution using Captum IntegratedGradients.

    Shows which of the 24 hourly timesteps contribute most to
    the prediction for a specific node.

    Parameters
    ----------
    model : HybridSTGNN
    x_input : (W, N, F) — a single sample
    edge_index : (2, E)
    target_node_idx : int
    device : torch.device
    n_steps : int — integration steps for IG

    Returns
    -------
    dict with per-hour attribution scores
    """
    from captum.attr import IntegratedGradients
    from xai.adapters import NodeTargetWrapper

    # Disable cuDNN for LSTM backward stability
    torch.backends.cudnn.enabled = False

    wrapper = NodeTargetWrapper(
        model, edge_index.to(device), target_node_idx,
        window_size, num_nodes, num_features,
    ).to(device)
    wrapper.eval()

    # Flatten input for the wrapper: (1, W*N*F)
    x_flat = x_input.flatten().unsqueeze(0).to(device)
    x_flat.requires_grad_(True)

    # Baseline: zeros (no crash activity)
    baseline = torch.zeros_like(x_flat)

    logger.info("  Running Captum IntegratedGradients for node %d...", target_node_idx)

    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(
        x_flat, baselines=baseline, n_steps=n_steps,
        internal_batch_size=1,
    )  # → (1, W*N*F)

    attr_4d = attributions.detach().cpu().numpy().reshape(
        window_size, num_nodes, num_features
    )

    # Temporal importance: mean |attribution| across nodes and features per timestep
    temporal_importance = np.mean(np.abs(attr_4d), axis=(1, 2))  # → (W,)

    # Normalize to sum to 1 for interpretability
    total = temporal_importance.sum()
    if total > 0:
        temporal_normalized = temporal_importance / total
    else:
        temporal_normalized = temporal_importance

    torch.backends.cudnn.enabled = True
    logger.info("  ✓ Captum complete. Peak hour offset: %d", int(np.argmax(temporal_importance)))

    return {
        "target_node_idx": target_node_idx,
        "temporal_importance_raw": temporal_importance.tolist(),
        "temporal_importance_normalized": temporal_normalized.tolist(),
        "peak_hour_offset": int(np.argmax(temporal_importance)),
        "window_size": window_size,
    }
