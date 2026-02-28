"""
Phase 5 — XAI Adapter Layer
==============================

Wraps HybridSTGNN for compatibility with SHAP, Captum, and
GNNExplainer, which expect different input/output shapes.

Two wrappers:
    - ``NodeTargetWrapper``: (B, W*N*F) flat → scalar (B,) for one node
      → Used by SHAP GradientExplainer and Captum IntegratedGradients
    - ``GCNExplainerWrapper``: (N, H) node embeddings → (N,) scores
      → Used by PyG's GNNExplainer (spatial-only explanation)

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-28
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_architecture import HybridSTGNN


class NodeTargetWrapper(nn.Module):
    """
    Adapter for SHAP / Captum.

    Accepts a **flat** input tensor ``(B, W*N*F)`` — since these tools
    often can't handle 4-D inputs — reshapes it to ``(B, W, N, F)``,
    runs the full HybridSTGNN, and returns the prediction for a single
    target node as ``(B,)``.

    Parameters
    ----------
    model : HybridSTGNN
        Trained model (will be set to eval mode).
    edge_index : torch.Tensor
        Static graph edge index ``(2, E)``.
    target_node_idx : int
        Which node's prediction to return.
    window_size : int
        Temporal window length (default 24).
    num_nodes : int
        Number of graph nodes (default 24697).
    num_features : int
        Input features per node per timestep (default 5).
    """

    def __init__(
        self,
        model: HybridSTGNN,
        edge_index: torch.Tensor,
        target_node_idx: int,
        window_size: int = 24,
        num_nodes: int = 24697,
        num_features: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.edge_index = edge_index
        self.target_node_idx = target_node_idx
        self.W = window_size
        self.N = num_nodes
        self.F = num_features

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_flat : (B, W*N*F) — flattened spatio-temporal input

        Returns
        -------
        (B,) — predicted risk score for the target node
        """
        B = x_flat.shape[0]
        x_4d = x_flat.reshape(B, self.W, self.N, self.F)
        pred = self.model(x_4d, self.edge_index)       # → (B, N)
        return pred[:, self.target_node_idx]            # → (B,)


class GCNExplainerWrapper(nn.Module):
    """
    Adapter for PyG GNNExplainer.

    GNNExplainer operates on a single graph snapshot ``(N, F)`` and
    explains which edges matter.  This wrapper feeds the node features
    through the **spatial encoder only** (2-layer GCN) — the temporal
    dimension is collapsed by using the *last timestep* of a given
    input window.

    The decoder MLP is applied on top so that the output is the actual
    risk score, making the explanation faithful to the prediction.

    Parameters
    ----------
    model : HybridSTGNN
        Trained model.
    """

    def __init__(self, model: HybridSTGNN) -> None:
        super().__init__()
        self.gcn1 = model.gcn1
        self.gcn2 = model.gcn2
        self.gcn_dropout = model.gcn_dropout
        self.decoder = model.decoder
        self.eval()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, F) — node features for a single timestep
        edge_index : (2, E)

        Returns
        -------
        (N,) — predicted risk score per node
        """
        import torch.nn.functional as F_act

        h = self.gcn1(x, edge_index)
        h = F_act.relu(h)
        h = self.gcn_dropout(h)
        h = self.gcn2(h, edge_index)
        h = F_act.relu(h)                               # → (N, H)
        pred = self.decoder(h).squeeze(-1)               # → (N,)
        return pred
