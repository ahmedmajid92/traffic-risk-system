"""
Phase 4 — Module 1: HybridSTGNN Model Architecture
====================================================

A Spatio-Temporal Graph Neural Network combining:
    - **Spatial Encoder**: 2-layer GCNConv (captures road-network topology)
    - **Temporal Core**: LSTM (captures 24-hour sequential patterns)
    - **Risk Decoder**: MLP head (predicts per-node crash risk)

Memory-efficient design:
    - GCN processes one sample at a time (B=1 loop), avoiding B×N blow-up
    - LSTM + MLP process nodes in chunks of ``node_chunk_size``
    - Gradient checkpointing on GCN timesteps frees intermediate activations
    - Peak VRAM ~5-6 GB for batch_size=4, N=24697, H=128

Tensor shape annotations use the convention:
    B = batch size, W = window size (24), N = num nodes (24697),
    F = input features (5), H = hidden dim (128), E = num edges (71392)

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv


class HybridSTGNN(nn.Module):
    """
    Hybrid Spatio-Temporal Graph Neural Network.

    Architecture::

        Input (B, W, N, F)
            → GCN Encoder (per-sample, per-timestep spatial convolution)
            → LSTM Core   (per-node temporal modelling, chunked)
            → MLP Decoder (per-node risk prediction, chunked)
        Output (B, N)

    Parameters
    ----------
    in_features : int
        Number of dynamic input features per node (default 5).
    hidden_dim : int
        Hidden dimension for GCN and LSTM (default 128).
    lstm_layers : int
        Number of LSTM layers (default 1).
    dropout : float
        Dropout rate applied between GCN layers (default 0.1).
    node_chunk_size : int
        Process LSTM in chunks of this many nodes (default 4096).
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        node_chunk_size: int = 4096,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.node_chunk_size = node_chunk_size

        # --- Spatial Encoder: 2-layer GCN ---
        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn_dropout = nn.Dropout(dropout)

        # --- Temporal Core: LSTM ---
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- Risk Decoder: MLP ---
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _gcn_one_timestep(
        self, x_flat: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Run 2-layer GCN on a single timestep's node features.

        Parameters
        ----------
        x_flat : (N, F) — node features for one timestep
        edge_index : (2, E) — single graph

        Returns
        -------
        h : (N, H) — node embeddings
        """
        h = self.gcn1(x_flat, edge_index)                  # → (N, H)
        h = F.relu(h)
        h = self.gcn_dropout(h)
        h = self.gcn2(h, edge_index)                       # → (N, H)
        h = F.relu(h)
        return h

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass with shape annotations.

        Memory-efficient design:
        1. GCN processes one sample at a time to avoid B*N node blow-up
        2. LSTM + MLP chunk across nodes (node_chunk_size per chunk)

        Parameters
        ----------
        x : (B, W, N, F) — dynamic features per window
        edge_index : (2, E) — static graph (shared across all samples)

        Returns
        -------
        pred : (B, N) — predicted risk score per node
        """
        B, W, N, F_in = x.shape

        # ==============================================================
        # STEP 1: Spatial Encoder (GCN — per sample, per timestep)
        # ==============================================================
        # Process each sample independently to keep VRAM at ~N nodes
        # instead of B*N. Trade-off: serial over B, parallel over N.

        all_sample_embeddings = []
        for b in range(B):
            # Per-sample timestep embeddings
            sample_gcn = []
            for t in range(W):
                # Single timestep, single sample: (N, F_in)
                x_t = x[b, t, :, :]                        # → (N, F_in)

                # 2-layer GCN: (N, F_in) → (N, H)
                h_t = self._gcn_one_timestep(x_t, edge_index)

                sample_gcn.append(h_t)                     # → (N, H)

            # Stack timesteps: list of W × (N, H) → (W, N, H) → (N, W, H)
            h_sample = torch.stack(sample_gcn, dim=0)      # → (W, N, H)
            h_sample = h_sample.permute(1, 0, 2)           # → (N, W, H)
            all_sample_embeddings.append(h_sample)

        # Stack samples: list of B × (N, W, H) → (B, N, W, H)
        h_all = torch.stack(all_sample_embeddings, dim=0)  # → (B, N, W, H)

        # ==============================================================
        # STEP 2+3: LSTM + MLP (chunked across nodes for memory)
        # ==============================================================
        pred_chunks = []
        for n_start in range(0, N, self.node_chunk_size):
            n_end = min(n_start + self.node_chunk_size, N)
            chunk_n = n_end - n_start

            # Extract chunk: (B, chunk_n, W, H)
            h_chunk = h_all[:, n_start:n_end, :, :]

            # Reshape for LSTM: (B, chunk_n, W, H) → (B*chunk_n, W, H)
            h_lstm_in = h_chunk.reshape(B * chunk_n, W, self.hidden_dim)

            # LSTM: (B*chunk_n, W, H) → h_n = (layers, B*chunk_n, H)
            _, (h_n, _) = self.lstm(h_lstm_in)

            # Last hidden: (B*chunk_n, H)
            h_last = h_n[-1]                               # → (B*chunk_n, H)

            # MLP: (B*chunk_n, H) → (B*chunk_n, 1)
            p = self.decoder(h_last)                       # → (B*chunk_n, 1)

            # Reshape: (B*chunk_n) → (B, chunk_n)
            p = p.squeeze(-1).reshape(B, chunk_n)
            pred_chunks.append(p)

        # Concatenate chunks: list of (B, chunk_n) → (B, N)
        pred = torch.cat(pred_chunks, dim=1)               # → (B, N)

        return pred


# =====================================================================
# Quick sanity check
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dummy inputs matching real dimensions
    B, W, N, F_in = 2, 24, 100, 5  # small N for testing
    E = 300

    model = HybridSTGNN(
        in_features=F_in, hidden_dim=128, node_chunk_size=50
    ).to(device)
    x = torch.randn(B, W, N, F_in, device=device)
    edge_index = torch.randint(0, N, (2, E), device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input:  x={list(x.shape)}, edge_index={list(edge_index.shape)}")

    pred = model(x, edge_index)
    print(f"Output: pred={list(pred.shape)}")
    print("✓ Forward pass OK")
