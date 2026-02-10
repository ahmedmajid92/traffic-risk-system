"""
Phase 2 — Module 4: PyTorch Geometric Data Conversion
======================================================

Converts the feature-enriched NetworkX road graph and DR-ISI targets
into a ``torch_geometric.data.Data`` object ready for GNN training.

Handles sanitisation of non-numeric / non-serialisable attributes
(Shapely geometries, string tags, lists) before tensor construction.

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature definitions (must match graph_builder.py outputs)
# ---------------------------------------------------------------------------
NODE_FEATURES = [
    "degree",
    "street_count",
    "betweenness",
    "bearing_entropy",
    "avg_speed",
]

EDGE_FEATURES = [
    "length",
    "bearing",
    "speed_mph",
    "highway_enc",
]

TARGET_ATTR = "drisi"
POS_ATTRS = ("x", "y")        # UTM coordinates from projected graph


# ============================================================================
# PyGConverter
# ============================================================================

class PyGConverter:
    """
    Convert a feature-enriched ``nx.MultiDiGraph`` to a PyG ``Data`` object.

    Parameters
    ----------
    output_path : Path
        Where to save the ``.pt`` file.
    """

    def __init__(
        self,
        output_path: Path = Path("data/processed/processed_graph_data.pt"),
    ) -> None:
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self,
        G: nx.MultiDiGraph,
        node_mapping: dict[int, int],
    ) -> Data:
        """
        Sanitise, build tensors, and return a PyG ``Data`` object.

        Parameters
        ----------
        G : nx.MultiDiGraph
            Road graph with node/edge features and ``drisi`` target.
        node_mapping : dict
            OSMnx node-id → contiguous index mapping.

        Returns
        -------
        torch_geometric.data.Data
        """
        # 1. Sanitise the graph
        G_clean = self._sanitise(G)

        # 2. Build tensors manually for full control
        data = self._build_data(G_clean, node_mapping)

        # 3. Save
        self._save(data)

        return data

    # ------------------------------------------------------------------
    # Sanitisation
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitise(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Remove non-numeric / non-serialisable attributes from nodes
        and edges so that tensor construction does not fail.
        """
        G = G.copy()

        # Node cleanup
        removed_node_attrs: set[str] = set()
        keep_node = set(NODE_FEATURES) | {TARGET_ATTR} | set(POS_ATTRS)
        for _, data in G.nodes(data=True):
            for key in list(data.keys()):
                if key not in keep_node:
                    removed_node_attrs.add(key)
                    del data[key]
        if removed_node_attrs:
            logger.info("Removed node attrs: %s", sorted(removed_node_attrs))

        # Edge cleanup
        removed_edge_attrs: set[str] = set()
        keep_edge = set(EDGE_FEATURES)
        for _, _, data in G.edges(data=True):
            for key in list(data.keys()):
                if key not in keep_edge:
                    removed_edge_attrs.add(key)
                    del data[key]
        if removed_edge_attrs:
            logger.info("Removed edge attrs: %s", sorted(removed_edge_attrs))

        return G

    # ------------------------------------------------------------------
    # Tensor construction
    # ------------------------------------------------------------------
    def _build_data(
        self,
        G: nx.MultiDiGraph,
        node_mapping: dict[int, int],
    ) -> Data:
        """
        Manually construct the PyG Data object from the graph.
        """
        n_nodes = len(node_mapping)
        sorted_nodes = sorted(node_mapping.keys(), key=lambda n: node_mapping[n])

        # --- Node features (N, F) ---
        node_feat_list: list[list[float]] = []
        targets: list[float] = []
        positions: list[list[float]] = []

        for nid in sorted_nodes:
            nd = G.nodes[nid]
            feats = [float(nd.get(f, 0.0)) for f in NODE_FEATURES]
            node_feat_list.append(feats)

            targets.append(float(nd.get(TARGET_ATTR, 0.0)))

            x_coord = float(nd.get("x", 0.0))
            y_coord = float(nd.get("y", 0.0))
            positions.append([x_coord, y_coord])

        x = torch.tensor(node_feat_list, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float64)

        # --- Edge index (2, E) and edge attr (E, Fe) ---
        src_list: list[int] = []
        dst_list: list[int] = []
        edge_feat_list: list[list[float]] = []

        for u, v, data in G.edges(data=True):
            if u in node_mapping and v in node_mapping:
                src_list.append(node_mapping[u])
                dst_list.append(node_mapping[v])
                efeats = [float(data.get(f, 0.0)) for f in EDGE_FEATURES]
                edge_feat_list.append(efeats)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)

        # --- Assemble Data object ---
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
        )

        # Metadata for later inspection
        data.node_feature_names = NODE_FEATURES
        data.edge_feature_names = EDGE_FEATURES
        data.num_node_features_count = len(NODE_FEATURES)
        data.num_edge_features_count = len(EDGE_FEATURES)

        logger.info(
            "PyG Data — x: %s, edge_index: %s, edge_attr: %s, y: %s, pos: %s",
            list(data.x.shape),
            list(data.edge_index.shape),
            list(data.edge_attr.shape),
            list(data.y.shape),
            list(data.pos.shape),
        )

        # Validation
        assert data.edge_index.max() < n_nodes, (
            f"edge_index max ({data.edge_index.max()}) >= num_nodes ({n_nodes})"
        )
        assert data.x.shape[0] == n_nodes
        assert data.y.shape[0] == n_nodes
        logger.info("Validation passed ✓")

        return data

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self, data: Data) -> None:
        """Save the PyG Data object."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, self.output_path)
        logger.info("PyG Data saved → %s", self.output_path)

    # ------------------------------------------------------------------
    # Loading helper
    # ------------------------------------------------------------------
    @staticmethod
    def load(path: Path = Path("data/processed/processed_graph_data.pt")) -> Data:
        """Load a previously saved PyG Data object."""
        data = torch.load(path, weights_only=False)
        logger.info("Loaded PyG Data from %s", path)
        return data


# ============================================================================
# Stand-alone execution
# ============================================================================
if __name__ == "__main__":
    pt_path = Path("data/processed/processed_graph_data.pt")
    if pt_path.exists():
        data = PyGConverter.load(pt_path)
        print(data)
    else:
        print("Run build_graph_pipeline.py first.")
