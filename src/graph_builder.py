"""
Phase 2 — Module 1: Spatial Graph Construction & Feature Engineering
=====================================================================

Constructs a projected, consolidated road network graph for Dallas, TX
using OSMnx. Computes node-level and edge-level features required for
the ST-GNN risk prediction model.

Key steps:
    1. Download/load Dallas driving network from OpenStreetMap
    2. Project to UTM Zone 14N (EPSG:32614) for metric distances
    3. Consolidate complex intersections (tolerance=15m)
    4. Extract the Largest Strongly Connected Component (LSCC)
    5. Compute node features: degree, street_count, betweenness,
       bearing_entropy, avg_incident_speed
    6. Compute edge features: length, bearing, maxspeed (imputed),
       highway_encoded

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRS_UTM_14N = "EPSG:32614"          # UTM Zone 14N covers Dallas
CONSOLIDATION_TOLERANCE = 15        # metres – merge complex junctions
PLACE_QUERY = "Dallas, Texas, USA"

# Speed imputation by highway type (mph)
SPEED_DEFAULTS: dict[str, int] = {
    "motorway": 65,
    "motorway_link": 55,
    "trunk": 55,
    "trunk_link": 45,
    "primary": 45,
    "primary_link": 35,
    "secondary": 35,
    "secondary_link": 30,
    "tertiary": 30,
    "tertiary_link": 25,
    "residential": 25,
    "unclassified": 25,
    "living_street": 20,
}
DEFAULT_SPEED: int = 30  # fallback


# ============================================================================
# GraphBuilder
# ============================================================================

class GraphBuilder:
    """
    Constructs a feature-enriched road network graph for Dallas, TX.

    The graph is downloaded from OpenStreetMap via OSMnx, projected to
    UTM coordinates, consolidated at complex intersections, and reduced
    to its Largest Strongly Connected Component.

    Attributes
    ----------
    graph_path : Path
        Location to cache the GraphML file.
    mapping_path : Path
        Location to store the node-id mapping dict.
    G : nx.MultiDiGraph | None
        The constructed graph (``None`` before ``build()`` is called).
    node_mapping : dict[int, int] | None
        OSMnx node-id → contiguous-index mapping.
    """

    def __init__(
        self,
        graph_path: Path = Path("data/processed/dallas_drive_net.graphml"),
        mapping_path: Path = Path("data/processed/node_mapping.pkl"),
    ) -> None:
        self.graph_path = graph_path
        self.mapping_path = mapping_path
        self.G: nx.MultiDiGraph | None = None
        self.node_mapping: dict[int, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, force_download: bool = False) -> nx.MultiDiGraph:
        """
        Build the full graph pipeline.

        Parameters
        ----------
        force_download : bool
            If *True*, re-download the graph even when a cached copy exists.

        Returns
        -------
        nx.MultiDiGraph
            The processed, feature-enriched graph.
        """
        self.G = self._get_raw_graph(force_download)
        logger.info(
            "Raw graph: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

        # Project → consolidate → LSCC
        self.G = self._project(self.G)
        self.G = self._consolidate(self.G)
        self.G = self._extract_lscc(self.G)

        # Feature engineering
        self._compute_edge_features()
        self._compute_node_features()

        # Build contiguous node mapping
        self.node_mapping = {
            nid: idx for idx, nid in enumerate(sorted(self.G.nodes()))
        }

        # Persist
        self._save()

        logger.info(
            "Final graph: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )
        return self.G

    # ------------------------------------------------------------------
    # Graph acquisition
    # ------------------------------------------------------------------
    def _get_raw_graph(self, force: bool) -> nx.MultiDiGraph:
        """Download or load the Dallas driving network."""
        if not force and self.graph_path.exists():
            logger.info("Loading cached graph from %s", self.graph_path)
            return ox.load_graphml(self.graph_path)

        logger.info("Downloading Dallas driving network from OSM …")
        G = ox.graph_from_place(PLACE_QUERY, network_type="drive")
        return G

    # ------------------------------------------------------------------
    # Spatial transforms
    # ------------------------------------------------------------------
    @staticmethod
    def _project(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Project the graph to UTM Zone 14N (metres)."""
        logger.info("Projecting to %s …", CRS_UTM_14N)
        return ox.project_graph(G, to_crs=CRS_UTM_14N)

    @staticmethod
    def _consolidate(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Merge complex-intersection nodes within *tolerance* metres."""
        logger.info(
            "Consolidating intersections (tolerance=%d m) …",
            CONSOLIDATION_TOLERANCE,
        )
        return ox.consolidate_intersections(
            G, tolerance=CONSOLIDATION_TOLERANCE, rebuild_graph=True, dead_ends=False
        )

    @staticmethod
    def _extract_lscc(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Keep only the Largest Strongly Connected Component."""
        before = G.number_of_nodes()
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        after = G.number_of_nodes()
        pct = (after / before) * 100 if before else 0
        logger.info(
            "LSCC: kept %d / %d nodes (%.1f%%)", after, before, pct
        )
        return G

    # ------------------------------------------------------------------
    # Edge features
    # ------------------------------------------------------------------
    def _compute_edge_features(self) -> None:
        """Compute / impute edge-level features on ``self.G``."""
        assert self.G is not None

        # --- Bearings (manual from UTM coordinates) ---
        # OSMnx's add_edge_bearings requires unprojected coords,
        # so we compute bearings from projected (x, y) via atan2.
        for u, v, k, data in self.G.edges(keys=True, data=True):
            u_data = self.G.nodes[u]
            v_data = self.G.nodes[v]
            dx = v_data.get("x", 0) - u_data.get("x", 0)
            dy = v_data.get("y", 0) - u_data.get("y", 0)
            # atan2 gives angle from east; convert to compass bearing (from north)
            bearing = (90 - np.degrees(np.arctan2(dy, dx))) % 360
            data["bearing"] = round(bearing, 2)

        # --- Speed imputation & highway encoding ---
        highway_labels: list[str] = []

        for u, v, k, data in self.G.edges(keys=True, data=True):
            # Normalise the 'highway' attribute (may be a list)
            hw = data.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0]
            hw = str(hw).lower()
            data["highway_str"] = hw
            highway_labels.append(hw)

            # Speed: parse existing or impute
            raw_speed = data.get("maxspeed", None)
            if isinstance(raw_speed, list):
                raw_speed = raw_speed[0]
            speed = self._parse_speed(raw_speed, hw)
            data["speed_mph"] = speed

            # Length should already be present (metres) after projection
            if "length" not in data:
                data["length"] = 0.0

        # Label-encode highway types
        le = LabelEncoder()
        encoded = le.fit_transform(highway_labels)
        for (u, v, k, data), enc in zip(
            self.G.edges(keys=True, data=True), encoded
        ):
            data["highway_enc"] = int(enc)

        logger.info("Edge features computed (bearing, speed, highway_enc)")

    @staticmethod
    def _parse_speed(raw: Any, highway_type: str) -> float:
        """Parse a raw maxspeed tag or impute from highway type."""
        if raw is not None:
            try:
                # "35 mph" → 35,  "50" → 50
                return float(str(raw).split()[0])
            except (ValueError, IndexError):
                pass
        return float(SPEED_DEFAULTS.get(highway_type, DEFAULT_SPEED))

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------
    def _compute_node_features(self) -> None:
        """Compute node-level features on ``self.G``."""
        assert self.G is not None
        n_nodes = self.G.number_of_nodes()

        # --- Degree ---
        for n, deg in self.G.degree():
            self.G.nodes[n]["degree"] = deg

        # --- Street count (may already exist from OSMnx) ---
        for n, data in self.G.nodes(data=True):
            if "street_count" not in data:
                data["street_count"] = data.get("degree", 0)

        # --- Betweenness centrality ---
        k_sample = 1000 if n_nodes > 10_000 else None
        logger.info(
            "Computing betweenness centrality (k=%s, nodes=%d) …",
            k_sample if k_sample else "full",
            n_nodes,
        )
        bc = nx.betweenness_centrality(self.G, k=k_sample, weight="length")
        nx.set_node_attributes(self.G, bc, "betweenness")

        # --- Bearing entropy ---
        self._compute_bearing_entropy()

        # --- Average incident edge speed ---
        self._compute_avg_incident_speed()

        logger.info("Node features computed (degree, betweenness, bearing_entropy, avg_speed)")

    def _compute_bearing_entropy(self) -> None:
        """Shannon entropy of incident-edge bearings (8 compass bins)."""
        assert self.G is not None
        n_bins = 8
        bin_edges = np.linspace(0, 360, n_bins + 1)

        for node in self.G.nodes():
            bearings: list[float] = []
            # Outgoing
            for _, _, data in self.G.out_edges(node, data=True):
                b = data.get("bearing", None)
                if b is not None:
                    bearings.append(float(b))
            # Incoming (reverse bearing)
            for _, _, data in self.G.in_edges(node, data=True):
                b = data.get("bearing", None)
                if b is not None:
                    bearings.append((float(b) + 180) % 360)

            if not bearings:
                self.G.nodes[node]["bearing_entropy"] = 0.0
                continue

            counts, _ = np.histogram(bearings, bins=bin_edges)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            self.G.nodes[node]["bearing_entropy"] = float(entropy)

    def _compute_avg_incident_speed(self) -> None:
        """Mean maxspeed of all incident edges for each node."""
        assert self.G is not None
        for node in self.G.nodes():
            speeds: list[float] = []
            for _, _, data in self.G.out_edges(node, data=True):
                speeds.append(data.get("speed_mph", DEFAULT_SPEED))
            for _, _, data in self.G.in_edges(node, data=True):
                speeds.append(data.get("speed_mph", DEFAULT_SPEED))
            avg = float(np.mean(speeds)) if speeds else float(DEFAULT_SPEED)
            self.G.nodes[node]["avg_speed"] = avg

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Save graph and node mapping to disk."""
        assert self.G is not None and self.node_mapping is not None

        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        # Save GraphML — strip any non-serialisable attrs first
        G_save = self.G.copy()
        for _, _, data in G_save.edges(data=True):
            for key in list(data.keys()):
                if isinstance(data[key], (list, dict, set)):
                    data[key] = str(data[key])
        for _, data in G_save.nodes(data=True):
            for key in list(data.keys()):
                if isinstance(data[key], (list, dict, set)):
                    data[key] = str(data[key])

        ox.save_graphml(G_save, filepath=self.graph_path)
        logger.info("Graph saved → %s", self.graph_path)

        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.node_mapping, f)
        logger.info("Node mapping saved → %s", self.mapping_path)


# ============================================================================
# Stand-alone execution (for testing)
# ============================================================================
if __name__ == "__main__":
    builder = GraphBuilder()
    G = builder.build()
    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Sample node features
    sample_node = list(G.nodes())[0]
    print(f"\nSample node ({sample_node}):")
    for k, v in G.nodes[sample_node].items():
        print(f"  {k}: {v}")

    # Sample edge features
    sample_edge = list(G.edges(data=True))[0]
    u, v, data = sample_edge
    print(f"\nSample edge ({u} → {v}):")
    for k, val in data.items():
        print(f"  {k}: {val}")
