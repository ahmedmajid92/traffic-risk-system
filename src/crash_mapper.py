"""
Phase 2 — Module 2: Vectorized Crash-to-Node Spatial Snapping
==============================================================

Maps each crash record from the annotated Dallas CSV to its nearest
graph node via *edge snapping* — crashes are projected to the closest
road segment, then assigned to the destination node of that edge
(modelling "intersection approach risk").

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CRS_UTM_14N = "EPSG:32614"


# ============================================================================
# CrashMapper
# ============================================================================

class CrashMapper:
    """
    Snap crash locations to the nearest graph edge / node.

    The mapping uses projected (UTM) coordinates for metric distance
    accuracy.  Each crash is associated with the **destination node**
    ``v`` of the nearest edge ``(u, v, key)``.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Projected road-network graph (must already be in UTM).
    crash_csv : Path
        Path to the annotated crash CSV.
    """

    def __init__(
        self,
        G: nx.MultiDiGraph,
        crash_csv: Path = Path("data/processed/dallas_crashes_annotated.csv"),
    ) -> None:
        self.G = G
        self.crash_csv = crash_csv
        self._transformer = Transformer.from_crs(
            "EPSG:4326", CRS_UTM_14N, always_xy=True
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def snap(self) -> pd.DataFrame:
        """
        Load crashes, project to UTM, snap to nearest edges, and return
        an augmented DataFrame with ``graph_node`` and ``snap_dist_m``.

        Returns
        -------
        pd.DataFrame
            Crash data with two new columns:
            - ``graph_node``: OSMnx node id of the destination node
            - ``snap_dist_m``: snapping distance in metres
        """
        df = self._load_crashes()
        df = self._project_coords(df)
        df = self._snap_to_graph(df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_crashes(self) -> pd.DataFrame:
        """Load and validate the crash CSV."""
        logger.info("Loading crashes from %s …", self.crash_csv)
        df = pd.read_csv(self.crash_csv)

        required = {"Start_Lat", "Start_Lng"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in crash data: {missing}")

        # Drop rows without coordinates
        before = len(df)
        df = df.dropna(subset=["Start_Lat", "Start_Lng"]).copy()
        after = len(df)
        if before != after:
            logger.warning(
                "Dropped %d rows with missing coordinates", before - after
            )

        logger.info("Loaded %d crash records", len(df))
        return df

    def _project_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add UTM-projected X / Y columns."""
        logger.info("Projecting crash coordinates to UTM …")
        xs, ys = self._transformer.transform(
            df["Start_Lng"].values,
            df["Start_Lat"].values,
        )
        df["utm_x"] = xs
        df["utm_y"] = ys
        return df

    def _snap_to_graph(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Snap each crash to the nearest edge and assign the dest node.

        Uses ``ox.nearest_edges`` for efficient vectorised look-up.
        """
        logger.info("Snapping %d crashes to nearest edges …", len(df))

        X = df["utm_x"].values
        Y = df["utm_y"].values

        # Vectorised nearest-edge look-up
        nearest = ox.nearest_edges(self.G, X, Y, return_dist=True)
        edges, dists = nearest

        # Extract destination (v) node for each crash
        dest_nodes = [edge[1] for edge in edges]

        df["graph_node"] = dest_nodes
        df["snap_dist_m"] = np.round(dists, 2)

        # Report snapping statistics
        median_dist = np.median(dists)
        p95_dist = np.percentile(dists, 95)
        max_dist = np.max(dists)
        logger.info(
            "Snap distances — median: %.1f m, P95: %.1f m, max: %.1f m",
            median_dist, p95_dist, max_dist,
        )

        # Flag potentially bad snaps (> 500 m)
        bad_snaps = (np.array(dists) > 500).sum()
        if bad_snaps:
            logger.warning(
                "%d crashes snapped > 500 m from nearest road", bad_snaps
            )

        return df

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_per_node(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate crash-level data to the node level.

        Returns a DataFrame indexed by ``graph_node`` with columns:
        - ``crash_count``:      total crashes at this node
        - ``distracted_count``: crashes flagged Is_Distracted == 1
        - ``avg_severity``:     mean Severity
        - ``max_severity``:     maximum Severity
        - ``severity_list``:    list of all Severity values

        Parameters
        ----------
        df : pd.DataFrame
            Crash-level data with ``graph_node``, ``Severity``,
            ``Is_Distracted`` columns.
        """
        agg = df.groupby("graph_node").agg(
            crash_count=("Severity", "count"),
            distracted_count=("Is_Distracted", "sum"),
            avg_severity=("Severity", "mean"),
            max_severity=("Severity", "max"),
            severity_list=("Severity", list),
        )
        return agg


# ============================================================================
# Stand-alone execution (for testing)
# ============================================================================
if __name__ == "__main__":
    import pickle

    graph_path = Path("data/processed/dallas_drive_net.graphml")
    if not graph_path.exists():
        print("Run graph_builder.py first.")
        raise SystemExit(1)

    G = ox.load_graphml(graph_path)
    G = ox.project_graph(G, to_crs=CRS_UTM_14N)

    mapper = CrashMapper(G)
    crashes = mapper.snap()
    print(f"\nSnapped crashes: {len(crashes)}")
    print(crashes[["ID", "graph_node", "snap_dist_m"]].head(10))

    agg = CrashMapper.aggregate_per_node(crashes)
    print(f"\nNodes with crashes: {len(agg)}")
    print(agg.head(10))
