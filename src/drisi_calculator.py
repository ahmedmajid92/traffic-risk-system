"""
Phase 2 — Module 3: Distraction-Related Intersection Severity Index (DR-ISI)
=============================================================================

Computes the DR-ISI target variable for every node in the spatial graph.
Uses EPDO (Equivalent Property-Damage-Only) severity weights and
log-normalisation to produce a continuous risk score suitable for
regression-based GNN training.

Formulation
-----------
For each node *n*:

    WSS(n) = Σ w(c)   for all distraction-related crashes c at node n

    DR_ISI(n) = log(WSS(n) + 1)

Where severity weights w follow the EPDO standard:
    Fatal / Incapacitating (Severity 4)  → 12
    Injury                 (Severity 3)  →  3
    PDO                    (Severity 1-2) →  1

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

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
EPDO_WEIGHTS: dict[int, float] = {
    4: 12.0,   # Fatal / Incapacitating
    3: 3.0,    # Injury
    2: 1.0,    # PDO
    1: 1.0,    # PDO
}


# ============================================================================
# DRISICalculator
# ============================================================================

class DRISICalculator:
    """
    Compute the Distraction-Related Intersection Severity Index.

    Parameters
    ----------
    severity_weights : dict[int, float] | None
        Mapping of Severity (1-4) → EPDO weight.
        Defaults to the standard EPDO table.
    """

    def __init__(
        self,
        severity_weights: dict[int, float] | None = None,
    ) -> None:
        self.weights = severity_weights or EPDO_WEIGHTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        crash_df: pd.DataFrame,
        G: nx.MultiDiGraph,
    ) -> dict[Any, float]:
        """
        Compute DR-ISI for all graph nodes.

        Parameters
        ----------
        crash_df : pd.DataFrame
            Crash data with columns: ``graph_node``, ``Severity``,
            ``Is_Distracted``.
        G : nx.MultiDiGraph
            The road-network graph.

        Returns
        -------
        dict[node_id, float]
            DR-ISI score per node.  Nodes without distraction crashes
            receive a score of 0.0.
        """
        # Filter distraction-related crashes only
        distracted = crash_df[crash_df["Is_Distracted"] == 1].copy()
        logger.info(
            "Distraction crashes for DR-ISI: %d / %d total",
            len(distracted), len(crash_df),
        )

        # Map severity → EPDO weight
        distracted["weight"] = distracted["Severity"].map(self.weights).fillna(1.0)

        # Weighted Severity Sum per node
        wss = distracted.groupby("graph_node")["weight"].sum()

        # Initialise all nodes to 0
        drisi: dict[Any, float] = {n: 0.0 for n in G.nodes()}

        # Fill in computed values (log-normalise)
        for node, score in wss.items():
            if node in drisi:
                drisi[node] = float(np.log(score + 1))

        # Statistics
        values = np.array(list(drisi.values()))
        non_zero = (values > 0).sum()
        logger.info(
            "DR-ISI stats — non-zero: %d / %d nodes (%.1f%%)",
            non_zero,
            len(drisi),
            (non_zero / len(drisi) * 100) if drisi else 0,
        )
        if non_zero > 0:
            nz_vals = values[values > 0]
            logger.info(
                "DR-ISI (non-zero) — min: %.3f, mean: %.3f, max: %.3f",
                nz_vals.min(), nz_vals.mean(), nz_vals.max(),
            )

        return drisi

    def apply_to_graph(
        self,
        drisi: dict[Any, float],
        G: nx.MultiDiGraph,
    ) -> nx.MultiDiGraph:
        """
        Set the ``drisi`` attribute on every node in the graph.

        Parameters
        ----------
        drisi : dict
            Output of :meth:`compute`.
        G : nx.MultiDiGraph
            The road-network graph.

        Returns
        -------
        nx.MultiDiGraph
            The graph with ``drisi`` node attribute set.
        """
        nx.set_node_attributes(G, drisi, "drisi")
        logger.info("DR-ISI applied to all %d nodes", len(drisi))
        return G


# ============================================================================
# Stand-alone execution
# ============================================================================
if __name__ == "__main__":
    print("DRISICalculator — run via build_graph_pipeline.py")
