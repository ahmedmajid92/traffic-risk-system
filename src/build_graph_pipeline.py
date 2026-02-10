"""
Phase 2 — Orchestrator: Build Graph Pipeline
==============================================

Ties all Phase 2 modules together into a single end-to-end pipeline:

    1. GraphBuilder   → construct & feature-enrich the road network
    2. CrashMapper    → snap annotated crashes to graph nodes
    3. DRISICalculator → compute distraction-risk target variable
    4. PyGConverter    → export to PyTorch Geometric Data object

Run from the project root:

    python src/build_graph_pipeline.py

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full Phase 2 graph-construction pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Build spatial graph, snap crashes, compute DR-ISI, export PyG"
    )
    parser.add_argument(
        "--crash-csv",
        type=Path,
        default=Path("data/processed/dallas_crashes_annotated.csv"),
        help="Path to the annotated crash CSV from Phase 1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for all output files",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the OSM graph even if a cached copy exists",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "dallas_drive_net.graphml"
    mapping_path = output_dir / "node_mapping.pkl"
    pyg_path = output_dir / "processed_graph_data.pt"
    crash_csv: Path = args.crash_csv

    # Validate crash CSV exists
    if not crash_csv.exists():
        logger.error("Crash CSV not found: %s", crash_csv)
        logger.error("Run Phase 1 (src/data_loader.py) first.")
        sys.exit(1)

    t0 = time.perf_counter()

    # ==================================================================
    # Step 1: Graph Construction
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 1 / 4 : Building Road Network Graph")
    print("=" * 70)

    from graph_builder import GraphBuilder

    builder = GraphBuilder(graph_path=graph_path, mapping_path=mapping_path)
    G = builder.build(force_download=args.force_download)

    print(f"  ✓ Nodes: {G.number_of_nodes():,}")
    print(f"  ✓ Edges: {G.number_of_edges():,}")

    # ==================================================================
    # Step 2: Crash Snapping
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2 / 4 : Snapping Crashes to Graph Nodes")
    print("=" * 70)

    from crash_mapper import CrashMapper

    mapper = CrashMapper(G, crash_csv=crash_csv)
    crash_df = mapper.snap()

    print(f"  ✓ Crashes snapped: {len(crash_df):,}")

    node_agg = CrashMapper.aggregate_per_node(crash_df)
    nodes_with_crashes = len(node_agg)
    pct_nodes = (nodes_with_crashes / G.number_of_nodes()) * 100
    print(f"  ✓ Nodes with crashes: {nodes_with_crashes:,} / {G.number_of_nodes():,} ({pct_nodes:.1f}%)")

    # ==================================================================
    # Step 3: DR-ISI Computation
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3 / 4 : Computing DR-ISI Target Variable")
    print("=" * 70)

    from drisi_calculator import DRISICalculator

    drisi_calc = DRISICalculator()
    drisi_scores = drisi_calc.compute(crash_df, G)
    G = drisi_calc.apply_to_graph(drisi_scores, G)

    import numpy as np
    vals = np.array(list(drisi_scores.values()))
    non_zero = (vals > 0).sum()
    print(f"  ✓ Nodes with DR-ISI > 0: {non_zero:,} / {len(vals):,}")
    if non_zero > 0:
        nz = vals[vals > 0]
        print(f"  ✓ DR-ISI range: [{nz.min():.3f}, {nz.max():.3f}]")
        print(f"  ✓ DR-ISI mean (non-zero): {nz.mean():.3f}")

    # ==================================================================
    # Step 4: PyG Conversion
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4 / 4 : Converting to PyTorch Geometric Data")
    print("=" * 70)

    from pyg_converter import PyGConverter

    converter = PyGConverter(output_path=pyg_path)
    data = converter.convert(G, builder.node_mapping)

    print(f"  ✓ x (node features) : {list(data.x.shape)}")
    print(f"  ✓ edge_index        : {list(data.edge_index.shape)}")
    print(f"  ✓ edge_attr         : {list(data.edge_attr.shape)}")
    print(f"  ✓ y (DR-ISI target) : {list(data.y.shape)}")
    print(f"  ✓ pos (UTM coords)  : {list(data.pos.shape)}")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time         : {elapsed:.1f}s")
    print(f"  Graph              : {graph_path}")
    print(f"  Node mapping       : {mapping_path}")
    print(f"  PyG Data           : {pyg_path}")
    print(f"  Node features ({len(data.node_feature_names)}) : {data.node_feature_names}")
    print(f"  Edge features ({len(data.edge_feature_names)}) : {data.edge_feature_names}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
