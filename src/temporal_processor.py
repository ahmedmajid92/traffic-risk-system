"""
Phase 3 — Spatio-Temporal Sequence Generation
===============================================

Transforms the static graph (Phase 2) and timestamped crash records
into temporal sliding-window sequences for a GCN-LSTM risk model.

Key design:
    - Sparse-first storage: crashes are aggregated into a scipy sparse
      (T, N) matrix (~few MB) instead of a dense tensor (~24 GB).
    - Lazy materialisation: each ``(24, N, 5)`` window is built on-the-fly
      inside ``__getitem__``, keeping peak RAM at ~2-3 GB.
    - Chronological 80/20 train/test split (no shuffle).

Dynamic features per timestep per node (5 total):
    crash_count, hour_sin, hour_cos, dow_sin, dow_cos

Target:
    Y_{t+1}(n) = log(1 + WSS_{t+1}(n))
    where WSS = sum of EPDO-weighted severities.

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
EPDO_WEIGHTS: dict[int, float] = {4: 12.0, 3: 3.0, 2: 1.0, 1: 1.0}
WINDOW_SIZE: int = 24       # hours of look-back
HORIZON: int = 1            # hours ahead to predict
NUM_DYNAMIC_FEATURES: int = 5
CRS_UTM_14N = "EPSG:32614"

# Date range (approved: full years only)
DATE_MIN = pd.Timestamp("2016-01-01")
DATE_MAX = pd.Timestamp("2022-12-31 23:59:59")


# ============================================================================
# TemporalAggregator
# ============================================================================

class TemporalAggregator:
    """
    Aggregate crash records into sparse temporal matrices.

    Produces two ``scipy.sparse.csr_matrix`` of shape ``(T, N)``:
        - ``crash_matrix``:  crash counts per (hour, node)
        - ``wss_matrix``:    EPDO-weighted severity sums per (hour, node)

    Also stores ``time_index``: a ``pd.DatetimeIndex`` of length T.
    """

    def __init__(
        self,
        crash_csv: Path,
        mapping_path: Path,
        graph_path: Path,
        mapped_csv: Path = Path("data/processed/crashes_mapped.csv"),
    ) -> None:
        self.crash_csv = crash_csv
        self.mapping_path = mapping_path
        self.graph_path = graph_path
        self.mapped_csv = mapped_csv

        with open(mapping_path, "rb") as f:
            self.node_mapping: dict[int, int] = pickle.load(f)
        self.num_nodes = len(self.node_mapping)

    # ------------------------------------------------------------------
    def aggregate(self) -> dict:
        """
        Run the full aggregation pipeline.

        Returns
        -------
        dict with keys:
            crash_matrix, wss_matrix, time_index, num_nodes, total_hours
        """
        df = self._get_mapped_crashes()
        df = self._filter_date_range(df)
        return self._build_sparse_matrices(df)

    # ------------------------------------------------------------------
    def _get_mapped_crashes(self) -> pd.DataFrame:
        """Load crash data with graph_node column (snap if needed)."""
        if self.mapped_csv.exists():
            logger.info("Loading pre-snapped crashes from %s", self.mapped_csv)
            df = pd.read_csv(self.mapped_csv)
            df["Start_Time"] = pd.to_datetime(df["Start_Time"])
            return df

        # Re-snap crashes to the graph
        logger.info("Snapping crashes to graph (first-time only) …")
        import osmnx as ox
        from crash_mapper import CrashMapper

        G = ox.load_graphml(self.graph_path)
        G = ox.project_graph(G, to_crs=CRS_UTM_14N)
        mapper = CrashMapper(G, crash_csv=self.crash_csv)
        df = mapper.snap()

        # Map OSMnx node-ids → contiguous indices
        df["node_idx"] = df["graph_node"].map(self.node_mapping)

        # Drop crashes that didn't map to LSCC nodes
        before = len(df)
        df = df.dropna(subset=["node_idx"]).copy()
        df["node_idx"] = df["node_idx"].astype(int)
        after = len(df)
        if before != after:
            logger.warning(
                "Dropped %d crashes outside LSCC", before - after
            )

        # Save intermediate file
        self.mapped_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.mapped_csv, index=False)
        logger.info("Saved mapped crashes → %s (%d rows)", self.mapped_csv, len(df))
        return df

    # ------------------------------------------------------------------
    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim to 2016-01-01 through 2022-12-31."""
        df["Start_Time"] = pd.to_datetime(df["Start_Time"])
        before = len(df)
        df = df[(df["Start_Time"] >= DATE_MIN) & (df["Start_Time"] <= DATE_MAX)].copy()
        logger.info(
            "Date filter: kept %d / %d records (%s → %s)",
            len(df), before, DATE_MIN.date(), DATE_MAX.date(),
        )
        return df

    # ------------------------------------------------------------------
    def _build_sparse_matrices(self, df: pd.DataFrame) -> dict:
        """
        Build sparse (T, N) matrices for crash counts and WSS.
        """
        # Floor timestamps to hour
        df["hour_floor"] = df["Start_Time"].dt.floor("h")

        # Continuous hourly time index
        t_min = DATE_MIN.floor("h")
        t_max = DATE_MAX.floor("h")
        time_index = pd.date_range(t_min, t_max, freq="h")
        total_hours = len(time_index)
        time_to_idx = {t: i for i, t in enumerate(time_index)}

        logger.info("Time index: %d hours (%s → %s)", total_hours, t_min, t_max)

        # Map each crash to (hour_idx, node_idx)
        df["hour_idx"] = df["hour_floor"].map(time_to_idx)
        df = df.dropna(subset=["hour_idx"]).copy()
        df["hour_idx"] = df["hour_idx"].astype(int)

        # EPDO weight per crash
        df["weight"] = df["Severity"].map(EPDO_WEIGHTS).fillna(1.0)

        # --- Crash count matrix ---
        crash_agg = df.groupby(["hour_idx", "node_idx"]).size().reset_index(name="count")
        crash_matrix = sp.csr_matrix(
            (crash_agg["count"].values,
             (crash_agg["hour_idx"].values, crash_agg["node_idx"].values)),
            shape=(total_hours, self.num_nodes),
        )

        # --- WSS matrix ---
        wss_agg = df.groupby(["hour_idx", "node_idx"])["weight"].sum().reset_index(name="wss")
        wss_matrix = sp.csr_matrix(
            (wss_agg["wss"].values,
             (wss_agg["hour_idx"].values, wss_agg["node_idx"].values)),
            shape=(total_hours, self.num_nodes),
        )

        # Stats
        nnz = crash_matrix.nnz
        total_cells = total_hours * self.num_nodes
        density = nnz / total_cells * 100
        logger.info(
            "Sparse matrices: nnz=%d, density=%.4f%%, size=%.1f MB",
            nnz, density,
            (crash_matrix.data.nbytes + crash_matrix.indices.nbytes +
             crash_matrix.indptr.nbytes) / 1e6,
        )

        return {
            "crash_matrix": crash_matrix,
            "wss_matrix": wss_matrix,
            "time_index": time_index,
            "num_nodes": self.num_nodes,
            "total_hours": total_hours,
        }


# ============================================================================
# TemporalFeatureBuilder
# ============================================================================

class TemporalFeatureBuilder:
    """
    Build cyclical time features for every hour in the time index.

    Produces a dense ``(T, 4)`` array:
        hour_sin, hour_cos, dow_sin, dow_cos

    (Crash count is node-specific and handled in the Dataset.)
    """

    @staticmethod
    def build(time_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Parameters
        ----------
        time_index : pd.DatetimeIndex
            Hourly timestamps of length T.

        Returns
        -------
        np.ndarray of shape (T, 4), dtype float32
        """
        hours = time_index.hour.values.astype(np.float32)
        dows = time_index.dayofweek.values.astype(np.float32)

        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        dow_sin = np.sin(2 * np.pi * dows / 7)
        dow_cos = np.cos(2 * np.pi * dows / 7)

        features = np.stack([hour_sin, hour_cos, dow_sin, dow_cos], axis=1)
        logger.info("Time features built: shape %s", features.shape)
        return features


# ============================================================================
# TemporalCrashDataset (Lazy-loading PyTorch Dataset)
# ============================================================================

class TemporalCrashDataset(Dataset):
    """
    Lazy-loading dataset for spatio-temporal crash prediction.

    Each sample ``(X, Y)`` represents:
        - X : ``(window_size, num_nodes, num_dynamic_features)``
              → 5 features: [crash_count, hour_sin, hour_cos, dow_sin, dow_cos]
        - Y : ``(num_nodes,)``
              → log(1 + WSS_{t+1}) per node

    Windows are materialised on-the-fly from sparse matrices to keep
    memory usage low (~2-3 GB peak).

    Parameters
    ----------
    crash_matrix : sp.csr_matrix
        (T, N) sparse crash counts.
    wss_matrix : sp.csr_matrix
        (T, N) sparse weighted severity sums.
    time_features : np.ndarray
        (T, 4) cyclical time encodings.
    start_idx, end_idx : int
        Inclusive range of valid first-timestep indices for windows.
    window_size : int
        Number of look-back hours (default 24).
    horizon : int
        Prediction horizon in hours (default 1).
    """

    def __init__(
        self,
        crash_matrix: sp.csr_matrix,
        wss_matrix: sp.csr_matrix,
        time_features: np.ndarray,
        start_idx: int,
        end_idx: int,
        window_size: int = WINDOW_SIZE,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        self.crash_matrix = crash_matrix
        self.wss_matrix = wss_matrix
        self.time_features = time_features
        self.window_size = window_size
        self.horizon = horizon
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.num_nodes = crash_matrix.shape[1]
        self._length = end_idx - start_idx

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        X : torch.FloatTensor of shape (window_size, num_nodes, 5)
        Y : torch.FloatTensor of shape (num_nodes,)
        """
        t = self.start_idx + idx

        # --- Features window: [t, t+window_size) ---
        # Crash counts: sparse → dense slice (window_size, N)
        crash_window = self.crash_matrix[t : t + self.window_size].toarray().astype(np.float32)

        # Time features: (window_size, 4) — broadcast across N
        time_window = self.time_features[t : t + self.window_size]  # (W, 4)

        # Expand time features to (window_size, N, 4)
        time_expanded = np.broadcast_to(
            time_window[:, np.newaxis, :],
            (self.window_size, self.num_nodes, 4),
        ).copy()  # copy to make contiguous

        # Stack: crash_count (W, N, 1) + time (W, N, 4) → (W, N, 5)
        X = np.concatenate(
            [crash_window[:, :, np.newaxis], time_expanded],
            axis=2,
        )

        # --- Target: WSS at t + window_size ---
        target_t = t + self.window_size
        wss_target = self.wss_matrix[target_t].toarray().flatten().astype(np.float32)
        Y = np.log1p(wss_target)  # log(1 + WSS)

        return torch.from_numpy(X), torch.from_numpy(Y)


# ============================================================================
# TemporalPipeline (Orchestrator)
# ============================================================================

class TemporalPipeline:
    """
    End-to-end pipeline: aggregate → features → split → save.

    Parameters
    ----------
    crash_csv : Path
        Annotated crash CSV from Phase 1.
    data_dir : Path
        Directory containing Phase 2 outputs and for saving results.
    train_ratio : float
        Fraction of timesteps for training (default 0.8).
    """

    def __init__(
        self,
        crash_csv: Path = Path("data/processed/dallas_crashes_annotated.csv"),
        data_dir: Path = Path("data/processed"),
        train_ratio: float = 0.8,
    ) -> None:
        self.crash_csv = crash_csv
        self.data_dir = data_dir
        self.train_ratio = train_ratio

    def run(self) -> None:
        """Execute the full pipeline."""
        t0 = time.perf_counter()

        # ==============================================================
        # Step 1: Temporal Aggregation
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 1 / 4 : Temporal Aggregation")
        print("=" * 70)

        aggregator = TemporalAggregator(
            crash_csv=self.crash_csv,
            mapping_path=self.data_dir / "node_mapping.pkl",
            graph_path=self.data_dir / "dallas_drive_net.graphml",
            mapped_csv=self.data_dir / "crashes_mapped.csv",
        )
        agg = aggregator.aggregate()

        crash_matrix = agg["crash_matrix"]
        wss_matrix = agg["wss_matrix"]
        time_index = agg["time_index"]
        num_nodes = agg["num_nodes"]
        total_hours = agg["total_hours"]

        print(f"  ✓ Total hours     : {total_hours:,}")
        print(f"  ✓ Num nodes       : {num_nodes:,}")
        print(f"  ✓ Non-zero entries: {crash_matrix.nnz:,}")
        print(f"  ✓ Density         : {crash_matrix.nnz / (total_hours * num_nodes) * 100:.4f}%")

        # ==============================================================
        # Step 2: Time Feature Engineering
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 2 / 4 : Cyclical Time Features")
        print("=" * 70)

        time_features = TemporalFeatureBuilder.build(time_index)
        print(f"  ✓ Time features shape: {time_features.shape}")

        # Sanity: midnight should have sin≈0, cos≈1
        midnight_idx = 0  # first entry is midnight Jan 1
        print(f"  ✓ Midnight check: sin={time_features[midnight_idx, 0]:.3f}, "
              f"cos={time_features[midnight_idx, 1]:.3f}")

        # ==============================================================
        # Step 3: Train/Test Split
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 3 / 4 : Chronological Train/Test Split")
        print("=" * 70)

        # Valid window range: [0, T - window_size - horizon)
        max_start = total_hours - WINDOW_SIZE - HORIZON
        split_idx = int(max_start * self.train_ratio)

        train_dataset = TemporalCrashDataset(
            crash_matrix, wss_matrix, time_features,
            start_idx=0, end_idx=split_idx,
        )
        test_dataset = TemporalCrashDataset(
            crash_matrix, wss_matrix, time_features,
            start_idx=split_idx, end_idx=max_start,
        )

        split_date = time_index[split_idx]
        print(f"  ✓ Split point     : hour {split_idx:,} ({split_date})")
        print(f"  ✓ Train samples   : {len(train_dataset):,}")
        print(f"  ✓ Test samples    : {len(test_dataset):,}")

        # Show sample shapes
        X_sample, Y_sample = train_dataset[0]
        print(f"  ✓ X sample shape  : {list(X_sample.shape)}  "
              f"(window={WINDOW_SIZE}, nodes={num_nodes}, features={NUM_DYNAMIC_FEATURES})")
        print(f"  ✓ Y sample shape  : {list(Y_sample.shape)}  (nodes={num_nodes})")
        print(f"  ✓ X dtype         : {X_sample.dtype}")
        print(f"  ✓ Y dtype         : {Y_sample.dtype}")

        # ==============================================================
        # Step 4: Save
        # ==============================================================
        print("\n" + "=" * 70)
        print("STEP 4 / 4 : Saving Datasets")
        print("=" * 70)

        # Load static features from Phase 2
        pyg_path = self.data_dir / "processed_graph_data.pt"
        static_data = torch.load(pyg_path, weights_only=False)

        # Save the shared temporal signal data
        signal_data = {
            "crash_matrix": crash_matrix,
            "wss_matrix": wss_matrix,
            "time_features": time_features,
            "time_index": time_index,
            "num_nodes": num_nodes,
            "total_hours": total_hours,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "split_idx": split_idx,
        }
        signal_path = self.data_dir / "temporal_signal.pt"
        torch.save(signal_data, signal_path)
        print(f"  ✓ Temporal signal → {signal_path}")

        # Save train dataset metadata
        train_meta = {
            "start_idx": 0,
            "end_idx": split_idx,
            "length": len(train_dataset),
            "edge_index": static_data.edge_index,
            "static_x": static_data.x,
            "edge_attr": static_data.edge_attr,
            "pos": static_data.pos,
            "node_feature_names": static_data.node_feature_names,
            "edge_feature_names": static_data.edge_feature_names,
        }
        train_path = self.data_dir / "train_dataset.pt"
        torch.save(train_meta, train_path)
        print(f"  ✓ Train dataset   → {train_path} ({len(train_dataset):,} samples)")

        # Save test dataset metadata
        test_meta = {
            "start_idx": split_idx,
            "end_idx": max_start,
            "length": len(test_dataset),
            "edge_index": static_data.edge_index,
            "static_x": static_data.x,
            "edge_attr": static_data.edge_attr,
            "pos": static_data.pos,
            "node_feature_names": static_data.node_feature_names,
            "edge_feature_names": static_data.edge_feature_names,
        }
        test_path = self.data_dir / "test_dataset.pt"
        torch.save(test_meta, test_path)
        print(f"  ✓ Test dataset    → {test_path} ({len(test_dataset):,} samples)")

        # ==============================================================
        # Summary
        # ==============================================================
        elapsed = time.perf_counter() - t0
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Date range        : {DATE_MIN.date()} → {DATE_MAX.date()}")
        print(f"  Total hours       : {total_hours:,}")
        print(f"  Window / Horizon  : {WINDOW_SIZE}h / {HORIZON}h")
        print(f"  Train samples     : {len(train_dataset):,}")
        print(f"  Test samples      : {len(test_dataset):,}")
        print(f"  X shape per sample: ({WINDOW_SIZE}, {num_nodes}, {NUM_DYNAMIC_FEATURES})")
        print(f"  Y shape per sample: ({num_nodes},)")
        print(f"  Dynamic features  : crash_count, hour_sin, hour_cos, dow_sin, dow_cos")
        print(f"  Target            : log(1 + WSS)")
        print(f"  Total time        : {elapsed:.1f}s")
        print("=" * 70 + "\n")


# ============================================================================
# Utility: Reconstruct dataset from saved files
# ============================================================================

def load_dataset(
    data_dir: Path = Path("data/processed"),
    split: str = "train",
) -> tuple[TemporalCrashDataset, dict]:
    """
    Reconstruct a TemporalCrashDataset from saved files.

    Parameters
    ----------
    data_dir : Path
        Directory containing temporal_signal.pt and {split}_dataset.pt.
    split : str
        Either "train" or "test".

    Returns
    -------
    dataset : TemporalCrashDataset
    meta : dict
        Contains edge_index, static_x, edge_attr, pos.
    """
    signal = torch.load(data_dir / "temporal_signal.pt", weights_only=False)
    meta = torch.load(data_dir / f"{split}_dataset.pt", weights_only=False)

    dataset = TemporalCrashDataset(
        crash_matrix=signal["crash_matrix"],
        wss_matrix=signal["wss_matrix"],
        time_features=signal["time_features"],
        start_idx=meta["start_idx"],
        end_idx=meta["end_idx"],
        window_size=signal["window_size"],
        horizon=signal["horizon"],
    )

    return dataset, meta


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Run the full Phase 3 temporal processing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: Temporal sequence generation for GCN-LSTM"
    )
    parser.add_argument(
        "--crash-csv",
        type=Path,
        default=Path("data/processed/dallas_crashes_annotated.csv"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()

    pipeline = TemporalPipeline(
        crash_csv=args.crash_csv,
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
