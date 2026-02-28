"""
Phase 7 — Baseline Benchmarking & Evaluation
===============================================

Academic rationale:
    To validate that the HybridSTGNN's spatio-temporal graph architecture
    provides measurable performance gains, we benchmark it against two
    standard baselines, each of which deliberately *removes* one data
    modality:

    ┌─────────────┬──────────────────────┬────────────────────────┐
    │ Model       │ Temporal Sequence?   │ Graph Connectivity?    │
    ├─────────────┼──────────────────────┼────────────────────────┤
    │ XGBoost     │ ✗ (flat features)    │ ✗ (per-node tabular)   │
    │ LSTM-Only   │ ✓ (24-h sequence)    │ ✗ (per-node series)    │
    │ HybridSTGNN │ ✓ (LSTM layer)       │ ✓ (GCN layers)         │
    └─────────────┴──────────────────────┴────────────────────────┘

    By comparing RMSE and MAE across these three models, we can
    quantify the marginal contribution of both temporal modelling
    (XGBoost → LSTM) and spatial graph modelling (LSTM → ST-GNN).

Data flattening:
    Both baselines treat each node independently.
    • XGBoost: X = (n_samples × N_nodes, W×F=120) — flat tabular rows.
    • LSTM:    X = (n_samples × N_nodes, W=24, F=5) — per-node sequences.

Memory-efficient sampling:
    Full data is ~24,697 nodes × ~2,000 windows = ~49M rows.  We randomly
    sample a subset of nodes (default: 500) and stride across windows to
    keep RAM and runtime manageable while remaining statistically fair
    (all models see the same sample).

Outputs:
    • Console comparison table (RMSE and MAE)
    • reports/figures/model_comparison.png — grouped bar chart
    • reports/metrics.json — machine-readable results

Usage:
    python src/evaluate_baselines.py
    python src/evaluate_baselines.py --n-nodes 200 --stride 12
    python src/evaluate_baselines.py --xgb-estimators 500 --lstm-epochs 10

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-28
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_processor import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# Constants
# =====================================================================

# ST-GNN reference metrics from Phase 4 test evaluation
# Obtained by running: python src/evaluate_model.py --stride 1
STGNN_METRICS = {
    "rmse": 0.0086,
    "mae": 0.0005,
}

# Feature names from Phase 3
FEATURE_NAMES = ["crash_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

# Output paths
FIG_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics.json")


# =====================================================================
# Step 1: Data Preparation
# =====================================================================

def prepare_flat_data(
    data_dir: Path,
    n_nodes: int = 500,
    stride: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Load temporal crash data and flatten it for baseline models.

    Strategy:
        1. Load the full train/test datasets via ``load_dataset()``.
        2. Randomly sample ``n_nodes`` nodes to keep memory manageable.
        3. For each sample, extract per-node features and targets.

    For XGBoost:
        X_flat = (n_windows × n_nodes, W × F)  — 24 × 5 = 120 features

    For LSTM:
        X_seq  = (n_windows × n_nodes, W, F)   — 24 timesteps × 5 features

    Both share the same Y = (n_windows × n_nodes,) targets.

    Parameters
    ----------
    data_dir : Path
        Directory with temporal_signal.pt and {split}_dataset.pt.
    n_nodes : int
        Number of nodes to sample (default 500).
    stride : int
        Step size across time windows (default 6).
    seed : int
        Random seed for reproducible node sampling.

    Returns
    -------
    X_train_flat : np.ndarray — (n_train × n_nodes, W×F)
    Y_train : np.ndarray — (n_train × n_nodes,)
    X_test_flat : np.ndarray — (n_test × n_nodes, W×F)
    Y_test : np.ndarray — (n_test × n_nodes,)
    window_size : int
    num_features : int
    """
    logger.info("Loading datasets from %s ...", data_dir)

    # Load full datasets
    train_dataset, _ = load_dataset(data_dir, split="train")
    test_dataset, _ = load_dataset(data_dir, split="test")

    window_size = train_dataset.window_size
    total_nodes = train_dataset.num_nodes
    num_features = 5  # crash_count, hour_sin, hour_cos, dow_sin, dow_cos

    logger.info("  Total nodes: %d, Window: %d, Features: %d",
                total_nodes, window_size, num_features)

    # --- Sample nodes ---
    rng = np.random.RandomState(seed)
    n_nodes = min(n_nodes, total_nodes)
    sampled_nodes = np.sort(rng.choice(total_nodes, size=n_nodes, replace=False))
    logger.info("  Sampled %d / %d nodes (seed=%d)", n_nodes, total_nodes, seed)

    # --- Extract train samples ---
    train_indices = list(range(0, len(train_dataset), stride))
    logger.info("  Train windows: %d (stride=%d)", len(train_indices), stride)

    X_train_list, Y_train_list = [], []
    for i, idx in enumerate(train_indices):
        X, Y = train_dataset[idx]
        # X: (W, N, F) → slice nodes → (W, n_nodes, F)
        X_sub = X[:, sampled_nodes, :].numpy()
        Y_sub = Y[sampled_nodes].numpy()
        X_train_list.append(X_sub)
        Y_train_list.append(Y_sub)

        if (i + 1) % 500 == 0:
            logger.info("    Train: %d / %d windows loaded", i + 1, len(train_indices))

    # Stack and reshape
    # X_train_3d: (n_windows, W, n_nodes, F) → reshape to per-node
    X_train_3d = np.stack(X_train_list, axis=0)  # (n_win, W, n_nodes, F)
    Y_train_2d = np.stack(Y_train_list, axis=0)  # (n_win, n_nodes)

    n_train = X_train_3d.shape[0]

    # Flatten to per-node samples: (n_win × n_nodes, ...)
    # Transpose to (n_win, n_nodes, W, F) then reshape
    X_train_4d = X_train_3d.transpose(0, 2, 1, 3)  # (n_win, n_nodes, W, F)
    X_train_seq = X_train_4d.reshape(n_train * n_nodes, window_size, num_features)
    X_train_flat = X_train_seq.reshape(n_train * n_nodes, window_size * num_features)
    Y_train = Y_train_2d.reshape(n_train * n_nodes)

    logger.info("  Train: X_flat=%s, X_seq=%s, Y=%s",
                X_train_flat.shape, X_train_seq.shape, Y_train.shape)

    # --- Extract test samples ---
    test_indices = list(range(0, len(test_dataset), stride))
    logger.info("  Test windows: %d (stride=%d)", len(test_indices), stride)

    X_test_list, Y_test_list = [], []
    for i, idx in enumerate(test_indices):
        X, Y = test_dataset[idx]
        X_sub = X[:, sampled_nodes, :].numpy()
        Y_sub = Y[sampled_nodes].numpy()
        X_test_list.append(X_sub)
        Y_test_list.append(Y_sub)

        if (i + 1) % 500 == 0:
            logger.info("    Test: %d / %d windows loaded", i + 1, len(test_indices))

    X_test_3d = np.stack(X_test_list, axis=0)
    Y_test_2d = np.stack(Y_test_list, axis=0)

    n_test = X_test_3d.shape[0]

    X_test_4d = X_test_3d.transpose(0, 2, 1, 3)
    X_test_seq = X_test_4d.reshape(n_test * n_nodes, window_size, num_features)
    X_test_flat = X_test_seq.reshape(n_test * n_nodes, window_size * num_features)
    Y_test = Y_test_2d.reshape(n_test * n_nodes)

    logger.info("  Test:  X_flat=%s, X_seq=%s, Y=%s",
                X_test_flat.shape, X_test_seq.shape, Y_test.shape)

    return X_train_flat, Y_train, X_test_flat, Y_test, window_size, num_features


# =====================================================================
# Step 2: XGBoost Baseline
# =====================================================================

def train_xgboost(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict:
    """
    Train and evaluate an XGBoost Regressor baseline.

    Theoretical rationale:
        XGBoost is the gold standard for tabular data.  By flattening the
        24-hour × 5-feature window into 120 flat features and treating each
        node independently, this baseline deliberately ignores:

        1. **Temporal ordering** — XGBoost sees "crash_count_t0, crash_count_t1, ..."
           as unordered features; it cannot learn sequential recurrence the
           way an LSTM can.
        2. **Spatial graph structure** — each node is an independent sample;
           the model has no way to pass information between neighbouring
           intersections via the road network.

        If the ST-GNN outperforms XGBoost, it validates that the graph +
        sequence architecture extracts information that flat tabular models
        cannot capture.

    Parameters
    ----------
    X_train, Y_train : np.ndarray
        Training data. X is (n_samples, W*F=120).
    X_test, Y_test : np.ndarray
        Test data.
    n_estimators : int
        Number of boosting rounds (kept low for quick benchmarking).
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.

    Returns
    -------
    dict with 'rmse', 'mae', 'train_time'
    """
    from xgboost import XGBRegressor

    logger.info("\n" + "=" * 60)
    logger.info("  BASELINE 1: XGBoost Regressor")
    logger.info("  Config: n_estimators=%d, max_depth=%d, lr=%.3f",
                n_estimators, max_depth, learning_rate)
    logger.info("=" * 60)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",       # Fast histogram-based method
        n_jobs=-1,                # Use all CPU cores
        random_state=42,
        verbosity=1,
    )

    logger.info("  Training on %d samples ...", X_train.shape[0])
    t0 = time.perf_counter()
    model.fit(X_train, Y_train)
    train_time = time.perf_counter() - t0
    logger.info("  Training complete in %.1fs", train_time)

    # Predict on test set
    predictions = model.predict(X_test)

    # Compute metrics
    errors = predictions - Y_test
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    logger.info("  Results: RMSE=%.6f, MAE=%.6f", rmse, mae)

    return {"rmse": rmse, "mae": mae, "train_time": train_time}


# =====================================================================
# Step 3: LSTM-Only Baseline
# =====================================================================

class LSTMBaseline(nn.Module):
    """
    A purely temporal LSTM baseline for node-level risk prediction.

    Theoretical rationale:
        This model processes each node's 24-hour feature sequence with
        an LSTM, capturing temporal dependencies (e.g., crash frequency
        trends, time-of-day effects).  However, it treats every node
        as an **independent time series** — it has NO access to the
        road network graph.

        Unlike the HybridSTGNN, which first encodes spatial context
        via GCN message passing before feeding into the LSTM, this
        baseline's LSTM only sees one node's own history.

        If the ST-GNN outperforms this LSTM, it proves that the GCN
        layers' ability to aggregate neighbourhood information (crash
        patterns at adjacent intersections) provides a measurable
        benefit over purely temporal modelling.

    Architecture:
        Input (B, W=24, F=5)
          → LSTM (2 layers, hidden_dim=64)
          → Last hidden state (B, 64)
          → Linear (64 → 1)
          → Scalar risk score (B,)

    Parameters
    ----------
    in_features : int
        Number of input features per timestep (default 5).
    hidden_dim : int
        LSTM hidden size (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout between LSTM layers (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, W, F)
            Per-node time series.

        Returns
        -------
        torch.Tensor of shape (B,)
            Predicted risk score per node.
        """
        # LSTM output: (B, W, hidden_dim), last hidden: (num_layers, B, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep's output
        last_output = lstm_out[:, -1, :]  # (B, hidden_dim)
        # Project to scalar
        return self.head(last_output).squeeze(-1)  # (B,)


def train_lstm(
    X_train_flat: np.ndarray,
    Y_train: np.ndarray,
    X_test_flat: np.ndarray,
    Y_test: np.ndarray,
    window_size: int,
    num_features: int,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
) -> dict:
    """
    Train and evaluate a per-node LSTM baseline.

    Parameters
    ----------
    X_train_flat, Y_train : np.ndarray
        Training data. X is (n_samples, W*F) — will be reshaped to (n, W, F).
    X_test_flat, Y_test : np.ndarray
        Test data.
    window_size : int
        Lookback window length (24).
    num_features : int
        Features per timestep (5).
    epochs : int
        Number of training epochs (kept low for fast benchmarking).
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    hidden_dim : int
        LSTM hidden dimension.

    Returns
    -------
    dict with 'rmse', 'mae', 'train_time'
    """
    logger.info("\n" + "=" * 60)
    logger.info("  BASELINE 2: LSTM-Only (No Graph)")
    logger.info("  Config: epochs=%d, hidden=%d, batch=%d, lr=%.4f",
                epochs, hidden_dim, batch_size, lr)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)

    # Reshape flat features back to sequences: (n, W*F) → (n, W, F)
    X_train_seq = X_train_flat.reshape(-1, window_size, num_features)
    X_test_seq = X_test_flat.reshape(-1, window_size, num_features)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_seq).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_test_t = torch.from_numpy(X_test_seq).float()
    Y_test_t = torch.from_numpy(Y_test).float()

    # Create DataLoaders
    train_data = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Build model
    model = LSTMBaseline(
        in_features=num_features,
        hidden_dim=hidden_dim,
    ).to(device)

    logger.info("  Parameters: %d",
                sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- Training loop ---
    logger.info("  Training on %d samples (%d batches/epoch) ...",
                len(train_data), len(train_loader))

    t0 = time.perf_counter()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred = model(X_batch)
            loss = criterion(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        logger.info("    Epoch %d/%d — Loss: %.6f", epoch + 1, epochs, avg_loss)

    train_time = time.perf_counter() - t0
    logger.info("  Training complete in %.1fs", train_time)

    # --- Evaluation ---
    model.eval()
    all_preds = []

    with torch.no_grad():
        # Evaluate in batches to avoid OOM
        for i in range(0, len(X_test_t), batch_size):
            batch = X_test_t[i : i + batch_size].to(device)
            pred = model(batch).cpu()
            all_preds.append(pred)

    predictions = torch.cat(all_preds).numpy()
    targets = Y_test

    errors = predictions - targets
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    logger.info("  Results: RMSE=%.6f, MAE=%.6f", rmse, mae)

    return {"rmse": rmse, "mae": mae, "train_time": train_time}


# =====================================================================
# Step 4: Comparison Table & Visualisation
# =====================================================================

def print_comparison_table(results: dict[str, dict]) -> None:
    """
    Print a formatted comparison table to the console.

    Parameters
    ----------
    results : dict
        Keys are model names, values are dicts with 'rmse' and 'mae'.
    """
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON — Baseline Benchmarking (Phase 7)")
    print("=" * 70)
    print(f"  {'Model':<20} {'RMSE':>12} {'MAE':>12} {'Train Time':>14}")
    print("  " + "-" * 58)

    for name, metrics in results.items():
        train_t = metrics.get("train_time", None)
        time_str = f"{train_t:.1f}s" if train_t is not None else "—"
        print(f"  {name:<20} {metrics['rmse']:>12.6f} {metrics['mae']:>12.6f} {time_str:>14}")

    print("=" * 70)

    # Highlight the winner
    best_rmse_model = min(results, key=lambda k: results[k]["rmse"])
    best_mae_model = min(results, key=lambda k: results[k]["mae"])
    print(f"\n  🏆 Best RMSE: {best_rmse_model} ({results[best_rmse_model]['rmse']:.6f})")
    print(f"  🏆 Best MAE:  {best_mae_model} ({results[best_mae_model]['mae']:.6f})")

    # Improvement percentages over the weakest baseline
    worst_rmse = max(v["rmse"] for v in results.values())
    stgnn_rmse = results.get("HybridSTGNN", {}).get("rmse", None)
    if stgnn_rmse and worst_rmse > 0:
        improvement = (worst_rmse - stgnn_rmse) / worst_rmse * 100
        print(f"\n  📊 ST-GNN RMSE improvement over weakest baseline: {improvement:.1f}%")

    print()


def plot_comparison(results: dict[str, dict], save_path: Path) -> None:
    """
    Generate a grouped bar chart comparing RMSE and MAE across models.

    Parameters
    ----------
    results : dict
        Keys are model names, values are dicts with 'rmse' and 'mae'.
    save_path : Path
        Path to save the PNG figure.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    rmse_vals = [results[m]["rmse"] for m in models]
    mae_vals = [results[m]["mae"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- RMSE bar chart ---
    colors_rmse = ["#2196F3", "#FF9800", "#4CAF50"]
    bars1 = ax1.bar(x, rmse_vals, width=0.6, color=colors_rmse[:len(models)],
                    edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title("RMSE Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- MAE bar chart ---
    colors_mae = ["#2196F3", "#FF9800", "#4CAF50"]
    bars2 = ax2.bar(x, mae_vals, width=0.6, color=colors_mae[:len(models)],
                    edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("MAE", fontsize=12)
    ax2.set_title("MAE Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Phase 7: Baseline Benchmarking — Model Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  📊 Saved comparison chart: %s", save_path)


def save_metrics_json(results: dict[str, dict], save_path: Path) -> None:
    """
    Save all model metrics to a JSON file.

    Parameters
    ----------
    results : dict
        Keys are model names, values are metric dicts.
    save_path : Path
        Path to save the JSON file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up for JSON serialization (remove non-essential keys)
    output = {}
    for name, metrics in results.items():
        output[name] = {
            "rmse": round(metrics["rmse"], 6),
            "mae": round(metrics["mae"], 6),
        }
        if "train_time" in metrics:
            output[name]["train_time_seconds"] = round(metrics["train_time"], 1)

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("  💾 Saved metrics: %s", save_path)


# =====================================================================
# Main Pipeline
# =====================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7: Train baseline models and compare with ST-GNN",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-nodes", type=int, default=500,
                        help="Number of nodes to sample (default: 500)")
    parser.add_argument("--stride", type=int, default=6,
                        help="Stride across time windows (default: 6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for node sampling")

    # XGBoost hyperparameters
    parser.add_argument("--xgb-estimators", type=int, default=200,
                        help="XGBoost: number of boosting rounds")
    parser.add_argument("--xgb-depth", type=int, default=6,
                        help="XGBoost: max tree depth")
    parser.add_argument("--xgb-lr", type=float, default=0.1,
                        help="XGBoost: learning rate")

    # LSTM hyperparameters
    parser.add_argument("--lstm-epochs", type=int, default=5,
                        help="LSTM: number of training epochs")
    parser.add_argument("--lstm-hidden", type=int, default=64,
                        help="LSTM: hidden dimension")
    parser.add_argument("--lstm-batch", type=int, default=256,
                        help="LSTM: batch size")
    parser.add_argument("--lstm-lr", type=float, default=1e-3,
                        help="LSTM: learning rate")

    args = parser.parse_args()

    t_start = time.perf_counter()

    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 7: Baseline Benchmarking & Evaluation")
    logger.info("=" * 60)

    # =================================================================
    # Step 1: Prepare Data
    # =================================================================
    logger.info("\n  STEP 1: Preparing flat data for baselines")

    X_train_flat, Y_train, X_test_flat, Y_test, window_size, num_features = \
        prepare_flat_data(
            data_dir=args.data_dir,
            n_nodes=args.n_nodes,
            stride=args.stride,
            seed=args.seed,
        )

    # =================================================================
    # Step 2: Train XGBoost
    # =================================================================
    xgb_metrics = train_xgboost(
        X_train_flat, Y_train,
        X_test_flat, Y_test,
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_depth,
        learning_rate=args.xgb_lr,
    )

    # =================================================================
    # Step 3: Train LSTM-Only
    # =================================================================
    lstm_metrics = train_lstm(
        X_train_flat, Y_train,
        X_test_flat, Y_test,
        window_size=window_size,
        num_features=num_features,
        epochs=args.lstm_epochs,
        batch_size=args.lstm_batch,
        lr=args.lstm_lr,
        hidden_dim=args.lstm_hidden,
    )

    # =================================================================
    # Step 4: Compile Results
    # =================================================================
    results = {
        "HybridSTGNN": STGNN_METRICS,
        "XGBoost": xgb_metrics,
        "LSTM-Only": lstm_metrics,
    }

    # =================================================================
    # Step 5: Output
    # =================================================================
    print_comparison_table(results)
    plot_comparison(results, FIG_DIR / "model_comparison.png")
    save_metrics_json(results, METRICS_PATH)

    elapsed = time.perf_counter() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("  ✓ PHASE 7 COMPLETE — %.0fs elapsed", elapsed)
    logger.info("  Chart: %s", FIG_DIR / "model_comparison.png")
    logger.info("  Metrics: %s", METRICS_PATH)
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
