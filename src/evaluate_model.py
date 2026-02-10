"""
Phase 4 — Standalone Model Evaluation
=======================================

Loads a saved HybridSTGNN checkpoint and evaluates it on the test set.
Can be run independently after training is complete.

Usage:
    python src/evaluate_model.py
    python src/evaluate_model.py --model-path models/best_stgnn_model.pth
    python src/evaluate_model.py --split test --stride 1  # full (no stride) test eval

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_architecture import HybridSTGNN
from temporal_processor import load_dataset


# =====================================================================
# Evaluation Metrics
# =====================================================================

@torch.no_grad()
def evaluate(
    model: HybridSTGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Full evaluation with RMSE, MAE, and per-category metrics.

    Returns
    -------
    dict with rmse, mae, rmse_nonzero, mae_nonzero, pct_nonzero_correct
    """
    model.eval()

    all_preds = []
    all_targets = []
    total_steps = len(loader)
    t0 = time.perf_counter()

    for step, (X_batch, Y_batch) in enumerate(loader):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        pred = model(X_batch, edge_index)  # → (B, N)
        all_preds.append(pred.cpu())
        all_targets.append(Y_batch.cpu())

        if (step + 1) % 50 == 0 or (step + 1) == total_steps:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / (step + 1)) * (total_steps - step - 1)
            print(
                f"  [Eval] Step {step+1:4d}/{total_steps} "
                f"| {elapsed:.0f}s elapsed | ETA: {eta:.0f}s",
                flush=True,
            )

    preds = torch.cat(all_preds, dim=0).numpy()    # (N_samples, N_nodes)
    targets = torch.cat(all_targets, dim=0).numpy()

    # --- Overall metrics ---
    errors = preds - targets
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    # --- Non-zero target metrics (the important ones) ---
    nonzero_mask = targets > 0
    n_nonzero = nonzero_mask.sum()

    if n_nonzero > 0:
        rmse_nz = np.sqrt(np.mean(errors[nonzero_mask] ** 2))
        mae_nz = np.mean(np.abs(errors[nonzero_mask]))
    else:
        rmse_nz = 0.0
        mae_nz = 0.0

    # --- Zero-target metrics ---
    zero_mask = targets == 0
    n_zero = zero_mask.sum()

    if n_zero > 0:
        # How often does the model correctly predict near-zero for zero targets?
        near_zero_threshold = 0.1
        correct_zeros = (np.abs(preds[zero_mask]) < near_zero_threshold).sum()
        pct_zero_correct = correct_zeros / n_zero * 100
    else:
        pct_zero_correct = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "rmse_nonzero": rmse_nz,
        "mae_nonzero": mae_nz,
        "n_total": targets.size,
        "n_nonzero": int(n_nonzero),
        "pct_nonzero": n_nonzero / targets.size * 100,
        "pct_zero_correct": pct_zero_correct,
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
    }


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a trained HybridSTGNN model on the test set"
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("models/best_stgnn_model.pth"),
        help="Path to saved model checkpoint",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--stride", type=int, default=6,
        help="Stride for sub-sampling (1 = full evaluation, 6 = faster)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # --- Check model file exists ---
    if not args.model_path.exists():
        print(f"\n❌ Model file not found: {args.model_path}")
        print("   Train the model first with: python src/train_model.py")
        sys.exit(1)

    # --- Load data ---
    print(f"\n  Loading {args.split} dataset...")
    dataset, meta = load_dataset(args.data_dir, split=args.split)
    edge_index = meta["edge_index"].to(device)

    # Apply stride
    if args.stride > 1:
        indices = list(range(0, len(dataset), args.stride))
        dataset = Subset(dataset, indices)

    print(f"  Samples: {len(dataset):,} (stride={args.stride})")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # --- Load model ---
    sample_x, _ = dataset[0]
    in_features = sample_x.shape[-1]

    model = HybridSTGNN(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  ✓ Model loaded from {args.model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Evaluate ---
    print(f"\n{'=' * 70}")
    print(f"  EVALUATING ON {args.split.upper()} SET")
    print(f"{'=' * 70}\n")

    t0 = time.perf_counter()
    metrics = evaluate(model, loader, edge_index, device)
    elapsed = time.perf_counter() - t0

    # --- Print results ---
    print(f"\n{'=' * 70}")
    print(f"  RESULTS ({args.split.upper()} SET)")
    print(f"{'=' * 70}")
    print(f"  Overall RMSE         : {metrics['rmse']:.4f}")
    print(f"  Overall MAE          : {metrics['mae']:.4f}")
    print(f"  Non-zero RMSE        : {metrics['rmse_nonzero']:.4f}")
    print(f"  Non-zero MAE         : {metrics['mae_nonzero']:.4f}")
    print(f"  ---")
    print(f"  Total predictions    : {metrics['n_total']:,}")
    print(f"  Non-zero targets     : {metrics['n_nonzero']:,} ({metrics['pct_nonzero']:.2f}%)")
    print(f"  Zero-target accuracy : {metrics['pct_zero_correct']:.1f}%")
    print(f"  ---")
    print(f"  Pred  mean/std       : {metrics['pred_mean']:.4f} / {metrics['pred_std']:.4f}")
    print(f"  Target mean/std      : {metrics['target_mean']:.4f} / {metrics['target_std']:.4f}")
    print(f"  ---")
    print(f"  Evaluation time      : {elapsed:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
