"""
Phase 4 — Module 2: Training Loop & Evaluation
=================================================

Trains the HybridSTGNN model on temporal crash-risk data using:
    - Weighted MSE loss  (10× weight on non-zero targets)
    - Adam optimiser     (lr = 0.001)
    - ReduceLROnPlateau  (patience = 3)
    - Early stopping     (patience = 5)
    - Gradient clipping  (max_norm = 1.0)
    - Gradient accumulation (effective batch = batch_size × accum_steps)

Metrics logged per epoch: Train Loss, Val Loss, Val RMSE, Val MAE.

Author: Traffic Risk System — ST-GNN Project
Date: 2026-02-10
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_architecture import HybridSTGNN
from temporal_processor import load_dataset


# =====================================================================
# Weighted MSE Loss
# =====================================================================

def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """
    MSE loss with higher weight on non-zero targets.

    Addresses extreme target sparsity (~87% nodes have zero risk)
    so the model doesn't collapse to predicting 0 everywhere.

    Parameters
    ----------
    pred   : (B, N) — predicted risk scores
    target : (B, N) — ground-truth log(1 + WSS)
    pos_weight : float — weight multiplier for non-zero targets

    Returns
    -------
    scalar loss
    """
    weights = torch.where(target > 0, pos_weight, 1.0)     # → (B, N)
    loss = weights * (pred - target) ** 2                   # → (B, N)
    return loss.mean()


# =====================================================================
# Trainer Class
# =====================================================================

class Trainer:
    """
    End-to-end training manager for HybridSTGNN.

    Uses gradient accumulation to achieve an effective batch size
    larger than what fits in VRAM.

    Parameters
    ----------
    model : HybridSTGNN
    edge_index : torch.Tensor — (2, E) static graph
    device : torch.device
    lr : float
    pos_weight : float — weight for non-zero targets in loss
    clip_norm : float — gradient clipping max norm
    accum_steps : int — gradient accumulation steps
    scheduler_patience : int
    early_stop_patience : int
    """

    def __init__(
        self,
        model: HybridSTGNN,
        edge_index: torch.Tensor,
        device: torch.device,
        lr: float = 0.001,
        pos_weight: float = 10.0,
        clip_norm: float = 1.0,
        accum_steps: int = 4,
        scheduler_patience: int = 3,
        early_stop_patience: int = 5,
    ) -> None:
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.clip_norm = clip_norm
        self.accum_steps = accum_steps
        self.early_stop_patience = early_stop_patience

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=scheduler_patience,
            factor=0.5, verbose=False,
        )

        # Early stopping state
        self.best_val_loss = float("inf")
        self.epochs_without_improve = 0
        self.best_state_dict: dict | None = None

    # -----------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader) -> float:
        """
        Run one training epoch with gradient accumulation.

        Effective batch = loader.batch_size × self.accum_steps
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for step, (X_batch, Y_batch) in enumerate(loader):
            # X_batch: (B, W, N, F), Y_batch: (B, N)
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            pred = self.model(X_batch, self.edge_index)     # → (B, N)
            loss = weighted_mse_loss(pred, Y_batch, self.pos_weight)

            # Scale loss by accumulation steps for correct gradient magnitude
            scaled_loss = loss / self.accum_steps
            scaled_loss.backward()

            total_loss += loss.item()
            n_batches += 1

            # Optimizer step every accum_steps (or at end of epoch)
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / max(n_batches, 1)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Run validation. Returns dict with val_loss, rmse, mae."""
        self.model.eval()
        total_loss = 0.0
        total_se = 0.0
        total_ae = 0.0
        total_count = 0
        n_batches = 0

        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            pred = self.model(X_batch, self.edge_index)

            loss = weighted_mse_loss(pred, Y_batch, self.pos_weight)
            total_loss += loss.item()
            n_batches += 1

            se = (pred - Y_batch) ** 2
            ae = (pred - Y_batch).abs()
            total_se += se.sum().item()
            total_ae += ae.sum().item()
            total_count += Y_batch.numel()

        mean_loss = total_loss / max(n_batches, 1)
        rmse = math.sqrt(total_se / max(total_count, 1))
        mae = total_ae / max(total_count, 1)

        return {"val_loss": mean_loss, "rmse": rmse, "mae": mae}

    # -----------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_path: Path = Path("models/best_stgnn_model.pth"),
    ) -> dict:
        """
        Full training loop with validation, scheduling, and early stopping.

        Returns
        -------
        history : dict with lists of per-epoch metrics
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [], "val_loss": [],
            "rmse": [], "mae": [], "lr": [],
        }

        eff_batch = train_loader.batch_size * self.accum_steps

        print("\n" + "=" * 80)
        print(f"  Training HybridSTGNN on {self.device}")
        print(f"  Parameters      : {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Batch size      : {train_loader.batch_size} × {self.accum_steps} accum = {eff_batch}")
        print(f"  Train steps     : {len(train_loader)}/epoch")
        print(f"  Val steps       : {len(val_loader)}/epoch")
        print("=" * 80 + "\n")

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            # --- Train ---
            train_loss = self.train_one_epoch(train_loader)

            # --- Validate ---
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]

            # --- Scheduler step ---
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Early stopping check ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improve = 0
                self.best_state_dict = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                torch.save(self.best_state_dict, save_path)
                marker = " ★ saved"
            else:
                self.epochs_without_improve += 1
                marker = ""

            elapsed = time.perf_counter() - t0

            # --- Log ---
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["rmse"].append(val_metrics["rmse"])
            history["mae"].append(val_metrics["mae"])
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:3d}/{epochs} │ "
                f"Train Loss: {train_loss:.6f} │ "
                f"Val Loss: {val_loss:.6f} │ "
                f"RMSE: {val_metrics['rmse']:.4f} │ "
                f"MAE: {val_metrics['mae']:.4f} │ "
                f"LR: {current_lr:.1e} │ "
                f"{elapsed:.0f}s{marker}"
            )

            if self.epochs_without_improve >= self.early_stop_patience:
                print(
                    f"\n⛔ Early stopping at epoch {epoch} "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                break

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print(f"\n✓ Best model restored (val_loss={self.best_val_loss:.6f})")
            print(f"✓ Checkpoint saved → {save_path}")

        return history


# =====================================================================
# DataLoader Helpers
# =====================================================================

def make_strided_subset(dataset, stride: int) -> Subset:
    """
    Sub-sample a dataset by taking every ``stride``-th sample.
    Preserves chronological order.
    """
    indices = list(range(0, len(dataset), stride))
    return Subset(dataset, indices)


def create_data_loaders(
    data_dir: Path = Path("data/processed"),
    batch_size: int = 2,
    stride: int = 6,
    val_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test DataLoaders.

    Splits the Phase 3 train set chronologically into train/val.

    Returns
    -------
    train_loader, val_loader, test_loader, meta
    """
    full_train_ds, train_meta = load_dataset(data_dir, split="train")
    test_ds, test_meta = load_dataset(data_dir, split="test")

    # Chronological train/val split (85% train, 15% val)
    total_train = len(full_train_ds)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, total_train))

    train_subset = Subset(full_train_ds, train_indices)
    val_subset = Subset(full_train_ds, val_indices)

    # Apply stride (reduces samples for faster training)
    if stride > 1:
        train_subset = make_strided_subset(
            Subset(full_train_ds, train_indices), stride
        )

    # Apply stride to val/test for faster evaluation
    if stride > 1:
        val_subset = make_strided_subset(
            Subset(full_train_ds, val_indices), stride
        )
        test_subset = make_strided_subset(test_ds, stride)
    else:
        test_subset = test_ds

    print(f"\n  Dataset sizes (stride={stride}):")
    print(f"    Train : {len(train_subset):,} samples")
    print(f"    Val   : {len(val_subset):,} samples")
    print(f"    Test  : {len(test_subset):,} samples")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_meta


# =====================================================================
# Test Evaluation
# =====================================================================

@torch.no_grad()
def evaluate_test(
    model: HybridSTGNN,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run final evaluation on the test set."""
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    total_count = 0

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        pred = model(X_batch, edge_index)
        total_se += ((pred - Y_batch) ** 2).sum().item()
        total_ae += (pred - Y_batch).abs().sum().item()
        total_count += Y_batch.numel()

    rmse = math.sqrt(total_se / max(total_count, 1))
    mae = total_ae / max(total_count, 1)

    return {"rmse": rmse, "mae": mae}


# =====================================================================
# Main Entry Point
# =====================================================================

def main() -> None:
    """Train the HybridSTGNN model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4: Train HybridSTGNN for crash risk prediction"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Micro-batch size per GPU step (default 2)")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × accum)")
    parser.add_argument("--stride", type=int, default=6,
                        help="Sample every N-th window (default 6)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Weight for non-zero targets in loss")
    parser.add_argument("--save-path", type=Path,
                        default=Path("models/best_stgnn_model.pth"))
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # --- Data ---
    train_loader, val_loader, test_loader, meta = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        stride=args.stride,
    )
    edge_index = meta["edge_index"]

    # --- Model ---
    sample_x, _ = train_loader.dataset[0]
    in_features = sample_x.shape[-1]  # F = 5

    model = HybridSTGNN(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        edge_index=edge_index,
        device=device,
        lr=args.lr,
        pos_weight=args.pos_weight,
        accum_steps=args.accum_steps,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=args.save_path,
    )

    # --- Test Evaluation ---
    print("\n" + "=" * 80)
    print("  TEST SET EVALUATION")
    print("=" * 80)

    test_metrics = evaluate_test(
        model, test_loader, edge_index.to(device), device
    )
    print(f"  Test RMSE : {test_metrics['rmse']:.4f}")
    print(f"  Test MAE  : {test_metrics['mae']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
