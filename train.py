"""
=============================================================================
Sezione 6 — TRAIN
=============================================================================
Ciclo di addestramento per FastDepth su NYU Depth V2
con ottimizzatore AdamW e loss MSELoss.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
) -> float:
    """
    Addestra FastDepth per una singola epoca.

    Args:
        model:     FastDepthMDE
        loader:    DataLoader di training (NYU Depth V2)
        optimizer: ottimizzatore AdamW
        criterion: MSELoss
        epoch:     indice dell'epoca corrente (0-based)

    Returns:
        Loss media di training per questa epoca.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (rgb, depth_gt) in enumerate(loader):
        rgb      = rgb.to(DEVICE, non_blocking=True)
        depth_gt = depth_gt.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        depth_pred = model(rgb)

        # Assicura corrispondenza delle dimensioni
        if depth_pred.shape != depth_gt.shape:
            depth_gt = F.interpolate(
                depth_gt, size=depth_pred.shape[2:],
                mode="bilinear", align_corners=False,
            )

        loss = criterion(depth_pred, depth_gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(loader)}]  "
                  f"Loss: {loss.item():.6f}")

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


def train_fastdepth(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> list:
    """
    Loop completo di addestramento per FastDepth su NYU Depth V2.

    Args:
        model:        istanza di FastDepthMDE (spostata su DEVICE internamente)
        train_loader: DataLoader di training
        epochs:       numero di epoche
        lr:           learning rate per AdamW
        weight_decay: regolarizzazione L2

    Returns:
        Lista delle loss medie per-epoca.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    epoch_losses = []
    print(f"\n{'='*60}")
    print(f"  Training FastDepth  |  Device: {DEVICE}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  WD: {weight_decay}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer,
                                   criterion, epoch)
        elapsed = time.time() - t0
        epoch_losses.append(avg_loss)
        print(f"  => Epoch [{epoch+1}/{epochs}]  "
              f"Avg Loss: {avg_loss:.6f}  "
              f"Time: {elapsed:.1f}s\n")

    print("Training complete.\n")
    return epoch_losses
