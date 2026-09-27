"""
=============================================================================
Sezione 6 — TRAIN
=============================================================================
Ciclo di addestramento per FastDepth su NYU Depth V2
con ottimizzatore AdamW, BerHu loss e scheduler OneCycleLR.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY


# ===========================================================================
# BerHu Loss — loss standard per Monocular Depth Estimation
# ===========================================================================

class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss per depth estimation.

    A differenza della MSE (che penalizza i grandi errori quadraticamente
    e i piccoli linearmente... sbagliando verso l'alto), la BerHu loss:
      - Usa L1 per errori piccoli (|e| ≤ c) → più robusta agli outlier
      - Usa L2 per errori grandi (|e| > c) → penalizza le predizioni molto sbagliate

    La soglia c è calcolata come una frazione del massimo errore assoluto
    nel batch, rendendola adattiva.

    Riferimento: Laina et al., "Deeper Depth Prediction with Fully
    Convolutional Residual Networks", 3DV 2016.
    """

    def __init__(self, threshold_ratio: float = 0.2):
        super().__init__()
        self.threshold_ratio = threshold_ratio

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        c = self.threshold_ratio * diff.max().detach()

        mask = diff <= c
        # L1 per errori piccoli, L2 smoothed per errori grandi
        loss = torch.where(mask, diff, (diff ** 2 + c ** 2) / (2.0 * c + 1e-8))
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    scheduler=None,
) -> float:
    """
    Addestra FastDepth per una singola epoca.

    Args:
        model:     FastDepthMDE
        loader:    DataLoader di training (NYU Depth V2)
        optimizer: ottimizzatore AdamW
        criterion: BerHuLoss (o MSELoss per compatibilità)
        epoch:     indice dell'epoca corrente (0-based)
        scheduler: lr scheduler (OneCycleLR); step() chiamato per batch

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

        # Gradient clipping per stabilità numerica
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # OneCycleLR step per-batch (non per-epoch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            lr_str = ""
            if scheduler is not None:
                lr_str = f"  LR: {scheduler.get_last_lr()[0]:.6f}"
            print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(loader)}]  "
                  f"Loss: {loss.item():.6f}{lr_str}")

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

    Usa BerHu loss (standard per depth estimation) e scheduler OneCycleLR
    (warmup + cosine annealing) per una convergenza più rapida e stabile.

    Args:
        model:        istanza di FastDepthMDE (spostata su DEVICE internamente)
        train_loader: DataLoader di training
        epochs:       numero di epoche
        lr:           learning rate massimo per OneCycleLR
        weight_decay: regolarizzazione L2

    Returns:
        Lista delle loss medie per-epoca.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = BerHuLoss(threshold_ratio=0.2)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    epoch_losses = []
    print(f"\n{'='*60}")
    print(f"  Training FastDepth  |  Device: {DEVICE}")
    print(f"  Epochs: {epochs}  |  Max LR: {lr}  |  WD: {weight_decay}")
    print(f"  Loss: BerHu  |  Scheduler: OneCycleLR")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer,
                                   criterion, epoch, scheduler=scheduler)
        elapsed = time.time() - t0
        epoch_losses.append(avg_loss)
        print(f"  => Epoch [{epoch+1}/{epochs}]  "
              f"Avg Loss: {avg_loss:.6f}  "
              f"Time: {elapsed:.1f}s\n")

    print("Training complete.\n")
    return epoch_losses
