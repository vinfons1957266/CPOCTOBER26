"""
=============================================================================
Sezione 3 — UTILS
=============================================================================
Funzioni di supporto: seed, metriche MDE, metriche OOD, latenza GPU.
"""

import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from config import DEVICE


# ---------------------------------------------------------------------------
# Riproducibilità
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fissa i seed casuali per NumPy, PyTorch e CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metriche MDE (regressione profondità)
# ---------------------------------------------------------------------------

def compute_mde_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Calcola le metriche standard di Monocular Depth Estimation.

    Args:
        pred:   mappa di profondità predetta  (B, 1, H, W) o (B, H, W)
        target: ground-truth                  (B, 1, H, W) o (B, H, W)

    Returns:
        dict con chiavi: abs_rel, rmse, delta_1, delta_2, delta_3
    """
    pred   = pred.detach().clone().view(-1)
    target = target.detach().clone().view(-1)

    # Filtra valori di profondità non validi (zero / negativi)
    valid  = target > 1e-3
    pred   = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return {"abs_rel": 0.0, "rmse": 0.0,
                "delta_1": 0.0, "delta_2": 0.0, "delta_3": 0.0}

    pred = pred.clamp(min=1e-3)

    # Absolute Relative Error
    abs_rel = (torch.abs(pred - target) / target).mean().item()

    # Root Mean Squared Error
    rmse = torch.sqrt(((pred - target) ** 2).mean()).item()

    # Threshold accuracy: max(pred/target, target/pred) < thr
    ratio   = torch.max(pred / target, target / pred)
    delta_1 = (ratio < 1.25).float().mean().item()
    delta_2 = (ratio < 1.25 ** 2).float().mean().item()
    delta_3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "abs_rel": abs_rel,
        "rmse":    rmse,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }


# ---------------------------------------------------------------------------
# Metriche OOD (AUROC, FPR95)
# ---------------------------------------------------------------------------

def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """
    Calcola AUROC e FPR@95%TPR per OOD detection.

    Convenzione: score più alto => più probabile OOD.
    Label:       ID = 0,  OOD = 1.

    Args:
        id_scores:  array 1-D di score OOD per campioni in-distribution.
        ood_scores: array 1-D di score OOD per campioni out-of-distribution.

    Returns:
        dict con chiavi: auroc, fpr95
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    # FPR al 95 % TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx   = np.searchsorted(tpr, 0.95)
    fpr95 = fpr[min(idx, len(fpr) - 1)]

    return {"auroc": auroc, "fpr95": fpr95}


# ---------------------------------------------------------------------------
# Latenza di inferenza GPU
# ---------------------------------------------------------------------------

def measure_inference_time_gpu(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """
    Misura accurata della latenza di inferenza single-sample (ms).

    Utilizza torch.cuda.synchronize() prima e dopo i timer
    per garantire la correttezza della misurazione su GPU.
    """
    model.eval()
    sample_input = sample_input.to(DEVICE)

    use_cuda = sample_input.is_cuda

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(sample_input)

    if use_cuda:
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) / repeats * 1000.0
    return latency_ms
