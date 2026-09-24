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

    Convenzione CORES (Eq. 1 del paper, Tang et al.): score più alto =>
    più probabile ID, non OOD — i kernel convoluzionali rispondono più
    intensamente a campioni ID (vedi CORESScorer). scikit-learn assume
    invece "score alto => classe positiva"; con label OOD=1 si usa quindi
    -score come "OOD-ness" (score alto => ID => -score basso => corretto).
    Label: ID = 0, OOD = 1 (la classe positiva resta OOD).

    Args:
        id_scores:  array 1-D di score CORES per campioni in-distribution.
        ood_scores: array 1-D di score CORES per campioni out-of-distribution.

    Returns:
        dict con chiavi: auroc, fpr95
    """
    # Convenzione OOD: TPR = frazione di ID classificati come ID (TNR)
    # FPR = frazione di OOD classificati come ID (FPR)
    # Per roc_curve, poniamo classe positiva = 1 = ID
    labels = np.concatenate([
        np.ones(len(id_scores)),      # ID = 1
        np.zeros(len(ood_scores)),    # OOD = 0
    ])
    # Score: alto => ID, basso => OOD.
    # Usiamo direttamente gli score perche' vogliamo che roc_curve
    # misuri TPR per la classe 1 usando soglie decrescenti.
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    # FPR al 95 % TPR (TPR = ID as ID, FPR = OOD as ID)
    fpr, tpr, _ = roc_curve(labels, scores)
    idx   = np.searchsorted(tpr, 0.95)
    fpr95 = fpr[min(idx, len(fpr) - 1)]

    return {"auroc": auroc, "fpr95": fpr95}

def calibrate_threshold(val_id_scores: np.ndarray,
                        tnr_target: float = 0.95) -> float:
    """
    Stima la soglia ID/OOD SOLO su score ID di validazione.
    Convenzione CORES (Eq. 1): score alto => ID. Predizione: OOD se
    score < threshold. Con tnr_target=0.95 si vuole che il 95% dei
    campioni ID di validazione resti SOPRA soglia (classificato ID);
    la soglia è quindi il quantile (1 - tnr_target) — non tnr_target —
    degli score ID di validazione.
    """
    return float(np.quantile(val_id_scores, 1.0 - tnr_target))


def compute_binary_ood_metrics(id_scores: np.ndarray,
                               ood_scores: np.ndarray,
                               threshold: float) -> dict:
    """
    Metriche a soglia fissa. Positivo = OOD.
    Convenzione CORES (Eq. 1): score alto => ID, quindi OOD se score < threshold.
    """
    tp = int((ood_scores <  threshold).sum())   # OOD rilevati
    fn = int((ood_scores >= threshold).sum())
    fp = int((id_scores  <  threshold).sum())   # falsi allarmi su ID
    tn = int((id_scores  >= threshold).sum())

    eps       = 1e-12
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)            # = TPR
    return {
        "threshold": threshold,
        "accuracy":  (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": precision,
        "recall":    recall,
        "f1":        2 * precision * recall / (precision + recall + eps),
        "tnr_test":  tn / (tn + fp + eps),      # sanity check: atteso ≈ 0.95
        "balanced_accuracy": 0.5 * (recall + tn / (tn + fp + eps)),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }
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
