"""
=============================================================================
Sezione 7 — EVALUATION
=============================================================================
Funzioni di valutazione e orchestrazione:
    evaluate_mde_on_id   — accuratezza MDE su test set ID
    evaluate_cores_ood   — OOD detection con CORES (AUROC, FPR95)
    run_full_experiment  — pipeline completa end-to-end
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from config import (
    DEVICE, IMAGE_SIZE, EPOCHS,
    TAU_POS, TAU_NEG, LAMBDA_1, LAMBDA_2,TNR_TARGET,
)
from utils import (
    set_seed, compute_mde_metrics, compute_ood_metrics,
    measure_inference_time_gpu, calibrate_threshold, compute_binary_ood_metrics,
)
from data import get_dataloaders
from network import FastDepthMDE, ForwardHookHandler, CORESScorer
from train import train_fastdepth
from ablation import run_ablation_study


# ===========================================================================
# Valutazione MDE su ID
# ===========================================================================

def evaluate_mde_on_id(
    model: torch.nn.Module,
    id_test_loader: DataLoader,
) -> dict:
    """
    Valuta l'accuratezza MDE sul test set in-distribution (NYU).

    Returns:
        Dict aggregato: abs_rel, rmse, delta_1, delta_2, delta_3
    """
    model.eval()
    model.to(DEVICE)

    all_metrics = {"abs_rel": [], "rmse": [],
                   "delta_1": [], "delta_2": [], "delta_3": []}

    with torch.no_grad():
        for rgb, depth_gt in id_test_loader:
            rgb      = rgb.to(DEVICE)
            depth_gt = depth_gt.to(DEVICE)

            depth_pred = model(rgb)

            if depth_pred.shape != depth_gt.shape:
                depth_gt = F.interpolate(
                    depth_gt, size=depth_pred.shape[2:],
                    mode="bilinear", align_corners=False,
                )

            m = compute_mde_metrics(depth_pred, depth_gt)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    agg = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    print(f"\n{'='*60}")
    print("  MDE Evaluation on ID (NYU Depth V2 Test)")
    print(f"{'='*60}")
    print(f"  AbsRel : {agg['abs_rel']:.4f}")
    print(f"  RMSE   : {agg['rmse']:.4f}")
    print(f"  δ₁     : {agg['delta_1']:.4f}")
    print(f"  δ₂     : {agg['delta_2']:.4f}")
    print(f"  δ₃     : {agg['delta_3']:.4f}")
    print(f"{'='*60}\n")

    return agg


# ===========================================================================
# Selezione dei layer target per CORES
# ===========================================================================

def _get_target_layers(model: FastDepthMDE) -> dict:
    """
    Seleziona i layer da encoder e decoder per il monitoraggio CORES.
    Restituisce dict che mappa nomi descrittivi a riferimenti nn.Module.
    """
    targets = {}

    # Encoder layers
    targets["enc_stage0"] = model.encoder.stage0
    targets["enc_stage1"] = model.encoder.stage1
    targets["enc_stage2"] = model.encoder.stage2
    targets["enc_stage3"] = model.encoder.stage3
    targets["enc_stage4"] = model.encoder.stage4
    targets["enc_stage5"] = model.encoder.stage5

    # Decoder layers
    targets["dec_up1"] = model.decoder.up1
    targets["dec_up2"] = model.decoder.up2
    targets["dec_up3"] = model.decoder.up3
    targets["dec_up4"] = model.decoder.up4
    targets["dec_up5"] = model.decoder.up5

    return targets


# ===========================================================================
# Valutazione OOD con CORES
# ===========================================================================
def _collect_cores_scores(model, loader, scorer, hook_handler) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for rgb, _ in loader:
            rgb = rgb.to(DEVICE)
            hook_handler.clear()
            _ = model(rgb)
            scores.append(scorer.compute_score_per_sample(
                hook_handler.get_features(), batch_size=rgb.size(0)))
    return np.concatenate(scores)

def evaluate_cores_ood(model, id_val_loader, id_test_loader, ood_test_loader) -> dict:
    model.eval().to(DEVICE)
    hook_handler = ForwardHookHandler()
    hook_handler.register(model, _get_target_layers(model))
    scorer = CORESScorer(tau_pos=TAU_POS, tau_neg=TAU_NEG,
                         lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

    val_scores = _collect_cores_scores(model, id_val_loader,  scorer, hook_handler)
    id_scores  = _collect_cores_scores(model, id_test_loader, scorer, hook_handler)
    ood_scores = _collect_cores_scores(model, ood_test_loader, scorer, hook_handler)
    hook_handler.remove()

    ood_metrics = compute_ood_metrics(id_scores, ood_scores)      # threshold-free
    threshold   = calibrate_threshold(val_scores, TNR_TARGET)     # solo su ID val
    binary      = compute_binary_ood_metrics(id_scores, ood_scores, threshold)

    print(f"  Threshold (q{TNR_TARGET:.2f} su ID-val, n={len(val_scores)}): {threshold:.4f}")
    print(f"  Accuracy {binary['accuracy']:.4f} | Precision {binary['precision']:.4f} | "
          f"Recall {binary['recall']:.4f} | F1 {binary['f1']:.4f}")
    print(f"  TNR su ID-test: {binary['tnr_test']:.4f}  (atteso ≈ {TNR_TARGET:.2f})")

    return {**ood_metrics, "id_scores": id_scores, "ood_scores": ood_scores,
            "val_scores": val_scores, "threshold": threshold, "binary": binary}


# ===========================================================================
# Generazione plot
# ===========================================================================

def _plot_results(epoch_losses: list, ood_result: dict,
                  save_dir: str = "./results") -> None:
    """Genera e salva curva di loss, istogrammi score OOD e curva ROC."""
    os.makedirs(save_dir, exist_ok=True)

    # --- Training loss ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axvline(ood_result["threshold"], color="black", linestyle="--", linewidth=2,
               label=f"Soglia (TNR95 val) = {ood_result['threshold']:.3f}")
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            marker="o", linewidth=2, color="#2196F3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("FastDepth Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
    plt.close(fig)

    # --- Distribuzioni score OOD ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(ood_result["id_scores"], bins=50, alpha=0.6,
            label="ID (NYU)", color="#4CAF50", density=True)
    ax.hist(ood_result["ood_scores"], bins=50, alpha=0.6,
            label="OOD (KITTI)", color="#F44336", density=True)
    ax.set_xlabel("CORES Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"OOD Score Distributions  |  AUROC={ood_result['auroc']:.3f}  "
        f"FPR95={ood_result['fpr95']:.3f}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ood_scores.png"), dpi=150)
    plt.close(fig)

    # --- Curva ROC ---
    labels = np.concatenate([
        np.zeros(len(ood_result["id_scores"])),
        np.ones(len(ood_result["ood_scores"])),
    ])
    scores = np.concatenate([
        ood_result["id_scores"],
        ood_result["ood_scores"],
    ])
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, color="#9C27B0",
            label=f"AUROC = {ood_result['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_title("ROC Curve — CORES OOD Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to: {save_dir}/")


# ===========================================================================
# Orchestrazione completa
# ===========================================================================

def run_full_experiment() -> None:
    """
    Orchestra l'esperimento completo:

        1. Fissa seed per riproducibilità
        2. Costruisce i data loader  (NYU ID + KITTI OOD)
        3. Istanzia il modello FastDepth
        4. Addestra su NYU Depth V2  (ID, indoor)
        5. Valuta accuratezza MDE    sul test set ID
        6. Misura latenza di inferenza GPU
        7. Valuta OOD detection      (CORES) su ID vs OOD
        8. Ablation study CORES      (per-layer, gruppi, cumulativa)
        9. Genera e salva i grafici
    """
    set_seed(42)

    # 1. Dati
    print("\n[1/7] Building data loaders ...")
    train_loader, id_val_loader, id_test_loader, ood_test_loader = get_dataloaders()

    # 2. Modello
    print("[2/7] Instantiating FastDepth ...")
    model = FastDepthMDE(pretrained_encoder=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Total parameters: {total_params:,}")

    # 3. Addestramento
    print("[3/7] Training on NYU Depth V2 (ID) ...")
    epoch_losses = train_fastdepth(model, train_loader)

    # 4. Valutazione MDE
    print("[4/7] Evaluating MDE on ID test set ...")
    mde_metrics = evaluate_mde_on_id(model, id_test_loader)

    # 5. Latenza di inferenza
    print("[5/7] Measuring GPU inference latency ...")
    sample_input = torch.randn(1, 3, *IMAGE_SIZE).to(DEVICE)
    latency = measure_inference_time_gpu(model, sample_input)
    print(f"       Inference latency: {latency:.2f} ms  "
          f"({1000.0/latency:.1f} FPS)")

    # 6. OOD Detection
    print("[6/7] Evaluating CORES OOD detection ...")
    ood_result = evaluate_cores_ood(model, id_val_loader, id_test_loader, ood_test_loader)

    # 7. Ablation study CORES
    print("[7/7] Running CORES ablation study ...")
    ablation_results = run_ablation_study(
        model, id_test_loader, ood_test_loader,
    )

    # 8. Grafici
    _plot_results(epoch_losses, ood_result )
    
    # --- Riepilogo ---
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Device         : {DEVICE}")
    print(f"  Parameters     : {total_params:,}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Final Train Loss: {epoch_losses[-1]:.6f}")
    print(f"  MDE AbsRel     : {mde_metrics['abs_rel']:.4f}")
    print(f"  MDE RMSE       : {mde_metrics['rmse']:.4f}")
    print(f"  MDE δ₁         : {mde_metrics['delta_1']:.4f}")
    print(f"  Latency        : {latency:.2f} ms")
    print(f"  AUROC          : {ood_result['auroc']:.4f}")
    print(f"  FPR95          : {ood_result['fpr95']:.4f}")
    print(f"{'='*60}\n")
    
