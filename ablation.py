"""
=============================================================================
ABLATION STUDY — CORES Layer-wise Analysis
=============================================================================
Analisi comparativa delle risposte convoluzionali attraverso diversi layer
e gruppi di layer (encoder vs decoder, shallow vs deep).

Funzioni principali:
    cores_per_layer_ablation   — AUROC/FPR95 per singolo layer
    cores_group_ablation       — confronto encoder-only / decoder-only / all
    cores_cumulative_ablation  — aggiunta incrementale di layer
    run_ablation_study         — orchestratore che esegue tutto e genera plot
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEVICE, TAU_POS, TAU_NEG, LAMBDA_1, LAMBDA_2,
)
from utils import compute_ood_metrics
from network import FastDepthMDE, ForwardHookHandler, CORESScorer


# ===========================================================================
# Helper: raccolta score con hook su un sotto-insieme di layer
# ===========================================================================

def _collect_scores(
    model: nn.Module,
    loader: DataLoader,
    target_layers: dict,
    scorer: CORESScorer,
) -> np.ndarray:
    """
    Esegue il forward pass con hook solo sui `target_layers` indicati
    e restituisce gli score CORES per-campione.
    """
    handler = ForwardHookHandler()
    handler.register(model, target_layers)

    scores_list = []
    with torch.no_grad():
        for rgb, _ in loader:
            rgb = rgb.to(DEVICE)
            handler.clear()
            _ = model(rgb)
            features = handler.get_features()
            batch_scores = scorer.compute_score_per_sample(
                features, batch_size=rgb.size(0),
            )
            scores_list.append(batch_scores)

    handler.remove()
    return np.concatenate(scores_list)


def _all_target_layers(model: FastDepthMDE) -> OrderedDict:
    """Restituisce tutti gli 11 layer monitorabili (6 enc + 5 dec), ordinati."""
    layers = OrderedDict()
    layers["enc_stage0"] = model.encoder.stage0
    layers["enc_stage1"] = model.encoder.stage1
    layers["enc_stage2"] = model.encoder.stage2
    layers["enc_stage3"] = model.encoder.stage3
    layers["enc_stage4"] = model.encoder.stage4
    layers["enc_stage5"] = model.encoder.stage5
    layers["dec_up1"]    = model.decoder.up1
    layers["dec_up2"]    = model.decoder.up2
    layers["dec_up3"]    = model.decoder.up3
    layers["dec_up4"]    = model.decoder.up4
    layers["dec_up5"]    = model.decoder.up5
    return layers


# ===========================================================================
# 1. Ablation per singolo layer
# ===========================================================================

def cores_per_layer_ablation(
    model: FastDepthMDE,
    id_loader: DataLoader,
    ood_loader: DataLoader,
) -> dict:
    """
    Calcola AUROC e FPR95 usando CORES su ciascun layer *singolarmente*.

    Risponde alla domanda:
        "Quali layer sono più discriminativi per la OOD detection?"

    Returns:
        dict[layer_name] -> {"auroc": float, "fpr95": float}
    """
    model.eval()
    model.to(DEVICE)

    all_layers = _all_target_layers(model)
    scorer = CORESScorer(tau_pos=TAU_POS, tau_neg=TAU_NEG,
                         lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

    results = {}
    print(f"\n{'='*60}")
    print("  ABLATION: Per-Layer CORES Analysis")
    print(f"{'='*60}")
    print(f"  {'Layer':<16s}  {'AUROC':>8s}  {'FPR95':>8s}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}")

    for name, module in all_layers.items():
        single = {name: module}
        id_scores  = _collect_scores(model, id_loader,  single, scorer)
        ood_scores = _collect_scores(model, ood_loader, single, scorer)

        try:
            metrics = compute_ood_metrics(id_scores, ood_scores)
        except ValueError:
            # Distribuzioni degenerate (tutte uguali) → skip
            metrics = {"auroc": 0.5, "fpr95": 1.0}

        results[name] = metrics
        print(f"  {name:<16s}  {metrics['auroc']:>8.4f}  {metrics['fpr95']:>8.4f}")

    print(f"{'='*60}\n")
    return results


# ===========================================================================
# 2. Ablation per gruppo (encoder / decoder / tutti)
# ===========================================================================

def cores_group_ablation(
    model: FastDepthMDE,
    id_loader: DataLoader,
    ood_loader: DataLoader,
) -> dict:
    """
    Confronta le prestazioni CORES usando:
        - solo layer encoder
        - solo layer decoder
        - encoder + decoder combinati

    Risponde alla domanda:
        "L'encoder è più informativo del decoder per la OOD detection?"

    Returns:
        dict[group_name] -> {"auroc": float, "fpr95": float}
    """
    model.eval()
    model.to(DEVICE)

    all_layers = _all_target_layers(model)
    scorer = CORESScorer(tau_pos=TAU_POS, tau_neg=TAU_NEG,
                         lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

    groups = {
        "Encoder only": OrderedDict(
            (k, v) for k, v in all_layers.items() if k.startswith("enc")
        ),
        "Decoder only": OrderedDict(
            (k, v) for k, v in all_layers.items() if k.startswith("dec")
        ),
        "Encoder + Decoder": all_layers,
    }

    results = {}
    print(f"\n{'='*60}")
    print("  ABLATION: Group Comparison (Enc / Dec / All)")
    print(f"{'='*60}")
    print(f"  {'Group':<22s}  {'Layers':>6s}  {'AUROC':>8s}  {'FPR95':>8s}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*8}  {'-'*8}")

    for group_name, layers in groups.items():
        id_scores  = _collect_scores(model, id_loader,  layers, scorer)
        ood_scores = _collect_scores(model, ood_loader, layers, scorer)

        try:
            metrics = compute_ood_metrics(id_scores, ood_scores)
        except ValueError:
            metrics = {"auroc": 0.5, "fpr95": 1.0}

        results[group_name] = metrics
        results[group_name]["num_layers"] = len(layers)
        print(f"  {group_name:<22s}  {len(layers):>6d}  "
              f"{metrics['auroc']:>8.4f}  {metrics['fpr95']:>8.4f}")

    print(f"{'='*60}\n")
    return results


# ===========================================================================
# 3. Ablation cumulativa (impatto della profondità)
# ===========================================================================

def cores_cumulative_ablation(
    model: FastDepthMDE,
    id_loader: DataLoader,
    ood_loader: DataLoader,
) -> dict:
    """
    Aggiunge un layer alla volta (dall'input verso l'output) e misura
    l'AUROC cumulativo.

    Risponde alla domanda:
        "Come cambia la capacità OOD al crescere della profondità del modello?"

    Returns:
        dict con:
            "layer_names":       lista ordinata dei nomi
            "cumulative_auroc":  AUROC dopo aver aggiunto ciascun layer
            "cumulative_fpr95":  FPR95 dopo aver aggiunto ciascun layer
    """
    model.eval()
    model.to(DEVICE)

    all_layers = _all_target_layers(model)
    scorer = CORESScorer(tau_pos=TAU_POS, tau_neg=TAU_NEG,
                         lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

    layer_names = list(all_layers.keys())
    cum_auroc = []
    cum_fpr95 = []

    print(f"\n{'='*60}")
    print("  ABLATION: Cumulative Layer Depth")
    print(f"{'='*60}")
    print(f"  {'Layers used':<40s}  {'AUROC':>8s}  {'FPR95':>8s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}")

    for i in range(1, len(layer_names) + 1):
        subset_names = layer_names[:i]
        subset = OrderedDict((k, all_layers[k]) for k in subset_names)

        id_scores  = _collect_scores(model, id_loader,  subset, scorer)
        ood_scores = _collect_scores(model, ood_loader, subset, scorer)

        try:
            metrics = compute_ood_metrics(id_scores, ood_scores)
        except ValueError:
            metrics = {"auroc": 0.5, "fpr95": 1.0}

        cum_auroc.append(metrics["auroc"])
        cum_fpr95.append(metrics["fpr95"])

        label = f"{subset_names[0]}...{subset_names[-1]}" if i > 1 else subset_names[0]
        print(f"  {label:<40s}  {metrics['auroc']:>8.4f}  {metrics['fpr95']:>8.4f}")

    print(f"{'='*60}\n")

    return {
        "layer_names":      layer_names,
        "cumulative_auroc": cum_auroc,
        "cumulative_fpr95": cum_fpr95,
    }


# ===========================================================================
# 4. Statistiche dettagliate per layer (RM+, RM-, RF+, RF-)
# ===========================================================================

def cores_detailed_statistics(
    model: FastDepthMDE,
    id_loader: DataLoader,
    ood_loader: DataLoader,
) -> dict:
    """
    Per ogni layer, raccoglie le statistiche RM+, RM-, RF+, RF- separatamente
    su dati ID e OOD.

    Risponde alla domanda:
        "Quale componente (magnitudine vs frequenza) è più discriminativa?"

    Returns:
        dict[layer_name] -> {
            "id":  {"rm_pos", "rm_neg", "rf_pos", "rf_neg"},
            "ood": {"rm_pos", "rm_neg", "rf_pos", "rf_neg"},
        }
    """
    model.eval()
    model.to(DEVICE)

    all_layers = _all_target_layers(model)
    handler = ForwardHookHandler()
    handler.register(model, all_layers)

    def _extract_components(loader):
        """Raccoglie RM+/RM-/RF+/RF- per-layer come medie sul dataset."""
        layer_stats = {name: {"rm_pos": [], "rm_neg": [],
                              "rf_pos": [], "rf_neg": []}
                       for name in all_layers}

        with torch.no_grad():
            for rgb, _ in loader:
                rgb = rgb.to(DEVICE)
                handler.clear()
                _ = model(rgb)
                features = handler.get_features()

                for name, feat in features.items():
                    if feat.dim() == 3:
                        feat = feat.unsqueeze(0)

                    flat = feat.view(feat.size(0), -1).float().cpu()
                    n = flat.size(1)

                    pos_mask = flat > TAU_POS
                    neg_mask = flat < TAU_NEG

                    pos_vals = flat * pos_mask.float()
                    rm_p = (pos_vals.sum(dim=1) /
                            (pos_mask.sum(dim=1).float() + 1e-8)).mean().item()

                    neg_vals = flat.abs() * neg_mask.float()
                    rm_n = (neg_vals.sum(dim=1) /
                            (neg_mask.sum(dim=1).float() + 1e-8)).mean().item()

                    rf_p = (pos_mask.float().sum(dim=1) / n).mean().item()
                    rf_n = (neg_mask.float().sum(dim=1) / n).mean().item()

                    layer_stats[name]["rm_pos"].append(rm_p)
                    layer_stats[name]["rm_neg"].append(rm_n)
                    layer_stats[name]["rf_pos"].append(rf_p)
                    layer_stats[name]["rf_neg"].append(rf_n)

        # Media su tutti i batch
        for name in layer_stats:
            for key in layer_stats[name]:
                layer_stats[name][key] = float(np.mean(layer_stats[name][key]))

        return layer_stats

    id_stats  = _extract_components(id_loader)
    ood_stats = _extract_components(ood_loader)

    handler.remove()

    results = {}
    print(f"\n{'='*70}")
    print("  ABLATION: Detailed CORES Components (RM+, RM-, RF+, RF-)")
    print(f"{'='*70}")
    print(f"  {'Layer':<14s} │ {'RM+ (ID)':>9s} {'RM+ (OOD)':>10s} │"
          f" {'RF+ (ID)':>9s} {'RF+ (OOD)':>10s} │"
          f" {'RM- (ID)':>9s} {'RM- (OOD)':>10s}")
    print(f"  {'-'*14} │ {'-'*9} {'-'*10} │"
          f" {'-'*9} {'-'*10} │"
          f" {'-'*9} {'-'*10}")

    for name in all_layers:
        results[name] = {"id": id_stats[name], "ood": ood_stats[name]}

        print(f"  {name:<14s} │"
              f" {id_stats[name]['rm_pos']:>9.4f} {ood_stats[name]['rm_pos']:>10.4f} │"
              f" {id_stats[name]['rf_pos']:>9.4f} {ood_stats[name]['rf_pos']:>10.4f} │"
              f" {id_stats[name]['rm_neg']:>9.4f} {ood_stats[name]['rm_neg']:>10.4f}")

    print(f"{'='*70}\n")
    return results


# ===========================================================================
# Plot dell'ablation study
# ===========================================================================

def _plot_ablation(
    per_layer: dict,
    group: dict,
    cumulative: dict,
    detailed: dict,
    save_dir: str = "./results",
) -> None:
    """Genera 4 plot riassuntivi dell'ablation study."""
    os.makedirs(save_dir, exist_ok=True)

    layer_names = list(per_layer.keys())
    short_names = [n.replace("enc_stage", "E").replace("dec_up", "D")
                   for n in layer_names]

    # ── Plot 1: AUROC per singolo layer (bar chart) ────────────────────────
    aurocs = [per_layer[n]["auroc"] for n in layer_names]
    colors = ["#2196F3" if n.startswith("enc") else "#FF9800"
              for n in layer_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, aurocs, color=colors, edgecolor="white",
                  linewidth=0.8)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("CORES AUROC per Layer (Blue=Encoder, Orange=Decoder)",
                 fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    # Valori sopra le barre
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_per_layer_auroc.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Confronto gruppi (bar chart orizzontale) ───────────────────
    group_names = list(group.keys())
    group_aurocs = [group[g]["auroc"] for g in group_names]
    group_fpr95s = [group[g]["fpr95"] for g in group_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # AUROC
    axes[0].barh(group_names, group_aurocs, color=["#2196F3", "#FF9800", "#4CAF50"])
    axes[0].set_xlabel("AUROC", fontsize=12)
    axes[0].set_title("AUROC by Layer Group", fontsize=13)
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(group_aurocs):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10)
    # FPR95
    axes[1].barh(group_names, group_fpr95s, color=["#2196F3", "#FF9800", "#4CAF50"])
    axes[1].set_xlabel("FPR95", fontsize=12)
    axes[1].set_title("FPR95 by Layer Group", fontsize=13)
    axes[1].set_xlim(0, 1.05)
    for i, v in enumerate(group_fpr95s):
        axes[1].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_group_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: AUROC cumulativo (curva) ───────────────────────────────────
    cum_names = [n.replace("enc_stage", "E").replace("dec_up", "D")
                 for n in cumulative["layer_names"]]
    cum_auroc = cumulative["cumulative_auroc"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cum_auroc) + 1), cum_auroc,
            marker="s", linewidth=2, color="#9C27B0", markersize=8)
    ax.set_xticks(range(1, len(cum_auroc) + 1))
    ax.set_xticklabels(cum_names, rotation=45, ha="right")
    ax.set_xlabel("Layers included (cumulative)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Cumulative AUROC — Impact of Model Depth", fontsize=13)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5)
    # Linea verticale encoder/decoder
    ax.axvline(x=6.5, color="red", linestyle=":", alpha=0.5,
               label="Encoder → Decoder")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_cumulative_auroc.png"), dpi=150)
    plt.close(fig)

    # ── Plot 4: Componenti CORES dettagliate (RM+ e RF+ ID vs OOD) ────────
    rm_pos_id  = [detailed[n]["id"]["rm_pos"]  for n in layer_names]
    rm_pos_ood = [detailed[n]["ood"]["rm_pos"] for n in layer_names]
    rf_pos_id  = [detailed[n]["id"]["rf_pos"]  for n in layer_names]
    rf_pos_ood = [detailed[n]["ood"]["rf_pos"] for n in layer_names]

    x = np.arange(len(short_names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # RM+
    axes[0].bar(x - w/2, rm_pos_id,  w, label="ID (NYU)",  color="#4CAF50", alpha=0.8)
    axes[0].bar(x + w/2, rm_pos_ood, w, label="OOD (KITTI)", color="#F44336", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=45, ha="right")
    axes[0].set_ylabel("RM+", fontsize=12)
    axes[0].set_title("Response Magnitude (RM+) — ID vs OOD", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    # RF+
    axes[1].bar(x - w/2, rf_pos_id,  w, label="ID (NYU)",  color="#4CAF50", alpha=0.8)
    axes[1].bar(x + w/2, rf_pos_ood, w, label="OOD (KITTI)", color="#F44336", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=45, ha="right")
    axes[1].set_ylabel("RF+", fontsize=12)
    axes[1].set_title("Response Frequency (RF+) — ID vs OOD", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ablation_cores_components.png"), dpi=150)
    plt.close(fig)

    print(f"Ablation plots saved to: {save_dir}/")


# ===========================================================================
# Orchestratore ablation study
# ===========================================================================

def run_ablation_study(
    model: FastDepthMDE,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    save_dir: str = "./results",
) -> dict:
    """
    Esegue l'ablation study completa su CORES:

        1. Per-layer analysis      → quale layer discrimina meglio?
        2. Group comparison        → encoder vs decoder vs combinati
        3. Cumulative depth        → impatto della profondità del modello
        4. Detailed components     → RM+/RM-/RF+/RF- per capire *cosa* cambia

    Args:
        model:      FastDepthMDE addestrato
        id_loader:  DataLoader test ID  (NYU)
        ood_loader: DataLoader test OOD (KITTI)
        save_dir:   cartella per i plot

    Returns:
        dict con i risultati di tutte e 4 le analisi
    """
    print(f"\n{'#'*60}")
    print("  CORES ABLATION STUDY")
    print(f"{'#'*60}")

    per_layer  = cores_per_layer_ablation(model, id_loader, ood_loader)
    group      = cores_group_ablation(model, id_loader, ood_loader)
    cumulative = cores_cumulative_ablation(model, id_loader, ood_loader)
    detailed   = cores_detailed_statistics(model, id_loader, ood_loader)

    _plot_ablation(per_layer, group, cumulative, detailed, save_dir)

    print(f"{'#'*60}")
    print("  ABLATION STUDY COMPLETE")
    print(f"{'#'*60}\n")

    return {
        "per_layer":  per_layer,
        "group":      group,
        "cumulative": cumulative,
        "detailed":   detailed,
    }
