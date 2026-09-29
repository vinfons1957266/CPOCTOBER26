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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from config import (
    DEVICE, IMAGE_SIZE, EPOCHS, BATCH_SIZE,
    TAU_POS, TAU_NEG, LAMBDA_1, LAMBDA_2, TNR_TARGET, TOPK_FRAC,
)
from utils import (
    set_seed, compute_mde_metrics, compute_ood_metrics,
    measure_inference_time_gpu, calibrate_threshold, compute_binary_ood_metrics,
)
from data import get_dataloaders
from network import FastDepthMDE, ForwardHookHandler, CORESScorer, CORESKernelSelector
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
    print(f"  delta_1: {agg['delta_1']:.4f}")
    print(f"  delta_2: {agg['delta_2']:.4f}")
    print(f"  delta_3: {agg['delta_3']:.4f}")
    print(f"{'='*60}\n")

    return agg


# ===========================================================================
# Selezione dei layer target per CORES
# ===========================================================================

def _get_target_layers(model: torch.nn.Module) -> dict:
    """
    Seleziona i layer da encoder e decoder per il monitoraggio CORES.
    Restituisce dict che mappa nomi descrittivi a riferimenti nn.Module.
    Supporta sia FastDepthMDE che METERMDE tramite get_cores_target_layers().
    """
    if hasattr(model, "get_cores_target_layers"):
        return model.get_cores_target_layers()

    targets = {}
    targets["enc_stage0"] = model.encoder.proj_s0[1]
    targets["enc_stage1"] = model.encoder.proj_s1[1]
    targets["enc_stage2"] = model.encoder.proj_s2[1]
    targets["enc_stage3"] = model.encoder.proj_s3[1]
    targets["enc_stage4"] = model.encoder.proj_bottleneck[1]
    targets["dec_up1"] = model.decoder.up1.conv.bn_pw
    targets["dec_up2"] = model.decoder.up2.conv.bn_pw
    targets["dec_up3"] = model.decoder.up3.conv.bn_pw
    targets["dec_up4"] = model.decoder.up4.conv.bn_pw
    targets["dec_up5"] = model.decoder.up5.conv.bn_pw
    return targets


def _get_target_layers_with_final_conv(model: torch.nn.Module) -> dict:
    """
    Come _get_target_layers, ma con un tap aggiuntivo su final_conv grezzo.
    Supporta sia FastDepthMDE che METERMDE.
    """
    if hasattr(model, "get_cores_target_layers_with_final_conv"):
        return model.get_cores_target_layers_with_final_conv()

    targets = _get_target_layers(model)
    targets["final_conv_raw"] = model.decoder.final_conv
    return targets


def _get_kernel_selector(model: torch.nn.Module):
    """Restituisce il selector CORES adatto per FastDepthMDE o METERMDE."""
    if hasattr(model, "__class__") and model.__class__.__name__ == "METERMDE":
        from meter import METERKernelSelector
        return METERKernelSelector(model)
    return CORESKernelSelector(model)


# ===========================================================================
# Valutazione OOD con CORES
# ===========================================================================
def _collect_cores_scores(model, loader, scorer, hook_handler,
                          exclude_keys: frozenset = frozenset()) -> np.ndarray:
    """
    exclude_keys: chiavi da NON passare a compute_score_per_sample. Serve
    quando hook_handler e' registrato con _get_target_layers_with_final_conv
    (percorso selezionato E non-selezionato condividono lo stesso hook-set
    per una sola forward pass per batch): "final_conv_raw" non e' un vero
    layer monitorato CORES, e senza questo filtro verrebbe incluso come un
    12-esimo layer nella media, corrompendo lo score non-selezionato.
    """
    scores = []
    with torch.no_grad():
        for rgb, _ in loader:
            rgb = rgb.to(DEVICE)
            hook_handler.clear()
            _ = model(rgb)
            feats = hook_handler.get_features()
            if exclude_keys:
                feats = {k: v for k, v in feats.items() if k not in exclude_keys}
            scores.append(scorer.compute_score_per_sample(
                feats, batch_size=rgb.size(0)))
    return np.concatenate(scores)


def _collect_cores_scores_selected(model, loader, scorer, selector,
                                   hook_handler, topk_frac: float) -> np.ndarray:
    """
    Come _collect_cores_scores, ma per il percorso CON selezione dei kernel
    sample-relevant (Eq. 9, Sez. 4.2): per ogni batch, dopo la forward pass,
    esegue la selezione iniziale (B2, su dec_up5 via final_conv_raw) e il
    backtracking (B3, lungo la catena a 5 nodi), poi calcola lo score
    selezionato (B4) invece di quello su tutti i canali.

    hook_handler deve essere registrato con
    _get_target_layers_with_final_conv (non _get_target_layers): serve
    "final_conv_raw" oltre ai 5 tap decoder.
    """
    scores = []
    with torch.no_grad():
        for rgb, _ in loader:
            rgb = rgb.to(DEVICE)
            hook_handler.clear()
            _ = model(rgb)
            feats = dict(hook_handler.get_features())
            raw_depth = feats.pop("final_conv_raw")

            k0 = max(1, round(topk_frac * feats["dec_up5"].shape[1]))
            i_pos0, i_neg0 = selector.select_initial_indices(raw_depth, feats["dec_up5"], k=k0)
            selections = selector.backtrack(i_pos0, i_neg0, topk_frac)

            scores.append(scorer.compute_score_per_sample_selected(
                feats, selections, batch_size=rgb.size(0)))
    return np.concatenate(scores)

def evaluate_cores_ood(model, id_val_loader, id_test_loader, ood_test_loader,
                       tau_pos: float = TAU_POS, tau_neg: float = TAU_NEG) -> dict:
    """
    Calcola DUE score CORES, sugli stessi id_val/id_test/ood_test e con lo
    stesso tau_pos/tau_neg, in un'unica forward pass per batch:

      - non-selezionato (Eq. 5, tutti gli 11 layer) — chiavi al livello
        superiore del dict restituito (id_scores, threshold, binary, ...),
        INVARIATO rispetto a prima di Parte B: preserva la compatibilità
        con _plot_results e con la calibrazione già fatta da
        calibrate_cores_taus (che ha verificato QUESTA formulazione).
      - selezionato/backtracked (Eq. 9, Sez. 4.2, 5 layer decoder via
        CORESKernelSelector) — sotto la chiave "selected", stessa struttura.

    Le due versioni si tengono ENTRAMBE (non si sostituisce l'una con
    l'altra) per tre motivi: (1) e' la stessa ablation "w/ vs w/o kernel
    selection" che il paper riporta come risultato centrale (Fig. 4);
    (2) permette di vedere se la selezione dei kernel mitiga anomalie già
    osservate sulla formulazione non-selezionata (es. inversione di AUROC);
    (3) calibrate_cores_taus ha calibrato tau_pos/tau_neg SOLO sulla
    formulazione non-selezionata — riusarli qui per lo score selezionato
    è un'approssimazione dichiarata, non ancora una ricalibrazione dedicata.

    tau_pos/tau_neg di default usano i valori grezzi di config.py, ma
    run_full_experiment() passa esplicitamente quelli calibrati da
    calibrate_cores_taus() (Sez. 5.1 del paper) — vedi lì per il perché.
    """
    model.eval().to(DEVICE)
    hook_handler = ForwardHookHandler()
    hook_handler.register(model, _get_target_layers_with_final_conv(model))
    scorer = CORESScorer(tau_pos=tau_pos, tau_neg=tau_neg,
                         lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)
    selector = _get_kernel_selector(model)   # costruito UNA volta, riusato sui 3 split

    # --- Non-selezionato (Eq. 5, 11 layer) ---
    val_scores = _collect_cores_scores(model, id_val_loader,  scorer, hook_handler,
                                       exclude_keys={"final_conv_raw"})
    id_scores  = _collect_cores_scores(model, id_test_loader, scorer, hook_handler,
                                       exclude_keys={"final_conv_raw"})
    ood_scores = _collect_cores_scores(model, ood_test_loader, scorer, hook_handler,
                                       exclude_keys={"final_conv_raw"})

    ood_metrics = compute_ood_metrics(id_scores, ood_scores)      # threshold-free
    threshold   = calibrate_threshold(val_scores, TNR_TARGET)     # solo su ID val
    binary      = compute_binary_ood_metrics(id_scores, ood_scores, threshold)

    # --- Selezionato/backtracked (Eq. 9, 5 layer decoder) ---
    val_scores_sel = _collect_cores_scores_selected(
        model, id_val_loader,  scorer, selector, hook_handler, TOPK_FRAC)
    id_scores_sel  = _collect_cores_scores_selected(
        model, id_test_loader, scorer, selector, hook_handler, TOPK_FRAC)
    ood_scores_sel = _collect_cores_scores_selected(
        model, ood_test_loader, scorer, selector, hook_handler, TOPK_FRAC)

    ood_metrics_sel = compute_ood_metrics(id_scores_sel, ood_scores_sel)
    threshold_sel   = calibrate_threshold(val_scores_sel, TNR_TARGET)
    binary_sel      = compute_binary_ood_metrics(id_scores_sel, ood_scores_sel, threshold_sel)

    hook_handler.remove()

    print("  --- Non-selezionato (Eq. 5, 11 layer) ---")
    print(f"  Threshold (TNR target {TNR_TARGET:.2f} su ID-val, n={len(val_scores)}): {threshold:.4f}")
    print(f"  AUROC {ood_metrics['auroc']:.4f} | FPR95 {ood_metrics['fpr95']:.4f}")
    print(f"  Accuracy {binary['accuracy']:.4f} | Precision {binary['precision']:.4f} | "
          f"Recall {binary['recall']:.4f} | F1 {binary['f1']:.4f}")
    print(f"  TNR su ID-test: {binary['tnr_test']:.4f}  (atteso ~ {TNR_TARGET:.2f})")

    print("  --- Selezionato/backtracked (Eq. 9, 5 layer decoder, TOPK_FRAC="
          f"{TOPK_FRAC:.2f}) ---")
    print(f"  Threshold (TNR target {TNR_TARGET:.2f} su ID-val, n={len(val_scores_sel)}): "
          f"{threshold_sel:.4f}")
    print(f"  AUROC {ood_metrics_sel['auroc']:.4f} | FPR95 {ood_metrics_sel['fpr95']:.4f}")
    print(f"  Accuracy {binary_sel['accuracy']:.4f} | Precision {binary_sel['precision']:.4f} | "
          f"Recall {binary_sel['recall']:.4f} | F1 {binary_sel['f1']:.4f}")
    print(f"  TNR su ID-test: {binary_sel['tnr_test']:.4f}  (atteso ~ {TNR_TARGET:.2f})")

    return {
        **ood_metrics, "id_scores": id_scores, "ood_scores": ood_scores,
        "val_scores": val_scores, "threshold": threshold, "binary": binary,
        "selected": {
            **ood_metrics_sel, "id_scores": id_scores_sel, "ood_scores": ood_scores_sel,
            "val_scores": val_scores_sel, "threshold": threshold_sel, "binary": binary_sel,
        },
    }


# ===========================================================================
# Calibrazione TAU_POS/TAU_NEG (Sez. 5.1 del paper CORES)
# ===========================================================================

def _generate_noise_batch(n_samples: int, image_size: tuple, kind: str,
                          img_transform: transforms.Normalize) -> torch.Tensor:
    """
    Genera un batch di rumore (gaussiano o uniforme) nello spazio pixel
    [0,1] delle immagini reali (prima della normalizzazione), poi applica
    la STESSA normalizzazione ImageNet usata per i dati reali (vedi
    data.py::get_dataloaders) — cosi' il rumore entra nella rete sulla
    stessa base statistica delle immagini vere, non su una scala arbitraria.
    """
    h, w = image_size
    if kind == "gaussian":
        pixels = (torch.randn(n_samples, 3, h, w) * 0.25 + 0.5).clamp(0.0, 1.0)
    elif kind == "uniform":
        pixels = torch.rand(n_samples, 3, h, w)
    else:
        raise ValueError(f"kind sconosciuto: {kind!r} (atteso 'gaussian' o 'uniform')")

    return img_transform(pixels)


def calibrate_cores_taus(
    model: FastDepthMDE,
    id_val_loader: DataLoader,
    n_noise_samples: int = 128,
    n_candidates: int = 6,
    tnr_target: float = TNR_TARGET,
) -> tuple:
    """
    Calibra TAU_POS/TAU_NEG (Sez. 5.1 del paper CORES: "we tune the
    thresholds tau_+ and tau_- by calibrating them to minimize the false
    positive rate of Gaussian noise and uniform noise").

    Disciplina train/val/test: usa SOLO id_val, mai id_test/ood_test —
    stessa disciplina di utils.calibrate_threshold, applicata qui a monte
    (i tau della risposta convoluzionale), non alla soglia decisionale
    finale (calcolata separatamente, dopo, sugli score prodotti con questi
    tau — vedi evaluate_cores_ood).

    Metodo: ricerca SIMMETRICA (tau_neg = -tau_pos) su una griglia di
    candidati derivati dai percentili della distribuzione |picco/valle
    per canale| osservata su id_val — la stessa quantita' che
    cores_response_components confronta con tau (Eq. 3/4), non i valori
    grezzi per-pixel. La ricerca e' adattiva ai dati (non un intervallo
    fisso hard-coded) perche' la scala delle attivazioni varia
    enormemente per layer e per stato di training (si va da ~1 a ~90 nei
    test di questo progetto). Il rumore Gaussiano e uniforme sostituisce
    dati OOD reali per la calibrazione, cosi' da non toccare mai
    id_test/ood_test durante questa fase.

    Per ciascun candidato tau:
      1. calcola gli score CORES (Eq. 5, non selezionati, sugli 11 layer
         di _get_target_layers) su id_val e su rumore Gaussiano/uniforme
      2. calibra una soglia da id_val (utils.calibrate_threshold, stesso
         tnr_target usato nel resto della pipeline)
      3. misura il FPR del rumore rispetto a quella soglia — frazione di
         rumore con score >= soglia, cioe' classificato (erroneamente)
         come ID (convenzione Eq. 1: score alto => ID)

    LIMITE NOTO (osservato con EPOCHS=2, modello smoke-test): la ricerca
    ha prodotto FPR=1.0 per OGNI candidato provato — il rumore riceve
    SEMPRE uno score piu' alto della soglia calibrata su id_val, quindi
    nessun tau in questo intervallo separa rumore da ID. Due ipotesi
    aperte, non ancora distinte:
      (a) sotto-addestramento: a 2 epoche l'encoder e' in gran parte
          random-init, quindi la premessa di CORES ("i kernel rispondono
          di piu' a pattern familiari") potrebbe non valere ancora;
      (b) il rumore i.i.d. per pixel, privo di correlazione spaziale,
          potrebbe produrre risposte convoluzionali sistematicamente piu'
          grandi delle immagini reali per motivi statistici indipendenti
          dalla qualita' del training.
    La griglia di candidati e' derivata SOLO dai percentili di id_val: se
    (b) fosse vera, un tau efficace potrebbe trovarsi fuori da questo
    intervallo (es. oltre il massimo di id_val) e la ricerca attuale non
    lo troverebbe mai. Prima di allargare la griglia o cambiare la
    generazione del rumore, ripetere questa calibrazione con EPOCHS
    ripristinato a un valore di training reale (non lo smoke-test 2) —
    se la saturazione persiste, l'ipotesi (b) diventa piu' plausibile e
    la ricerca va rivista; se si risolve da sola, nessuna modifica serve.
    Sceglie il candidato che minimizza la media tra FPR gaussiano e FPR
    uniforme.

    Args:
        model:          FastDepthMDE gia' addestrato, verra' messo in eval().
        id_val_loader:  DataLoader di calibrazione (MAI id_test/ood_test).
        n_noise_samples: campioni di rumore generati per tipo (gaussiano
                         E uniforme, quindi 2x questo numero in totale).
        n_candidates:   numero di candidati tau nella griglia.
        tnr_target:     stesso TNR_TARGET usato da calibrate_threshold.

    Returns:
        (tau_pos, tau_neg, diagnostics): i tau scelti (tau_neg = -tau_pos)
        e un dict {"candidates": [...], "chosen": {...}} coi dettagli di
        ogni candidato provato (utile per debug/plot).
    """
    model.eval().to(DEVICE)
    target_layers = _get_target_layers(model)
    hook_handler = ForwardHookHandler()
    hook_handler.register(model, target_layers)

    img_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )

    # --- 1. Raccoglie picchi e valli PER OGNI SINGOLO LAYER su id_val ---
    layer_peaks = {name: [] for name in target_layers}
    layer_troughs = {name: [] for name in target_layers}
    with torch.no_grad():
        for rgb, _ in id_val_loader:
            rgb = rgb.to(DEVICE)
            hook_handler.clear()
            _ = model(rgb)
            feats = hook_handler.get_features()
            for name in target_layers:
                feat = feats[name]
                layer_peaks[name].append(feat.amax(dim=(2, 3)).flatten().cpu())
                layer_troughs[name].append(feat.amin(dim=(2, 3)).flatten().cpu())

    layer_peaks_cat = {name: torch.cat(layer_peaks[name]) for name in target_layers}
    layer_troughs_cat = {name: torch.cat(layer_troughs[name]) for name in target_layers}

    # Griglia di percentili per la ricerca (da 50% a 90%)
    percentiles = torch.linspace(50.0, 90.0, n_candidates)

    # --- 2. Rumore generato UNA sola volta ---
    gauss_loader = DataLoader(
        TensorDataset(_generate_noise_batch(n_noise_samples, IMAGE_SIZE, "gaussian", img_transform),
                     torch.zeros(n_noise_samples)),
        batch_size=BATCH_SIZE,
    )
    unif_loader = DataLoader(
        TensorDataset(_generate_noise_batch(n_noise_samples, IMAGE_SIZE, "uniform", img_transform),
                     torch.zeros(n_noise_samples)),
        batch_size=BATCH_SIZE,
    )

    # --- 3. Ricerca del miglior percentile per-layer sulla griglia ---
    diagnostics = []
    best = None
    for p in percentiles:
        p_val = float(p)
        # Per ogni layer: tau_pos al percentile p_val dei picchi,
        # tau_neg al percentile (100 - p_val) delle valli
        cand_tau_pos = {}
        cand_tau_neg = {}
        for name in target_layers:
            tp = float(torch.quantile(layer_peaks_cat[name], p_val / 100.0))
            tn = float(torch.quantile(layer_troughs_cat[name], (100.0 - p_val) / 100.0))
            cand_tau_pos[name] = max(tp, 1e-4)
            cand_tau_neg[name] = min(tn, -1e-4)

        scorer = CORESScorer(tau_pos=cand_tau_pos, tau_neg=cand_tau_neg,
                             lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

        val_scores   = _collect_cores_scores(model, id_val_loader, scorer, hook_handler)
        gauss_scores = _collect_cores_scores(model, gauss_loader,  scorer, hook_handler)
        unif_scores  = _collect_cores_scores(model, unif_loader,   scorer, hook_handler)

        threshold = calibrate_threshold(val_scores, tnr_target)
        fpr_gauss = float((gauss_scores >= threshold).mean())
        fpr_unif  = float((unif_scores  >= threshold).mean())
        avg_fpr   = 0.5 * (fpr_gauss + fpr_unif)

        entry = {
            "percentile": p_val,
            "tau_pos": cand_tau_pos,
            "tau_neg": cand_tau_neg,
            "threshold": threshold,
            "fpr_gaussian": fpr_gauss,
            "fpr_uniform": fpr_unif,
            "avg_fpr": avg_fpr,
        }
        diagnostics.append(entry)
        if best is None or avg_fpr < best["avg_fpr"]:
            best = entry

    hook_handler.remove()

    print(f"\n{'='*60}")
    print("  Calibrazione PER-LAYER TAU_POS/TAU_NEG (rumore Gaussiano/uniforme)")
    print(f"{'='*60}")
    print(f"  {'Percentile':>10s} {'FPR gauss':>10s} {'FPR unif':>10s} {'avg FPR':>10s}")
    for d in diagnostics:
        marker = "  <-- scelto" if d is best else ""
        print(f"  {d['percentile']:>9.1f}% {d['fpr_gaussian']:>10.4f} "
              f"{d['fpr_uniform']:>10.4f} {d['avg_fpr']:>10.4f}{marker}")

    print("\n  Valori tau per-layer scelti:")
    for name in target_layers:
        print(f"    {name:<16s}: tau_pos = {best['tau_pos'][name]:>8.4f}, "
              f"tau_neg = {best['tau_neg'][name]:>8.4f}")
    print(f"{'='*60}\n")

    return best["tau_pos"], best["tau_neg"], {"candidates": diagnostics, "chosen": best}


# ===========================================================================
# Generazione plot
# ===========================================================================

def _plot_results(epoch_losses: list, ood_result: dict,
                  save_dir: str = "./results",
                  model_name: str = "FastDepth") -> None:
    """Genera e salva curva di loss, istogrammi score OOD e curva ROC."""
    os.makedirs(save_dir, exist_ok=True)
    p = f"{model_name.lower()}_"

    # --- Training loss ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            marker="o", linewidth=2, color="#2196F3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"[{model_name}] Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{p}training_loss.png"), dpi=150)
    if model_name.lower() == "fastdepth":
        fig.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
    plt.close(fig)

    # --- Distribuzioni score OOD ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axvline(ood_result["threshold"], color="black", linestyle="--", linewidth=2,
               label=f"Soglia (TNR95 val) = {ood_result['threshold']:.3f}")
    ax.hist(ood_result["id_scores"], bins=50, alpha=0.6,
            label="ID (NYU)", color="#4CAF50", density=True)
    ax.hist(ood_result["ood_scores"], bins=50, alpha=0.6,
            label="OOD (KITTI)", color="#F44336", density=True)
    ax.set_xlabel("CORES Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"[{model_name}] OOD Score Distributions  |  AUROC={ood_result['auroc']:.3f}  "
        f"FPR95={ood_result['fpr95']:.3f}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{p}ood_scores.png"), dpi=150)
    if model_name.lower() == "fastdepth":
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
    ax.set_title(f"[{model_name}] ROC Curve — CORES OOD Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{p}roc_curve.png"), dpi=150)
    if model_name.lower() == "fastdepth":
        fig.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to: {save_dir}/ ({p}*)")


# ===========================================================================
# Orchestrazione completa
# ===========================================================================

def run_full_experiment(model_type: str = "meter", epochs: int = EPOCHS,
                        skip_ablation: bool = False) -> None:
    """
    Orchestra l'esperimento completo per il modello specificato:

        1. Fissa seed per riproducibilità
        2. Costruisce i data loader  (NYU ID + KITTI OOD)
        3. Istanzia il modello scelto (METERMDE o FastDepthMDE)
        4. Addestra su NYU Depth V2  (ID, indoor)
        5. Valuta accuratezza MDE    sul test set ID
        6. Misura latenza di inferenza GPU
        7. Calibra TAU_POS/TAU_NEG   su id_val
        8. Valuta OOD detection      (CORES) su ID vs OOD
        9. Ablation study CORES      (opzionale via skip_ablation)
        10. Genera e salva i grafici dedicati
    """
    set_seed(42)

    # 1. Dati
    print("\n[1/8] Building data loaders ...", flush=True)
    train_loader, id_val_loader, id_test_loader, ood_test_loader = get_dataloaders()

    # 2. Modello
    if model_type.lower() in ("meter", "m"):
        from meter import METERMDE
        model_name = "METER"
        print("[2/8] Instantiating METER (Mobile Vision Transformer) ...", flush=True)
        model = METERMDE()
    else:
        model_name = "FastDepth"
        print("[2/8] Instantiating FastDepth (MobileNetV2 + NNConv5) ...", flush=True)
        model = FastDepthMDE(pretrained_encoder=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Model: {model_name}  |  Total parameters: {total_params:,}", flush=True)

    # 3. Addestramento
    print(f"[3/8] Training {model_name} on NYU Depth V2 (ID) ...", flush=True)
    from train import train_model
    epoch_losses = train_model(model, train_loader, model_name=model_name, epochs=epochs)

    # 4. Valutazione MDE
    print(f"[4/8] Evaluating {model_name} MDE on ID test set ...", flush=True)
    mde_metrics = evaluate_mde_on_id(model, id_test_loader)

    # 5. Latenza di inferenza
    print(f"[5/8] Measuring GPU inference latency for {model_name} ...", flush=True)
    sample_input = torch.randn(1, 3, *IMAGE_SIZE).to(DEVICE)
    latency = measure_inference_time_gpu(model, sample_input)
    print(f"       Inference latency: {latency:.2f} ms  "
          f"({1000.0/latency:.1f} FPS)", flush=True)

    # 6. Calibrazione TAU_POS/TAU_NEG
    print(f"[6/8] Calibrating CORES TAU_POS/TAU_NEG on trained {model_name} ...", flush=True)
    tau_pos, tau_neg, tau_diagnostics = calibrate_cores_taus(model, id_val_loader)
    if isinstance(tau_pos, dict):
        avg_tp = float(np.mean(list(tau_pos.values())))
        avg_tn = float(np.mean(list(tau_neg.values())))
        print(f"       Calibrated per-layer: avg TAU_POS={avg_tp:.4f}  avg TAU_NEG={avg_tn:.4f}  "
              f"({len(tau_pos)} layers)", flush=True)
    else:
        print(f"       Calibrated: TAU_POS={tau_pos:.4f}  TAU_NEG={tau_neg:.4f}  "
              f"(config.py defaults: {TAU_POS}/{TAU_NEG})", flush=True)

    # 7. OOD Detection
    print(f"[7/8] Evaluating CORES OOD detection for {model_name} ...", flush=True)
    ood_result = evaluate_cores_ood(model, id_val_loader, id_test_loader, ood_test_loader,
                                    tau_pos=tau_pos, tau_neg=tau_neg)

    # 8. Ablation study CORES
    if not skip_ablation:
        print(f"[8/8] Running CORES ablation study for {model_name} ...", flush=True)
        ablation_results = run_ablation_study(
            model, id_test_loader, ood_test_loader, prefix=model_name.lower(),
        )
    else:
        print(f"[8/8] Skipping CORES ablation study as requested (--skip-ablation).", flush=True)

    # 9. Grafici
    _plot_results(epoch_losses, ood_result, model_name=model_name)
    
    # --- Riepilogo ---
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Model          : {model_name}")
    print(f"  Device         : {DEVICE}")
    print(f"  Parameters     : {total_params:,}")
    print(f"  Epochs         : {epochs}")
    print(f"  Final Train Loss: {epoch_losses[-1]:.6f}")
    print(f"  MDE AbsRel     : {mde_metrics['abs_rel']:.4f}")
    print(f"  MDE RMSE       : {mde_metrics['rmse']:.4f}")
    print(f"  MDE delta_1    : {mde_metrics['delta_1']:.4f}")
    print(f"  Latency        : {latency:.2f} ms ({1000.0/latency:.1f} FPS)")
    print(f"  AUROC (non-sel.): {ood_result['auroc']:.4f}")
    print(f"  FPR95 (non-sel.): {ood_result['fpr95']:.4f}")
    print(f"  AUROC (selected): {ood_result['selected']['auroc']:.4f}")
    print(f"  FPR95 (selected): {ood_result['selected']['fpr95']:.4f}")
    print(f"{'='*60}\n")
    
