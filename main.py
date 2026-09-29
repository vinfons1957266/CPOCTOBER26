#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Out-of-Distribution (OOD) Detection in Monocular Depth Estimation
using FastDepth and CORES
=============================================================================

Entry point — esegue l'esperimento completo.

Struttura del progetto (7 sezioni → 6 moduli):
    config.py    — Sezione 2: Globals (iperparametri)
    utils.py     — Sezione 3: Utils   (seed, metriche, latenza)
    data.py      — Sezione 4: Data    (NYU ID, KITTI OOD, dataloader)
    network.py   — Sezione 5: Network (FastDepth, CORES)
    train.py     — Sezione 6: Train   (AdamW + MSELoss)
    evaluate.py  — Sezione 7: Evaluation (MDE, OOD, orchestrazione)

    Sezione 1 (Imports) è distribuita nei moduli sopra.

Uso:
    python main.py
"""

import argparse
import sys

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")

from config import EPOCHS
from evaluate import run_full_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="OOD Detection in Monocular Depth Estimation with CORES (FastDepth vs METER)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meter",
        choices=["meter", "fastdepth"],
        help="Modello da eseguire in isolamento: 'meter' (Mobile ViT, default) oppure 'fastdepth'."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Numero di epoche di addestramento (default: {EPOCHS} da config.py)."
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Salta l'ablation study (risparmia oltre 50 passate per test rapidi)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_experiment(model_type=args.model, epochs=args.epochs, skip_ablation=args.skip_ablation)
