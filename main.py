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

import sys

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from evaluate import run_full_experiment

if __name__ == "__main__":
    run_full_experiment()
