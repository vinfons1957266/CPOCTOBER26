"""
=============================================================================
Sezione 2 — GLOBALS
=============================================================================
Variabili globali e iperparametri chiave per il progetto
FastDepth + CORES OOD Detection.
"""

import torch

# ---- Device ----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data ------------------------------------------------------------------
IMAGE_SIZE  = (224, 224)   # (H, W) risoluzione di input
BATCH_SIZE  = 16
NUM_WORKERS = 2

# ---- Training --------------------------------------------------------------
EPOCHS        = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4

# ---- CORES hyper-parameters ------------------------------------------------
LAMBDA_1 = 10.0    # peso per la componente Response Magnitude
LAMBDA_2 = 1.0     # peso per la componente Response Frequency
TAU_POS  = 0.5     # soglia di attivazione positiva
TAU_NEG  = -0.5    # soglia di attivazione negativa
