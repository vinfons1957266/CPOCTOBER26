"""
=============================================================================
Sezione 2 — GLOBALS
=============================================================================
Variabili globali e iperparametri chiave per il progetto
FastDepth + CORES OOD Detection.
"""

import torch

# ---- Device ----------------------------------------------------------------
try:
    import torch_directml
    DEVICE = torch_directml.device()
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data ------------------------------------------------------------------
IMAGE_SIZE  = (224, 224)   # (H, W) risoluzione di input
BATCH_SIZE  = 16
NUM_WORKERS = 2

# ---- Training --------------------------------------------------------------
EPOCHS        = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4

# ---- CORES hyper-parameters ------------------------------------------------
LAMBDA_1 = 10.0    # peso per la componente Response Magnitude
LAMBDA_2 = 1.0     # peso per la componente Response Frequency
TAU_POS  = 0.5     # soglia di attivazione positiva
TAU_NEG  = -0.5    # soglia di attivazione negativa
CORES_EPS = 1e-6   # floor numerico stabile per i log() nello score in log-spazio (Eq. 5/9)
TOPK_FRAC = 0.20   # frazione di canali selezionati ad ogni layer (Sez. 5.1: top/bottom 20%)
# ---- OOD threshold calibration --------------------------------------------
VAL_SPLIT  = 0.15   # frazione di NYU-train riservata alla calibrazione (no training)
TNR_TARGET = 0.95   # quantile degli score ID di validazione → TNR95