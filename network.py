"""
=============================================================================
Sezione 5 — NETWORK
=============================================================================
Architettura FastDepth (MobileNetV1 encoder + NNConv5 decoder)
e meccanismo CORES per OOD detection.

Classi:
    DepthwiseSeparableConv   — blocco convoluzionale separabile in profondità
    NNConv5UpSampleBlock     — upsampling NN ×2 + 5×5 DSConv + skip additivo
    MobileNetV1Encoder       — encoder a 6 stadi (32→1024)
    NNConv5Decoder           — decoder a 5 stadi (1024→1)
    FastDepthMDE             — rete completa encoder-decoder
    ForwardHookHandler       — gestione forward hook per cattura feature map
    CORESScorer              — calcolo score OOD layer-by-layer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import LAMBDA_1, LAMBDA_2, TAU_POS, TAU_NEG


# ===========================================================================
# Blocchi base
# ===========================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Convoluzione Separabile in Profondità = Depthwise Conv + Pointwise Conv.
    Utilizzata nelle architetture MobileNet per efficienza.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_dw(self.depthwise(x)), inplace=True)
        x = F.relu(self.bn_pw(self.pointwise(x)), inplace=True)
        return x


class NNConv5UpSampleBlock(nn.Module):
    """
    Blocco decoder per FastDepth (NNConv5 upsampling).

    Pipeline:
        1. Nearest-neighbour upsample (×2)
        2. Convoluzione separabile 5×5
        3. Connessione skip additiva dall'encoder (se le dimensioni coincidono)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = DepthwiseSeparableConv(
            in_channels, out_channels, stride=1, padding=2,
        )
        # Sovrascrittura kernel a 5×5 per depthwise
        self.conv.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=5, stride=1, padding=2,
            groups=in_channels, bias=False,
        )
        self.conv.bn_dw = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)

        # Connessione skip additiva
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(
                    skip, size=x.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            if x.shape[1] == skip.shape[1]:
                x = x + skip

        return x


# ===========================================================================
# Encoder — MobileNetV1
# ===========================================================================

class MobileNetV1Encoder(nn.Module):
    """
    Encoder stile MobileNetV1 a 6 stadi.

    Produce feature map a 1/2, 1/4, 1/8, 1/16, 1/32 della risoluzione di input.
    Progressione canali: 32 → 64 → 128 → 256 → 512 → 1024
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Stage 0: conv standard  (3 → 32, stride=2)
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32 → 64  (stride=1)  → 1/2
        self.stage1 = DepthwiseSeparableConv(32, 64, stride=1)

        # Stage 2: 64 → 128 (stride=2)  → 1/4
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
        )

        # Stage 3: 128 → 256 (stride=2) → 1/8
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )

        # Stage 4: 256 → 512 (stride=2) → 1/16
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, stride=2),
            *[DepthwiseSeparableConv(512, 512, stride=1) for _ in range(5)],
        )

        # Stage 5: 512 → 1024 (stride=2) → 1/32
        self.stage5 = nn.Sequential(
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),
        )

        if pretrained:
            self._init_from_mobilenet_v2_approx()

    def _init_from_mobilenet_v2_approx(self):
        """
        Inizializzazione approssimata da torchvision MobileNetV2.
        Copia i pesi del primo strato conv (3→32) poiché le architetture
        differiscono nei livelli successivi.  Il resto usa Kaiming normal.
        """
        try:
            mv2 = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
            with torch.no_grad():
                self.stage0[0].weight.copy_(mv2.features[0][0].weight)
        except Exception:
            pass  # fallback silenzioso a init casuale

        # Kaiming init per i layer rimanenti
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.stage0[0]:
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            out:   feature map finale dell'encoder (1/32)
            skips: lista di feature map intermedie [s1, s2, s3, s4]
        """
        s0 = self.stage0(x)   # (B, 32,  112, 112)
        s1 = self.stage1(s0)  # (B, 64,  112, 112)
        s2 = self.stage2(s1)  # (B, 128,  56,  56)
        s3 = self.stage3(s2)  # (B, 256,  28,  28)
        s4 = self.stage4(s3)  # (B, 512,  14,  14)
        s5 = self.stage5(s4)  # (B, 1024,  7,   7)

        return s5, [s1, s2, s3, s4]


# ===========================================================================
# Decoder — NNConv5
# ===========================================================================

class NNConv5Decoder(nn.Module):
    """
    Decoder NNConv5 per FastDepth.

    5 stadi di upsampling: 1024 → 512 → 256 → 128 → 64 → 32
    Conv 1×1 finale per produrre la predizione monocanale di profondità.
    """

    def __init__(self):
        super().__init__()
        self.up1 = NNConv5UpSampleBlock(1024, 512)
        self.up2 = NNConv5UpSampleBlock(512, 256)
        self.up3 = NNConv5UpSampleBlock(256, 128)
        self.up4 = NNConv5UpSampleBlock(128, 64)
        self.up5 = NNConv5UpSampleBlock(64, 32)

        # Proiezione finale a singolo canale (profondità)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        """
        Args:
            x:     output dell'encoder  (B, 1024, 7, 7)
            skips: [s1(64), s2(128), s3(256), s4(512)]
        """
        x = self.up1(x, skips[3])  #  7 → 14,  skip s4 (512)
        x = self.up2(x, skips[2])  # 14 → 28,  skip s3 (256)
        x = self.up3(x, skips[1])  # 28 → 56,  skip s2 (128)
        x = self.up4(x, skips[0])  # 56 → 112, skip s1 (64)
        x = self.up5(x)            # 112→ 224, nessuno skip

        depth = F.relu(self.final_conv(x))  # profondità non-negativa
        return depth


# ===========================================================================
# FastDepth — rete completa
# ===========================================================================

class FastDepthMDE(nn.Module):
    """
    FastDepth — Monocular Depth Estimation.

    Architettura:
        Encoder: MobileNetV1 (convoluzioni separabili in profondità)
        Decoder: NNConv5 (NN upsample + 5×5 DSConv + skip additivi)

    Input:  (B, 3, 224, 224)  RGB normalizzato
    Output: (B, 1, 224, 224)  mappa di profondità predetta
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = MobileNetV1Encoder(pretrained=pretrained_encoder)
        self.decoder = NNConv5Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, skips = self.encoder(x)
        depth = self.decoder(enc_out, skips)
        return depth


# ===========================================================================
# Forward Hook Handler
# ===========================================================================

class ForwardHookHandler:
    """
    Gestisce i forward hook di PyTorch per catturare i feature map
    da layer specificati.  Utilizzato da CORESScorer per ispezionare
    le risposte convoluzionali layer-by-layer.

    Utilizzo:
        handler = ForwardHookHandler()
        handler.register(model, target_layers)
        _ = model(input)
        features = handler.get_features()   # dict[name] → Tensor
        handler.remove()
    """

    def __init__(self):
        self.features: dict = {}
        self._hooks: list = []

    def _make_hook(self, name: str):
        """Crea una closure che salva l'output del layer sotto `name`."""
        def hook_fn(module, input, output):
            self.features[name] = output.detach()
        return hook_fn

    def register(self, model: nn.Module, target_layers: dict) -> None:
        """
        Registra i forward hook sui layer indicati.

        Args:
            model:         il nn.Module
            target_layers: dict che mappa  nome → riferimento nn.Module
        """
        for name, module in target_layers.items():
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def get_features(self) -> dict:
        """Restituisce le feature catturate (da chiamare dopo un forward pass)."""
        return dict(self.features)

    def clear(self) -> None:
        """Pulisce le feature salvate (da chiamare prima del prossimo forward)."""
        self.features.clear()

    def remove(self) -> None:
        """Rimuove tutti gli hook registrati."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.features.clear()


# ===========================================================================
# CORES Scorer
# ===========================================================================

class CORESScorer:
    """
    Convolutional Response-based OOD Scoring (CORES).

    Per ogni layer monitorato l, calcola:
        RM_l^+ = media delle attivazioni > tau_pos   (Response Magnitude positiva)
        RM_l^- = media di |attivazioni| dove act < tau_neg (RM negativa)
        RF_l^+ = frazione di attivazioni > tau_pos   (Response Frequency positiva)
        RF_l^- = frazione di attivazioni < tau_neg   (RF negativa)

    Score per layer:
        S_l = λ₁ · (RM_l^+ + RM_l^-) + λ₂ · (RF_l^+ + RF_l^-)

    Score finale (su L layer):
        S = (1/L) · Σ_l S_l

    Score più alto => più probabile che il campione sia OOD.
    """

    def __init__(
        self,
        tau_pos:  float = TAU_POS,
        tau_neg:  float = TAU_NEG,
        lambda_1: float = LAMBDA_1,
        lambda_2: float = LAMBDA_2,
    ):
        self.tau_pos  = tau_pos
        self.tau_neg  = tau_neg
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def compute_layer_score(self, feat: torch.Tensor) -> float:
        """
        Calcola lo score CORES per un singolo feature map.

        Args:
            feat: (B, C, H, W) o (C, H, W) feature map da un layer.

        Returns:
            Score scalare per questo layer (media sul batch).
        """
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)

        # Flatten spaziale + canali per campione  → (B, C*H*W)
        flat = feat.view(feat.size(0), -1).float()
        n = flat.size(1)

        pos_mask = flat > self.tau_pos
        neg_mask = flat < self.tau_neg

        # RM+: magnitudine media delle attivazioni positive
        pos_vals = flat * pos_mask.float()
        rm_pos = pos_vals.sum(dim=1) / (pos_mask.sum(dim=1).float() + 1e-8)

        # RM-: magnitudine media delle |attivazioni negative|
        neg_vals = flat.abs() * neg_mask.float()
        rm_neg = neg_vals.sum(dim=1) / (neg_mask.sum(dim=1).float() + 1e-8)

        # RF+, RF-: frazioni
        rf_pos = pos_mask.float().sum(dim=1) / n
        rf_neg = neg_mask.float().sum(dim=1) / n

        # Score per campione
        score = (self.lambda_1 * (rm_pos + rm_neg)
                 + self.lambda_2 * (rf_pos + rf_neg))

        return score.mean().item()

    def compute_score(self, features: dict) -> float:
        """
        Calcola lo score CORES aggregato su tutti i layer monitorati.

        Args:
            features: dict[nome_layer] → Tensor (B, C, H, W)

        Returns:
            Media degli score per-layer (scalare).
        """
        if len(features) == 0:
            return 0.0

        layer_scores = [self.compute_layer_score(feat)
                        for feat in features.values()]
        return float(np.mean(layer_scores))

    def compute_score_per_sample(self, features: dict,
                                  batch_size: int) -> np.ndarray:
        """
        Calcola gli score CORES per-campione (per costruire array di score).

        Args:
            features:   dict[nome_layer] → Tensor (B, C, H, W)
            batch_size: B

        Returns:
            np.ndarray di forma (B,) con gli score per campione.
        """
        if len(features) == 0:
            return np.zeros(batch_size)

        per_sample = torch.zeros(batch_size, device="cpu")

        for feat in features.values():
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)

            flat = feat.view(feat.size(0), -1).float().cpu()
            n = flat.size(1)

            pos_mask = flat > self.tau_pos
            neg_mask = flat < self.tau_neg

            pos_vals = flat * pos_mask.float()
            rm_pos = pos_vals.sum(dim=1) / (pos_mask.sum(dim=1).float() + 1e-8)

            neg_vals = flat.abs() * neg_mask.float()
            rm_neg = neg_vals.sum(dim=1) / (neg_mask.sum(dim=1).float() + 1e-8)

            rf_pos = pos_mask.float().sum(dim=1) / n
            rf_neg = neg_mask.float().sum(dim=1) / n

            score = (self.lambda_1 * (rm_pos + rm_neg)
                     + self.lambda_2 * (rf_pos + rf_neg))

            per_sample += score

        per_sample /= len(features)
        return per_sample.numpy()
