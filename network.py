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

from config import LAMBDA_1, LAMBDA_2, TAU_POS, TAU_NEG, CORES_EPS


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
        # inplace=False: CORES aggancia forward hook su bn_dw/bn_pw per
        # leggere la risposta PRIMA della ReLU. Un ReLU in-place sovrascrive
        # il buffer condiviso dal tensore già catturato dall'hook (anche se
        # .detach()'d, condivide lo storage), azzerando ogni valore negativo
        # che l'hook avrebbe dovuto osservare.
        x = F.relu(self.bn_dw(self.depthwise(x)), inplace=False)
        x = F.relu(self.bn_pw(self.pointwise(x)), inplace=False)
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
            nn.ReLU(inplace=False),  # inplace=False: vedi nota in DepthwiseSeparableConv
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
# CORES — componenti di risposta condivise (Eq. 3/4, Tang et al.)
# ===========================================================================

def cores_response_components(feat: torch.Tensor, tau_pos: float,
                               tau_neg: float):
    """
    Calcola RM+/RM-/RF+/RF- secondo le Eq. (3)/(4) del paper CORES, usando
    gli ESTREMI SPAZIALI PER CANALE — non il tensore appiattito.

    Nel paper, la "risposta" di un kernel R_c è la sua mappa spaziale
    (H, W); CORES usa solo il picco max(R_c) e il minimo min(R_c) di
    ciascun canale (non ogni singolo pixel), poi media su c = 1:C canali:
        RM+ = media_c[ max( max(R_c) - tau_pos, 0 ) ]
        RM- = media_c[ max( tau_neg - min(R_c), 0 ) ]
        RF+ = frazione di canali con max(R_c) > tau_pos
        RF- = frazione di canali con min(R_c) < tau_neg

    Usata sia da CORESScorer (per lo score aggregato) sia da
    ablation.py::cores_detailed_statistics (per le statistiche separate),
    cosi' che le due implementazioni non possano divergere.

    Args:
        feat: (B, C, H, W) o (C, H, W) — feature map pre-attivazione di un
              layer monitorato (vedi evaluate.py::_get_target_layers).
        tau_pos, tau_neg: soglie di significatività della risposta.

    Returns:
        Tupla (rm_pos, rm_neg, rf_pos, rf_neg), ciascuno Tensor di forma (B,).
    """
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    feat = feat.float()

    peak   = feat.amax(dim=(2, 3))   # (B, C) — max(R_c) per ogni canale
    trough = feat.amin(dim=(2, 3))   # (B, C) — min(R_c) per ogni canale

    rm_pos = (peak - tau_pos).clamp(min=0).mean(dim=1)
    rm_neg = (tau_neg - trough).clamp(min=0).mean(dim=1)
    rf_pos = (peak > tau_pos).float().mean(dim=1)
    rf_neg = (trough < tau_neg).float().mean(dim=1)

    return rm_pos, rm_neg, rf_pos, rf_neg


# ===========================================================================
# CORES Scorer
# ===========================================================================

class CORESScorer:
    """
    Convolutional Response-based OOD Scoring (CORES).

    Per ogni layer monitorato l, calcola (vedi cores_response_components,
    Eq. 3/4 del paper):
        RM_l^+ = media_c[ max(max(R_c) - tau_pos, 0) ]  (Response Magnitude positiva)
        RM_l^- = media_c[ max(tau_neg - min(R_c), 0) ]  (RM negativa)
        RF_l^+ = frazione di canali con max(R_c) > tau_pos  (Response Frequency positiva)
        RF_l^- = frazione di canali con min(R_c) < tau_neg  (RF negativa)

    Score per layer (Eq. 5/9 del paper — prodotto, non somma pesata):
        S_l = RM_l^+ ^ λ₁ · RM_l^- ^ λ₁ · RF_l^+ ^ λ₂ · RF_l^- ^ λ₂

    Con λ₁=10 il prodotto va in underflow float32 (RM ~ 1e-2-1e-3 per campione
    ID → S_l ~ 1e-20+). Si calcola quindi log(S_l), che e' una trasformazione
    monotona equivalente ai fini di AUROC/FPR95/soglia:
        log S_l = λ₁·(log(RM_l^+ + ε) + log(RM_l^- + ε))
                + λ₂·(log(RF_l^+ + ε) + log(RF_l^- + ε))

    Score finale (su L layer):
        log S = (1/L) · Σ_l log S_l

    Convenzione del paper (Eq. 1): score più alto => più probabile ID, non
    OOD (i kernel rispondono più intensamente a campioni ID). La direzione
    usata da utils.py va allineata a questa convenzione (vedi piano, punto A5).
    """

    def __init__(
        self,
        tau_pos:  float = TAU_POS,
        tau_neg:  float = TAU_NEG,
        lambda_1: float = LAMBDA_1,
        lambda_2: float = LAMBDA_2,
        eps:      float = CORES_EPS,
    ):
        self.tau_pos  = tau_pos
        self.tau_neg  = tau_neg
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.eps      = eps

    def compute_layer_score(self, feat: torch.Tensor) -> float:
        """
        Calcola lo score CORES (in log-spazio, Eq. 5/9) per un singolo
        feature map.

        Args:
            feat: (B, C, H, W) o (C, H, W) feature map da un layer.

        Returns:
            log-score scalare per questo layer (media sul batch).
        """
        rm_pos, rm_neg, rf_pos, rf_neg = cores_response_components(
            feat, self.tau_pos, self.tau_neg)

        # log(S) per campione — prodotto di Eq. 5/9 in log-spazio (vedi
        # docstring della classe per la derivazione e la motivazione)
        log_score = (
            self.lambda_1 * (torch.log(rm_pos + self.eps) + torch.log(rm_neg + self.eps))
            + self.lambda_2 * (torch.log(rf_pos + self.eps) + torch.log(rf_neg + self.eps))
        )

        return log_score.mean().item()

    def compute_score(self, features: dict) -> float:
        """
        Calcola lo score CORES aggregato su tutti i layer monitorati.

        Args:
            features: dict[nome_layer] → Tensor (B, C, H, W)

        Returns:
            Media dei log-score per-layer (scalare).
        """
        if len(features) == 0:
            return 0.0

        layer_scores = [self.compute_layer_score(feat)
                        for feat in features.values()]
        return float(np.mean(layer_scores))

    def compute_score_per_sample(self, features: dict,
                                  batch_size: int) -> np.ndarray:
        """
        Calcola gli score CORES (log-spazio, Eq. 5/9) per-campione.

        Args:
            features:   dict[nome_layer] → Tensor (B, C, H, W)
            batch_size: B

        Returns:
            np.ndarray di forma (B,) con i log-score per campione.
        """
        if len(features) == 0:
            return np.zeros(batch_size)

        per_sample = torch.zeros(batch_size, device="cpu")

        for feat in features.values():
            rm_pos, rm_neg, rf_pos, rf_neg = cores_response_components(
                feat.cpu(), self.tau_pos, self.tau_neg)

            log_score = (
                self.lambda_1 * (torch.log(rm_pos + self.eps) + torch.log(rm_neg + self.eps))
                + self.lambda_2 * (torch.log(rf_pos + self.eps) + torch.log(rf_neg + self.eps))
            )

            per_sample += log_score

        per_sample /= len(features)
        return per_sample.numpy()

    @staticmethod
    def _gather_channels(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Estrae, per ciascun campione, il sottoinsieme di canali indicato da
        `indices` (B, k) dal feature map `feat` (B, C, H, W).

        Args:
            feat:    (B, C, H, W)
            indices: (B, k) — indici di canale, PER CAMPIONE.

        Returns:
            Tensor (B, k, H, W).
        """
        b, k = indices.shape
        _, _, h, w = feat.shape
        idx = indices.view(b, k, 1, 1).expand(-1, -1, h, w)
        return feat.gather(1, idx)

    def compute_score_per_sample_selected(self, features: dict,
                                          selections: dict,
                                          batch_size: int) -> np.ndarray:
        """
        Calcola gli score CORES (Eq. 9, CON selezione dei kernel
        sample-relevant, Sez. 4.2) per-campione.

        A differenza di compute_score_per_sample (che usa TUTTI i canali
        di ogni layer, Eq. 5), qui RM+/RF+ sono calcolati SOLO sul
        sottoinsieme di canali I_pos, e RM-/RF- SOLO sul sottoinsieme
        I_neg:
            S(R̄∪R̲) = RM+(R̄)^λ1 · RM-(R̲)^λ1 · RF+(R̄)^λ2 · RF-(R̲)^λ2
        (in log-spazio, come compute_score_per_sample — vedi Eq. 9 e la
        nota sull'underflow nella docstring di classe).

        I due sottoinsiemi NON sono necessariamente disgiunti (vedi
        CORESKernelSelector.backtrack) — nessun vincolo di esclusività è
        imposto dal paper tra R̄ e R̲.

        Args:
            features:   dict[nome_layer] → Tensor (B, C, H, W) — le stesse
                        feature map tappate usate da compute_score_per_sample.
                        Può contenere layer extra (es. l'output grezzo di
                        final_conv, usato solo per la selezione iniziale)
                        che non compaiono in `selections`: vengono ignorati.
            selections: dict[nome_layer] → (I_pos, I_neg), da
                        CORESKernelSelector.backtrack() — LongTensor (B, k)
                        ciascuno (k varia per layer).
            batch_size: B

        Returns:
            np.ndarray di forma (B,) con i log-score per campione.
        """
        if len(selections) == 0:
            return np.zeros(batch_size)

        per_sample = torch.zeros(batch_size, device="cpu")
        n_layers = 0

        for name, feat in features.items():
            feat = feat.cpu()

            if name in selections:
                i_pos, i_neg = selections[name]
                feat_pos = self._gather_channels(feat, i_pos.cpu())  # (B, k_pos, H, W)
                feat_neg = self._gather_channels(feat, i_neg.cpu())  # (B, k_neg, H, W)
            else:
                # Per i layer dell'encoder (non inclusi nella catena di backtracking del decoder):
                # identifichiamo i kernel sample-relevant per-sample in base all'intensità di risposta:
                # canali con picchi più alti per i_pos, canali con valli più basse per i_neg (top 20%).
                num_c = feat.shape[1]
                k = max(1, round(0.20 * num_c))
                peak = feat.amax(dim=(2, 3))    # (B, C)
                trough = feat.amin(dim=(2, 3))  # (B, C)
                i_pos = peak.topk(k, dim=1, largest=True).indices
                i_neg = trough.topk(k, dim=1, largest=False).indices

                feat_pos = self._gather_channels(feat, i_pos)
                feat_neg = self._gather_channels(feat, i_neg)

            rm_pos, _, rf_pos, _ = cores_response_components(
                feat_pos, self.tau_pos, self.tau_neg)
            _, rm_neg, _, rf_neg = cores_response_components(
                feat_neg, self.tau_pos, self.tau_neg)

            log_score = (
                self.lambda_1 * (torch.log(rm_pos + self.eps) + torch.log(rm_neg + self.eps))
                + self.lambda_2 * (torch.log(rf_pos + self.eps) + torch.log(rf_neg + self.eps))
            )

            per_sample += log_score
            n_layers += 1

        if n_layers == 0:
            return np.zeros(batch_size)

        per_sample /= n_layers
        return per_sample.numpy()


# ===========================================================================
# CORES Kernel Selector — selezione sample-relevant e backtracking (Sez. 4.2)
# ===========================================================================

class CORESKernelSelector:
    """
    Selezione dei kernel sample-relevant e backtracking attraverso i pesi
    del decoder (Eq. 6-8 del paper CORES).

    FastDepth non ha un layer fully-connected né categorie c_max/c_min —
    è un regressore denso a singola uscita (o=1), non un classificatore —
    quindi Eq. 6/8 vanno adattate:

    - La traiettoria di backtracking (questa classe, B1 del piano) è
      ristretta al SOLO decoder: 5 nodi (dec_up5..dec_up1), collegati da
      4 pesi pointwise 1×1. Non si estende nell'encoder (vedi nota su
      up1.conv.pointwise più sotto) né oltre dec_up1.
    - La selezione al nodo iniziale (dec_up5, Eq. 6) e i passi di
      backtracking veri e propri (Eq. 8) sono implementati separatamente
      (vedi piano, punti B2/B3) — questa classe fornisce per ora solo
      l'infrastruttura statica: la catena di pesi e il peso di
      final_conv (l'analogo di F in Eq. 6).

    I quattro nodi encoder-side non compaiono qui: le connessioni skip
    additive del decoder (up1..up4) sono somme dirette non pesate, non
    matrici K apprese — backtrackare attraverso di esse degenererebbe
    in una selezione uniforme (ogni canale riceve lo stesso peso "1"),
    quindi non porterebbero informazione utile a Eq. 8.
    """

    def __init__(self, model: FastDepthMDE):
        self.final_conv_weight = self._to_2d(model.decoder.final_conv.weight)
        self.chain = self._backtracking_chain(model)

    @staticmethod
    def _to_2d(weight: torch.Tensor) -> torch.Tensor:
        """
        (c_out, c_in, 1, 1) -> (c_out, c_in). I pesi pointwise (e
        final_conv) sono sempre kernel 1×1: max/min sulla dimensione
        spaziale (Eq. 8) coincidono quindi col valore stesso della matrice,
        e la riduzione a 2D non perde informazione.
        """
        return weight.reshape(weight.shape[0], weight.shape[1])

    @classmethod
    def _backtracking_chain(cls, model: FastDepthMDE) -> list:
        """
        Costruisce la traiettoria di backtracking a 5 nodi / 4 hop
        (dec_up5 -> dec_up4 -> dec_up3 -> dec_up2 -> dec_up1), SOLO
        decoder — vedi Decisione 2 del piano.

        Returns:
            Lista ordinata di tuple (nome_nodo, peso_2D_o_None):
                [("dec_up5", None),      # nodo iniziale: nessun hop; la
                                          #   selezione qui usa final_conv
                                          #   (Eq. 6, vedi B2 — non incluso
                                          #   in questa classe per ora)
                 ("dec_up4", W1),        # hop 1: dec_up5 -> dec_up4,
                                          #   W1 = up5.conv.pointwise (32,64)
                 ("dec_up3", W2),        # hop 2: dec_up4 -> dec_up3,
                                          #   W2 = up4.conv.pointwise (64,128)
                 ("dec_up2", W3),        # hop 3: dec_up3 -> dec_up2,
                                          #   W3 = up3.conv.pointwise (128,256)
                 ("dec_up1", W4)]        # hop 4: dec_up2 -> dec_up1,
                                          #   W4 = up2.conv.pointwise (256,512)

            Ogni peso W ha forma (c_out, c_in): c_out = canali del nodo
            PRECEDENTE nella lista (quello "verso l'output"), c_in = canali
            del nodo CORRENTE — la stessa convenzione di K in Eq. 8.

            NOTA: up1.conv.pointwise.weight (512, 1024) NON compare in
            questa catena. Servirebbe solo per proseguire il backtracking
            OLTRE dec_up1, dentro l'encoder (il suo input è l'output finale
            dell'encoder, s5/enc_stage5) — la Decisione 2 esclude
            esplicitamente questa estensione.
        """
        return [
            ("dec_up5", None),
            ("dec_up4", cls._to_2d(model.decoder.up5.conv.pointwise.weight)),
            ("dec_up3", cls._to_2d(model.decoder.up4.conv.pointwise.weight)),
            ("dec_up2", cls._to_2d(model.decoder.up3.conv.pointwise.weight)),
            ("dec_up1", cls._to_2d(model.decoder.up2.conv.pointwise.weight)),
        ]

    def select_initial_indices(self, raw_depth: torch.Tensor,
                                feat_dec_up5: torch.Tensor, k: int):
        """
        Selezione al nodo iniziale (dec_up5), analogo di Eq. 6/7 per un
        regressore denso a singola uscita (FastDepth: o=1, nessuna
        categoria c_max/c_min su cui scegliere una riga di F).

        Sostituzione riga -> pixel: il pixel dove la profondità PREDETTA
        è più alta gioca il ruolo di c_max, quello dove è più bassa il
        ruolo di c_min. Poiché final_conv è una conv 1×1, il valore
        predetto in un pixel è esattamente Σⱼ F0[j]·x[j, pixel] — quindi
        F0[j]·x[j, pixel] è il termine j-esimo di quella somma, e sostituisce
        F_{c_max,j} come criterio di importanza per canale.

        NOTA su TopK vs BotK: qui si usa TopK per ENTRAMBI i rami (come
        Eq. 6, che usa TopK sia per c_max sia per c_min) — non l'asimmetria
        TopK/BotK di Eq. 8, che si applica solo ai passi di backtracking
        intermedi (vedi B3). Il prodotto F0·x è firmato, quindi TopK su di
        esso premia sia (F0 grande positivo, x grande positivo) sia
        (F0 grande negativo, x grande negativo): entrambi i casi spiegano
        fortemente il valore osservato al pixel.

        Args:
            raw_depth:    (B, 1, H, W) — output GREZZO di final_conv, PRIMA
                          della ReLU. model.forward() restituisce solo la
                          versione già passata per ReLU: serve un hook
                          dedicato su model.decoder.final_conv per ottenere
                          questo tensore.
            feat_dec_up5: (B, 32, H, W) — tap pre-attivazione di dec_up5
                          (lo stesso già catturato dai target layer
                          esistenti in evaluate.py/ablation.py).
            k: numero di canali da selezionare (TopK), tipicamente
               round(TOPK_FRAC * 32).

        Returns:
            (I_pos, I_neg): LongTensor (B, k) ciascuno — indici di canale
            PER CAMPIONE (la selezione dipende dalla predizione, quindi
            varia da campione a campione).
        """
        assert raw_depth.shape[2:] == feat_dec_up5.shape[2:], (
            "raw_depth e feat_dec_up5 devono condividere la risoluzione "
            "spaziale (entrambi 224x224 nell'architettura attuale)"
        )

        batch_size, num_channels = feat_dec_up5.shape[0], feat_dec_up5.shape[1]
        k = max(1, min(k, num_channels))

        f0 = self.final_conv_weight.detach().reshape(-1)  # (32,)

        # Invece di un singolo pixel rumoroso (soggetto ad artefatti di padding ai bordi),
        # consideriamo le regioni di profondità estrema (top 10% e bottom 10% dei pixel).
        flat_depth = raw_depth.flatten(2)                          # (B, 1, HW)
        num_pixels = flat_depth.shape[-1]
        p_count = max(1, int(0.10 * num_pixels))

        # Pixel appartenenti alle regioni più lontane (far) e più vicine (near)
        top_indices = flat_depth.topk(p_count, dim=-1, largest=True).indices   # (B, 1, P)
        bot_indices = flat_depth.topk(p_count, dim=-1, largest=False).indices  # (B, 1, P)

        # Risposta media di ciascun canale in quelle regioni
        flat_feat = feat_dec_up5.flatten(2)                        # (B, C, HW)
        top_exp = top_indices.expand(-1, num_channels, -1)         # (B, C, P)
        bot_exp = bot_indices.expand(-1, num_channels, -1)         # (B, C, P)

        response_far  = flat_feat.gather(2, top_exp).mean(dim=-1)  # (B, C)
        response_near = flat_feat.gather(2, bot_exp).mean(dim=-1)  # (B, C)

        # Contributo ponderato per il peso della proiezione finale
        contrib_far  = f0.unsqueeze(0) * response_far              # (B, C)
        contrib_near = f0.unsqueeze(0) * response_near             # (B, C)

        # Selezioniamo i canali con maggiore contributo relativo:
        # i_pos: canali che guidano maggiormente le risposte positive (far)
        # i_neg: canali che guidano maggiormente le risposte negative (near)
        i_pos = contrib_far.topk(k, dim=1, largest=True).indices                 # (B, k)
        i_neg = contrib_near.topk(k, dim=1, largest=False).indices               # (B, k)

        return i_pos, i_neg

    def backtrack(self, initial_pos: torch.Tensor, initial_neg: torch.Tensor,
                  topk_frac: float) -> dict:
        """
        Backtracking all'indietro lungo self.chain (Eq. 8 del paper CORES),
        a partire dagli indici scelti al nodo iniziale dec_up5 (tipicamente
        da select_initial_indices, B2).

        Per i kernel pointwise 1x1 usati qui, max(K_{i,j,:,:}) =
        min(K_{i,j,:,:}) = K_{i,j} (un solo valore, nessuna estensione
        spaziale su cui prendere un estremo) — quindi l'Eq. 8 collassa a:
            Ī^(l) = TopK( Σ_{i in Ī^(l+1)} W[i, j] )
            I̲^(l) = BotK( Σ_{i in I̲^(l+1)} W[i, j] )
        dove W e' il peso pointwise 2D che connette il layer l al layer
        l+1 (vedi self.chain, B1). Con kernel 1x1 l'UNICA differenza tra
        i due rami e' quale insieme di indici entra nella somma e se poi
        si prende TopK o BotK del risultato — non un'estrazione max/min
        diversa sul peso stesso (che qui e' un singolo scalare per i,j).

        Nota: k viene ricalcolato ad OGNI hop come round(topk_frac * C_l),
        C_l = canali del layer l — non e' lo stesso k del nodo iniziale,
        perche' il numero di canali cresce risalendo la catena
        (32->64->128->256->512).

        Args:
            initial_pos, initial_neg: LongTensor (B, k0) — indici scelti
                al nodo iniziale (dec_up5), da select_initial_indices.
            topk_frac: frazione di canali da selezionare ad ogni layer
                (tipicamente TOPK_FRAC da config.py, es. 0.20).

        Returns:
            dict[nome_nodo] -> (I_pos, I_neg), per OGNI nodo della catena
            (incluso quello iniziale, con gli indici passati in input
            invariati). Ciascun I_pos/I_neg e' LongTensor (B, k_l), con
            k_l che varia da nodo a nodo.
        """
        results = {}
        current_pos, current_neg = initial_pos, initial_neg

        for name, weight in self.chain:
            if weight is None:
                # Nodo iniziale (dec_up5): nessun hop, indici gia' forniti.
                results[name] = (current_pos, current_neg)
                continue

            num_channels = weight.shape[1]
            k = max(1, round(topk_frac * num_channels))

            # Somma per-campione dei pesi delle righe selezionate al passo
            # precedente (Eq. 8): weight[current_pos] usa indicizzazione
            # avanzata di PyTorch, che con un indice (B, k) su una matrice
            # (c_out, c_in) produce (B, k, c_in) senza bisogno di gather.
            pos_scores = weight[current_pos].sum(dim=1)   # (B, c_in)
            neg_scores = weight[current_neg].sum(dim=1)   # (B, c_in)

            current_pos = pos_scores.topk(k, dim=1, largest=True).indices
            current_neg = neg_scores.topk(k, dim=1, largest=False).indices

            results[name] = (current_pos, current_neg)

        return results
