"""
=============================================================================
METER — Mobile Vision Transformer for Monocular Depth Estimation
Riferimento: Guerrer, Papa, Planamente, Amerini et al. (arXiv:2403.08368)
=============================================================================

Questo modulo implementa l'architettura METER ottimizzata per CORES OOD Detection.

Migliorie architetturali e accademico-scientifiche:
    1. Transfer Learning:
       Gli stadi iniziali dell'encoder (Stage 0, 1, 2) sono basati sul backbone
       MobileNetV2 pre-addestrato su ImageNet-1K (torchvision). Questo fornisce
       filtri visivi maturi ed estrae rappresentazioni semantiche ricche
       fondamentali per la sensibilità del meccanismo CORES (Tang et al.).
    2. Modulo Ibrido METERBlock (Local-Global-Local):
       Combina convoluzioni locali con un TransformerEncoderLayer applicato su
       patch 2x2 tramite folding/unfolding 4D (F.pixel_unshuffle / F.pixel_shuffle),
       garantendo compatibilità e velocità nativa sia su DirectML che CUDA/CPU.
    3. Proiezioni 1x1 Pre-Attivazione per CORES:
       I tap point per CORES sono posizionati sulle BatchNorm delle proiezioni
       prima di qualsiasi ReLU, preservando intatta la risposta negativa (RM-, RF-).
    4. Canalizzazione Adeguata per Backtracking:
       I canali dell'encoder (64, 128, 256, 384, 512) e del decoder (384, 256, 128, 64, 32)
       forniscono una popolazione sufficientemente ampia affinché la selezione
       del top 20% (TOPK_FRAC) isoli sottoinsiemi statisticamente stabili e altamente
       discriminativi per OOD detection.
    5. METERKernelSelector:
       Catena di backtracking a 5 nodi decoder (Eq. 6-8) per la selezione
       dei kernel sample-relevant.
"""

import math
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ===========================================================================
# 1. Componenti del Vision Transformer
# ===========================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) per sequenze di token spaziali.
    Formula di attenzione scaled dot-product:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) deve essere divisibile per num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # ciascuno: (B, num_heads, N, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        out = self.out_proj(out)
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Strato Transformer Encoder Pre-LayerNorm:
        x = x + MHSA(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ===========================================================================
# 2. Blocco Ibrido METERBlock (Local-Global-Local)
# ===========================================================================

class InvertedResidualBlock(nn.Module):
    """
    Blocco MobileNetV2 Inverted Residual:
    1x1 Conv (espansione) -> 3x3 Depthwise Conv -> 1x1 Pointwise Conv.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=False),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class METERBlock(nn.Module):
    """
    Blocco Ibrido METER:
        1. Rappresentazione Locale: Conv 3x3 + Pointwise 1x1 verso transformer_dim
        2. Unfold 4D: pixel_unshuffle (p=2) -> token spaziali
        3. Attenzione Globale: TransformerEncoderLayer
        4. Fold 4D: pixel_shuffle (p=2) -> ripristino mappa 2D
        5. Fusione Ibrida: Concat[input, trans_out] -> Conv 1x1 + BatchNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int = 64,
        patch_size: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim

        # 1. Rappresentazione locale
        self.local_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn_local1 = nn.BatchNorm2d(in_channels)
        self.local_conv2 = nn.Conv2d(in_channels, transformer_dim, kernel_size=1, bias=False)
        self.bn_local2 = nn.BatchNorm2d(transformer_dim)

        # 2. Transformer globale (embed_dim = transformer_dim * p^2)
        embed_dim = transformer_dim * (patch_size ** 2)
        self.transformer = TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads)

        # 3. Fusione finale
        self.fusion_conv = nn.Conv2d(in_channels + transformer_dim, out_channels, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        p = self.patch_size

        # Estrazione locale
        loc = F.relu(self.bn_local1(self.local_conv1(x)), inplace=False)
        loc = F.relu(self.bn_local2(self.local_conv2(loc)), inplace=False)

        # Padding riflessivo se dimensioni non divisibili per patch_size (es. 7x7 -> 8x8)
        pad_h = (p - (h % p)) % p
        pad_w = (p - (w % p)) % p
        if pad_h > 0 or pad_w > 0:
            loc = F.pad(loc, (0, pad_w, 0, pad_h), mode="reflect")

        # Unfold 4D nativo
        loc_unfolded = F.pixel_unshuffle(loc, downscale_factor=p)  # (B, D * P^2, H/p, W/p)
        b, c_unf, hp, wp = loc_unfolded.shape
        tokens = loc_unfolded.flatten(2).permute(0, 2, 1)          # (B, N, D * P^2)

        # Transformer globale
        trans_tokens = self.transformer(tokens)                    # (B, N, D * P^2)

        # Fold 4D nativo
        trans_unfolded = trans_tokens.permute(0, 2, 1).reshape(b, c_unf, hp, wp)
        trans_folded = F.pixel_shuffle(trans_unfolded, upscale_factor=p)  # (B, D, H_pad, W_pad)

        # Rimozione padding
        if pad_h > 0 or pad_w > 0:
            trans_folded = trans_folded[:, :, :h, :w]

        # Fusione Ibrida
        fused = torch.cat([residual, trans_folded], dim=1)
        out = F.relu(self.bn_fusion(self.fusion_conv(fused)), inplace=False)
        return out


# ===========================================================================
# 3. METER Encoder (con Transfer Learning MobileNetV2 ImageNet)
# ===========================================================================

class METEREncoder(nn.Module):
    """
    Encoder Ibrido METER con Backbone MobileNetV2 pre-addestrato:
        - Stage 0: MobileNetV2 feats[0:2] (16 ch, 112x112)  [Pre-trained ImageNet]
        - Stage 1: MobileNetV2 feats[2:4] (24 ch, 56x56)    [Pre-trained ImageNet]
        - Stage 2: MobileNetV2 feats[4:7] (32 ch, 28x28)    [Pre-trained ImageNet]
        - Stage 3: InvertedResidual + METERBlock (64 ch, 14x14)
        - Stage 4: InvertedResidual + METERBlock (128 ch, 7x7, Bottleneck)

    Proiezioni 1x1 + BatchNorm (senza ReLU) per tap CORES pre-attivazione:
        proj_s0:          16  -> 64  (112x112)
        proj_s1:          24  -> 128 (56x56)
        proj_s2:          32  -> 256 (28x28)
        proj_s3:          64  -> 384 (14x14)
        proj_bottleneck: 128  -> 512 (7x7)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        feats = list(backbone.features.children())

        # Stadi convoluzionali superficiali con pesi ImageNet
        self.stage0 = nn.Sequential(*feats[0:2])  # 224 -> 112, 16 ch
        self.stage1 = nn.Sequential(*feats[2:4])  # 112 -> 56,  24 ch
        self.stage2 = nn.Sequential(*feats[4:7])  # 56  -> 28,  32 ch

        # Stadi profondi con METER Blocks
        self.stage3_down = InvertedResidualBlock(32, 64, stride=2, expand_ratio=2)  # 28 -> 14
        self.stage3_meter = METERBlock(64, 64, transformer_dim=64, patch_size=2)

        self.stage4_down = InvertedResidualBlock(64, 128, stride=2, expand_ratio=2) # 14 -> 7
        self.stage4_meter = METERBlock(128, 128, transformer_dim=96, patch_size=2)

        # Proiezioni 1x1 con BatchNorm (senza ReLU) per isolare tap CORES
        self.proj_s0 = nn.Sequential(nn.Conv2d(16, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.proj_s1 = nn.Sequential(nn.Conv2d(24, 128, 1, bias=False), nn.BatchNorm2d(128))
        self.proj_s2 = nn.Sequential(nn.Conv2d(32, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.proj_s3 = nn.Sequential(nn.Conv2d(64, 384, 1, bias=False), nn.BatchNorm2d(384))
        self.proj_bottleneck = nn.Sequential(nn.Conv2d(128, 512, 1, bias=False), nn.BatchNorm2d(512))

        self._init_custom_layers()

    def _init_custom_layers(self):
        for proj in [self.proj_s0, self.proj_s1, self.proj_s2, self.proj_s3, self.proj_bottleneck]:
            nn.init.kaiming_normal_(proj[0].weight, mode="fan_out", nonlinearity="relu")
            nn.init.ones_(proj[1].weight)
            nn.init.zeros_(proj[1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        s0 = self.stage0(x)                       # (B, 16,  112, 112)
        s1 = self.stage1(s0)                      # (B, 24,   56,  56)
        s2 = self.stage2(s1)                      # (B, 32,   28,  28)
        s3 = self.stage3_meter(self.stage3_down(s2)) # (B, 64,   14,  14)
        s4 = self.stage4_meter(self.stage4_down(s3)) # (B, 128,   7,   7)

        p0 = self.proj_s0(s0)                     # (B, 64,  112, 112)
        p1 = self.proj_s1(s1)                     # (B, 128,  56,  56)
        p2 = self.proj_s2(s2)                     # (B, 256,  28,  28)
        p3 = self.proj_s3(s3)                     # (B, 384,  14,  14)
        out = self.proj_bottleneck(s4)            # (B, 512,   7,   7)

        return out, [p0, p1, p2, p3]


# ===========================================================================
# 4. METER Decoder (con popolazioni di canali adeguate per CORES)
# ===========================================================================

class METERUpBlock(nn.Module):
    """
    Blocco di upsampling convoluzionale con Depthwise-Separable Convolution
    e connessione skip additiva (come da architettura METER/FastDepth):
        1. Upsample x2 (bilineare)
        2. Depthwise Conv 3x3 (in_channels -> in_channels) + BatchNorm + ReLU
        3. Pointwise Conv 1x1 (in_channels -> out_channels) + BatchNorm (TAP CORES)
        4. Skip connection additiva dall'encoder (se presente e con stessi canali)
        5. Attivazione ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1,
            groups=in_channels, bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)  # <--- TAP POINT CORES PRE-ATTIVAZIONE

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        x = F.relu(self.bn_dw(self.depthwise(x)), inplace=False)
        x = self.bn_pw(self.pointwise(x))  # tap prima della ReLU
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            if x.shape[1] == skip.shape[1]:
                x = x + skip
        return F.relu(x, inplace=False)


class METERDecoder(nn.Module):
    """
    Decoder convoluzionale a 5 stadi:
        up1: 7  -> 14  (512 in -> 384 out, skip p3: 384)
        up2: 14 -> 28  (384 in -> 256 out, skip p2: 256)
        up3: 28 -> 56  (256 in -> 128 out, skip p1: 128)
        up4: 56 -> 112 (128 in -> 64  out, skip p0: 64)
        up5: 112-> 224 (64  in -> 32  out, skip: None)
        final_conv: Conv 1x1 (32 -> 1)
    """

    def __init__(self):
        super().__init__()
        self.up1 = METERUpBlock(in_channels=512, out_channels=384)
        self.up2 = METERUpBlock(in_channels=384, out_channels=256)
        self.up3 = METERUpBlock(in_channels=256, out_channels=128)
        self.up4 = METERUpBlock(in_channels=128, out_channels=64)
        self.up5 = METERUpBlock(in_channels=64,  out_channels=32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        # skips: [p0(64), p1(128), p2(256), p3(384)]
        x = self.up1(x, skips[3])  # 7   -> 14,  skip p3 (384)
        x = self.up2(x, skips[2])  # 14  -> 28,  skip p2 (256)
        x = self.up3(x, skips[1])  # 28  -> 56,  skip p1 (128)
        x = self.up4(x, skips[0])  # 56  -> 112, skip p0 (64)
        x = self.up5(x, None)      # 112 -> 224, nessun skip

        depth = F.relu(self.final_conv(x))
        return depth


# ===========================================================================
# 5. Rete Completa METERMDE
# ===========================================================================

class METERMDE(nn.Module):
    """
    Rete METER completa per Monocular Depth Estimation.
    Fornisce l'interfaccia unificata per CORES OOD Detection.
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = METEREncoder(pretrained=pretrained_encoder)
        self.decoder = METERDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, skips = self.encoder(x)
        depth = self.decoder(enc_out, skips)
        return depth

    def get_cores_target_layers(self) -> Dict[str, nn.Module]:
        """
        Restituisce i 10 layer pre-attivazione (BatchNorm) monitorati da CORES.
        Tutti i layer sono BatchNorm pre-attivazione (nessuna ReLU intermedia),
        preservando le risposte convoluzionali sia positive che negative.
        """
        targets = OrderedDict()
        targets["enc_stage0"] = self.encoder.proj_s0[1]           # 64 ch
        targets["enc_stage1"] = self.encoder.proj_s1[1]           # 128 ch
        targets["enc_stage2"] = self.encoder.proj_s2[1]           # 256 ch
        targets["enc_stage3"] = self.encoder.proj_s3[1]           # 384 ch
        targets["enc_stage4"] = self.encoder.proj_bottleneck[1]   # 512 ch

        targets["dec_up1"] = self.decoder.up1.bn_pw               # 384 ch
        targets["dec_up2"] = self.decoder.up2.bn_pw               # 256 ch
        targets["dec_up3"] = self.decoder.up3.bn_pw               # 128 ch
        targets["dec_up4"] = self.decoder.up4.bn_pw               # 64 ch
        targets["dec_up5"] = self.decoder.up5.bn_pw               # 32 ch
        return targets

    def get_cores_target_layers_with_final_conv(self) -> Dict[str, nn.Module]:
        """Come get_cores_target_layers, ma include final_conv grezzo per il backtracking."""
        targets = self.get_cores_target_layers()
        targets["final_conv_raw"] = self.decoder.final_conv
        return targets


# ===========================================================================
# 6. CORES Kernel Selector per METER
# ===========================================================================

class METERKernelSelector:
    """
    Selezione dei kernel sample-relevant e backtracking (CORES Eq. 6-8)
    specializzato per il decoder convoluzionale di METER.
    """

    def __init__(self, model: METERMDE):
        self.final_conv_weight = self._to_2d(model.decoder.final_conv.weight)
        self.chain = self._backtracking_chain(model)

    @staticmethod
    def _to_2d(weight: torch.Tensor) -> torch.Tensor:
        return weight.reshape(weight.shape[0], weight.shape[1])

    @classmethod
    def _backtracking_chain(cls, model: METERMDE) -> List[Tuple[str, torch.Tensor]]:
        """
        Catena di backtracking a 5 nodi decoder:
        dec_up5 (32) -> dec_up4 (64) -> dec_up3 (128) -> dec_up2 (256) -> dec_up1 (384).
        """
        return [
            ("dec_up5", None),
            ("dec_up4", cls._to_2d(model.decoder.up5.pointwise.weight)),
            ("dec_up3", cls._to_2d(model.decoder.up4.pointwise.weight)),
            ("dec_up2", cls._to_2d(model.decoder.up3.pointwise.weight)),
            ("dec_up1", cls._to_2d(model.decoder.up2.pointwise.weight)),
        ]

    def select_initial_indices(
        self, raw_depth: torch.Tensor, feat_dec_up5: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels = feat_dec_up5.shape[0], feat_dec_up5.shape[1]
        k = max(1, min(k, num_channels))

        flat_depth = raw_depth.flatten(2)
        num_pixels = flat_depth.shape[-1]
        p_count = max(1, int(0.10 * num_pixels))

        _, far_indices = torch.topk(flat_depth, k=p_count, dim=-1, largest=True)
        _, near_indices = torch.topk(flat_depth, k=p_count, dim=-1, largest=False)

        feat_flat = feat_dec_up5.flatten(2)
        f0 = self.final_conv_weight.detach().reshape(-1).to(feat_dec_up5.device)

        idx_far = far_indices.expand(-1, num_channels, -1)
        feat_far = feat_flat.gather(2, idx_far).mean(dim=2)
        s_pos = feat_far * f0.unsqueeze(0)
        _, i_pos = torch.topk(s_pos, k=k, dim=1, largest=True)

        idx_near = near_indices.expand(-1, num_channels, -1)
        feat_near = feat_flat.gather(2, idx_near).mean(dim=2)
        s_neg = feat_near * f0.unsqueeze(0)
        _, i_neg = torch.topk(s_neg, k=k, dim=1, largest=True)

        return i_pos, i_neg

    def backtrack_step(
        self, i_pos_prev: torch.Tensor, i_neg_prev: torch.Tensor,
        weight: torch.Tensor, k_curr: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = i_pos_prev.shape[0]
        c_out, c_in = weight.shape
        k_curr = max(1, min(k_curr, c_in))

        w = weight.detach().to(i_pos_prev.device)
        w_expanded = w.unsqueeze(0).expand(batch_size, -1, -1)

        idx_pos = i_pos_prev.unsqueeze(2).expand(-1, -1, c_in)
        sub_pos = w_expanded.gather(1, idx_pos)
        k_bar = sub_pos.amax(dim=1)
        _, i_pos_curr = torch.topk(k_bar, k=k_curr, dim=1, largest=True)

        idx_neg = i_neg_prev.unsqueeze(2).expand(-1, -1, c_in)
        sub_neg = w_expanded.gather(1, idx_neg)
        k_under = sub_neg.amin(dim=1)
        _, i_neg_curr = torch.topk(k_under, k=k_curr, dim=1, largest=False)

        return i_pos_curr, i_neg_curr

    def backtrack(
        self, initial_pos: torch.Tensor, initial_neg: torch.Tensor,
        topk_frac: float = 0.20
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        selections = {}
        selections["dec_up5"] = (initial_pos, initial_neg)
        i_pos, i_neg = initial_pos, initial_neg

        for name, weight in self.chain[1:]:
            c_in = weight.shape[1]
            k_curr = max(1, round(topk_frac * c_in))
            i_pos, i_neg = self.backtrack_step(i_pos, i_neg, weight, k_curr)
            selections[name] = (i_pos, i_neg)

        return selections
