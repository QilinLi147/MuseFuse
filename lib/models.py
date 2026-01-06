from typing import Dict

import torch
import torch.nn as nn


class MuseFuseModel(nn.Module):
    """MuseFuse: Multi-View Feature Fusion for Music Emotion Recognition.

    Integrates three core modules as described in the MuseFuse paper:

    - ProtoAlign: Prototype-guided semantic alignment with EMA-updated prototype bank
      (momentum = 0.9, see Eq. (L_proto)).
    - ReliaPseudo: Reliability-guided pseudo-labelling with entropy-based weighting
      and curriculum thresholding (0.6 → 0.3, Eq. (L_pseudo)).
    - TriDistill: Symmetric tri-branch KL distillation across fusion–Mel–cochleagram
      branches (temperature T = 2.0, Eqs. (L_FM, L_MF, L_FC, L_CF)).

    Architecture:
    - Dual-view encoders for Mel spectrograms and cochleagrams.
    - Adaptive gating network for sample-specific fusion of Mel/Cocheragram features.
    - Three classification heads for Mel / Cochleagram / Fusion branches.
    """

    def __init__(
        self,
        mel_dim: int,
        coch_dim: int,
        hidden_mel: int,
        hidden_coch: int,
        fusion_hidden: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        # View-specific encoders
        self.mel_encoder = nn.Sequential(
            nn.Linear(mel_dim, hidden_mel),
            nn.BatchNorm1d(hidden_mel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_mel, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
        )
        self.coch_encoder = nn.Sequential(
            nn.Linear(coch_dim, hidden_coch),
            nn.BatchNorm1d(hidden_coch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_coch, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
        )

        # Adaptive gating for sample-specific fusion
        self.gating = nn.Sequential(
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.Sigmoid(),
        )

        # Classification heads
        self.mel_head = nn.Linear(fusion_hidden, num_classes)
        self.coch_head = nn.Linear(fusion_hidden, num_classes)
        self.fusion_head = nn.Linear(fusion_hidden, num_classes)

        # Prototype bank for ProtoAlign (Eq. L_proto)
        self.prototypes = nn.Parameter(torch.randn(num_classes, fusion_hidden))

    def forward(self, mel: torch.Tensor, coch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through dual encoders, gating fusion and heads."""
        batch_size = mel.size(0)
        mel_flat = mel.view(batch_size, -1)
        coch_flat = coch.view(batch_size, -1)

        mel_feat = self.mel_encoder(mel_flat)
        coch_feat = self.coch_encoder(coch_flat)

        gate = self.gating(torch.cat([mel_feat, coch_feat], dim=-1))
        fusion_feat = gate * mel_feat + (1.0 - gate) * coch_feat

        mel_logits = self.mel_head(mel_feat)
        coch_logits = self.coch_head(coch_feat)
        fusion_logits = self.fusion_head(fusion_feat)

        return {
            "mel_feat": mel_feat,
            "coch_feat": coch_feat,
            "fusion_feat": fusion_feat,
            "mel_logits": mel_logits,
            "coch_logits": coch_logits,
            "fusion_logits": fusion_logits,
        }

