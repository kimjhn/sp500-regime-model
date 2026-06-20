"""Two-Track GRU + MLP with attention late-fusion (proposal architecture).

Track 1 (GRU) encodes the 60-day price/technical sequence; Track 2 (MLP) encodes
the macro snapshot. The two modality embeddings are fused by additive attention
(a learned query scores the two modality "tokens", softmax-weighted sum), which
keeps the proposal's "Attention Fusion (Late Fusion)" while exposing per-modality
weights for interpretability.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class RegimePredictor(nn.Module):
    def __init__(self, t1_features=len(C.TRACK1_FEATS), t2_features=len(C.TRACK2_FEATS),
                 n_regimes=C.N_REGIMES, gru_hidden=64, gru_layers=2,
                 fusion_dim=32, dropout=0.3, gru_dropout=0.2):
        super().__init__()
        # Track 1: GRU over the sequence -> last-layer hidden -> fusion_dim
        self.gru = nn.GRU(t1_features, gru_hidden, gru_layers,
                          batch_first=True,
                          dropout=gru_dropout if gru_layers > 1 else 0.0)
        self.t1_proj = nn.Linear(gru_hidden, fusion_dim)

        # Track 2: MLP on the macro snapshot -> fusion_dim
        self.mlp = nn.Sequential(
            nn.Linear(t2_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.t2_proj = nn.Linear(16, fusion_dim)

        # Attention fusion: a learned query scores the 2 modality tokens
        self.attn_query = nn.Parameter(torch.randn(fusion_dim) * 0.1)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_regimes),
        )
        self.fusion_dim = fusion_dim

    def forward(self, x1, x2, return_attn=False):
        _, h_n = self.gru(x1)            # h_n: (layers, B, hidden)
        t1 = self.t1_proj(h_n[-1])       # (B, fusion_dim)
        t2 = self.t2_proj(self.mlp(x2))  # (B, fusion_dim)

        tokens = torch.stack([t1, t2], dim=1)             # (B, 2, d)
        scores = tokens @ self.attn_query / (self.fusion_dim ** 0.5)  # (B, 2)
        attn = F.softmax(scores, dim=1)                   # (B, 2): [price, macro]
        fused = (attn.unsqueeze(-1) * tokens).sum(dim=1)  # (B, d)

        logits = self.head(fused)        # (B, n_regimes)
        if return_attn:
            return logits, attn
        return logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = RegimePredictor()
    x1 = torch.randn(8, C.SEQ_LEN, len(C.TRACK1_FEATS))
    x2 = torch.randn(8, len(C.TRACK2_FEATS))
    logits, attn = m(x1, x2, return_attn=True)
    print("logits", tuple(logits.shape), "attn", tuple(attn.shape),
          "params", count_params(m))
    assert logits.shape == (8, C.N_REGIMES)
    assert torch.allclose(attn.sum(1), torch.ones(8), atol=1e-5)
    print("OK")
