"""
unified_model.py
================
Core architecture for UnifiedTMIL: a single-pass neural model that jointly
performs account-level phishing classification and transaction-level localization
on Ethereum transaction sequences.

Architecture overview:
    - Counterparty embedding (learned, BERT4ETH-style)
    - I/O direction embedding
    - Numerical feature projection (log-amount, log-delta-t) via MLP + LayerNorm
    - 1D Temporal Convolutional Network (TCN) with residual connection
    - Gated Attention (Ilse et al., 2018) with optional engineered feature injection
    - Linear account classifier on the attention-weighted context vector
    - Attention weights used directly as transaction-level ranking scores

Reference:
    Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple
    instance learning. ICML 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedTMIL(nn.Module):
    """Unified Transaction-level Multiple Instance Learning model.

    A single-weight model that solves two tasks in one forward pass:
        1. Account-level binary classification (phishing vs. benign).
        2. Transaction-level localization via attention-as-ranking.

    Args:
        vocab_size (int): Number of unique counterparty addresses (vocabulary size).
        embed_dim (int): Dimensionality of all learned embeddings. Default: 64.
        hc_dim (int): Dimensionality of the numerical (hand-crafted) feature vector
            per transaction (log-amount, log-delta-t). Default: 2.
        extra_feat_dim (int): Dimensionality of additional engineered features
            (e.g., counterparty reputation, transaction degree) injected into the
            attention layer. Default: 0.
        use_tcn (bool): Whether to apply the 1D-TCN temporal context layer.
            Default: True.
        direction_mode (str): How to handle transaction direction (IN/OUT).
            ``"io_embed"`` adds a learned direction embedding to the transaction
            representation; ``"hardmask"`` restricts attention to outbound
            transactions only. Default: ``"io_embed"``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hc_dim: int = 2,
        extra_feat_dim: int = 0,
        use_tcn: bool = True,
        direction_mode: str = "io_embed",
    ) -> None:
        super().__init__()

        self.direction_mode = direction_mode
        self.use_tcn = use_tcn

        # ── Embedding layers ──────────────────────────────────────────────
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)

        # ── Numerical feature projection ──────────────────────────────────
        self.hc_proj = nn.Sequential(
            nn.Linear(hc_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # ── Temporal Convolutional Network (optional) ─────────────────────
        if use_tcn:
            self.tcn = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm(embed_dim)

        # ── Gated Attention (Ilse et al., 2018) ───────────────────────────
        attn_input_dim = embed_dim + extra_feat_dim
        self.attn_V = nn.Linear(attn_input_dim, 128)
        self.attn_U = nn.Linear(attn_input_dim, 128)
        self.attn_w = nn.Linear(128, 1)

        # ── Account-level classifier ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ids: torch.Tensor,
        io: torch.Tensor,
        hc: torch.Tensor,
        mask: torch.Tensor,
        extra_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            ids (torch.Tensor): Counterparty address token IDs, shape
                ``(batch_size, seq_len)``, dtype ``torch.long``.
            io (torch.Tensor): Transaction direction indicators (0=PAD, 1=OUT,
                2=IN), shape ``(batch_size, seq_len)``, dtype ``torch.long``.
            hc (torch.Tensor): Numerical features (log-amount, log-delta-t),
                shape ``(batch_size, seq_len, hc_dim)``, dtype ``torch.float32``.
            mask (torch.Tensor): Boolean padding mask, shape
                ``(batch_size, seq_len)``, ``True`` for valid positions.
            extra_feats (torch.Tensor | None): Optional engineered features
                injected before the attention layer, shape
                ``(batch_size, seq_len, extra_feat_dim)``. Default: ``None``.

        Returns:
            account_logit (torch.Tensor): Scalar logit per account, shape
                ``(batch_size,)``.
            attention_weights (torch.Tensor): Soft attention distribution over
                transactions (used as ranking scores), shape
                ``(batch_size, seq_len)``.
        """
        # ── Build transaction representations ─────────────────────────────
        h = self.cp_embed(ids) + self.hc_proj(hc)
        if self.direction_mode == "io_embed":
            h = h + self.io_embed(io)
        h = self.norm(h)

        # ── TCN temporal context ──────────────────────────────────────────
        if self.use_tcn:
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)

        # ── Feature injection for attention ───────────────────────────────
        attn_input = h
        if extra_feats is not None:
            attn_input = torch.cat([h, extra_feats], dim=-1)

        # ── Gated attention scores ────────────────────────────────────────
        s = self.attn_w(
            torch.tanh(self.attn_V(attn_input)) * torch.sigmoid(self.attn_U(attn_input))
        ).squeeze(-1)

        # Mask padding positions
        s = s.masked_fill(~mask, -1e9)

        # Hard outbound mask (direction_mode == "hardmask")
        if self.direction_mode == "hardmask":
            outb = io == 1
            has_out = outb.any(dim=1, keepdim=True)
            s = s.masked_fill(has_out & ~outb & mask, -1e9)

        # ── Softmax → attention weights ───────────────────────────────────
        a = F.softmax(s, dim=1)

        # ── Attention-weighted context vector ─────────────────────────────
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)

        # ── Account-level classification ──────────────────────────────────
        account_logit = self.classifier(z).squeeze(-1)

        return account_logit, a


def entropy_loss(attention_weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy regularization loss to encourage peaked attention distributions.

    Minimizing entropy pushes the model to concentrate attention on a small
    number of transactions, which is desirable for transaction localization.

    Args:
        attention_weights (torch.Tensor): Attention distribution, shape
            ``(batch_size, seq_len)``.
        mask (torch.Tensor): Boolean padding mask (unused here; kept for API
            consistency), shape ``(batch_size, seq_len)``.

    Returns:
        torch.Tensor: Scalar mean entropy over the batch.
    """
    eps = 1e-8
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=1)
    return entropy.mean()
