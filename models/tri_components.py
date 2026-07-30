"""Reusable components for the combined MedTsLLM model.

The implementation deliberately avoids importing LAVIS.  The Q-Former below
uses the same core idea as BLIP-2: a small set of learned query tokens attends
into a frozen/pretrained encoder's token memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict-like or attribute-style config object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    value = getattr(cfg, key, default)
    return value


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(dim, int(dim * expansion))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class QFormerLayer(nn.Module):
    """One learned-query transformer block with self- and cross-attention."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn = FeedForward(dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        q = self.self_norm(queries)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self.dropout(self_out)

        q = self.cross_norm(queries)
        cross_out, attn = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = queries + self.dropout(cross_out)
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries, attn


class LearnedQueryFormer(nn.Module):
    """Compact BLIP-2-style query transformer implemented in pure PyTorch."""

    def __init__(
        self,
        dim: int,
        num_queries: int = 32,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"q_dim={dim} must be divisible by q_heads={heads}.")
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.layers = nn.ModuleList(
            [QFormerLayer(dim, heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        memory: Tensor,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        batch = memory.shape[0]
        queries = self.query_tokens.expand(batch, -1, -1)
        attention_maps: list[Tensor] = []
        for layer in self.layers:
            queries, attn = layer(queries, memory, memory_padding_mask)
            attention_maps.append(attn)
        return self.norm(queries), attention_maps


class GatedResidualFusion(nn.Module):
    """Fuse two aligned token streams while retaining the MedTsLLM branch."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, med_tokens: Tensor, moment_tokens: Tensor) -> tuple[Tensor, Tensor]:
        if med_tokens.shape != moment_tokens.shape:
            raise ValueError(
                "Fusion inputs must have identical shapes; got "
                f"{tuple(med_tokens.shape)} and {tuple(moment_tokens.shape)}."
            )
        joined = torch.cat([med_tokens, moment_tokens], dim=-1)
        gate = self.gate(joined)
        fused = med_tokens + gate * self.delta(joined)
        return self.norm(fused), gate


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        scores = self.score(self.norm(tokens)).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return torch.einsum("bn,bnd->bd", weights, tokens)


class LocalMomentFallback(nn.Module):
    """Small patch encoder used only for smoke tests or explicit fallback mode."""

    def __init__(self, input_channels: int, token_dim: int, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, token_dim, patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv1d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x_cf: Tensor) -> Tensor:
        if x_cf.shape[-1] < self.patch_size:
            x_cf = F.pad(x_cf, (0, self.patch_size - x_cf.shape[-1]))
        return self.encoder(x_cf).transpose(1, 2)


@dataclass
class MomentOutput:
    tokens: Tensor
    raw: Any = None


class MomentTokenEncoder(nn.Module):
    """Adapter around MOMENT with defensive output-shape normalization.

    Input:  x_cf [batch, channels, time]
    Output: tokens [batch, token_count, token_dim]
    """

    def __init__(
        self,
        input_channels: int,
        token_dim: int,
        model_id: str = "AutonLab/MOMENT-1-base",
        backend: str = "moment",
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
        allow_local_fallback: bool = False,
        local_patch_size: int = 16,
        max_leads: int = 64,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.token_dim = token_dim
        self.backend = backend.lower()
        self.freeze_backbone = freeze_backbone
        self.allow_local_fallback = allow_local_fallback
        self.model_id = model_id
        self.unfreeze_last_n = int(unfreeze_last_n)
        self.backbone: Optional[nn.Module] = None
        self.local = LocalMomentFallback(input_channels, token_dim, local_patch_size)
        self.proj = nn.LazyLinear(token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.lead_embeddings = nn.Parameter(torch.empty(max_leads, token_dim))
        self.lead_score = nn.Linear(token_dim, 1)
        nn.init.trunc_normal_(self.lead_embeddings, std=0.02)

        if self.backend == "moment":
            self._load_moment(trust_remote_code)
        elif self.backend != "local":
            raise ValueError("moment.backend must be 'moment' or 'local'.")

    @property
    def using_real_moment(self) -> bool:
        return self.backbone is not None

    @staticmethod
    def _transformer_blocks(module: nn.Module) -> list[nn.Module]:
        candidates = (
            ("block",),
            ("layers",),
            ("encoder", "block"),
            ("encoder", "layers"),
        )
        for path in candidates:
            current: Any = module
            for name in path:
                if not hasattr(current, name):
                    break
                current = getattr(current, name)
            else:
                if isinstance(current, (nn.ModuleList, list, tuple)):
                    return list(current)
        return []

    def _configure_trainability(self) -> None:
        if self.backbone is None:
            return
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            return
        self.backbone.requires_grad_(True)
        head = getattr(self.backbone, "head", None)
        if isinstance(head, nn.Module):
            head.requires_grad_(False)
        if self.unfreeze_last_n > 0:
            self.backbone.requires_grad_(False)
            encoder = getattr(self.backbone, "encoder", self.backbone)
            blocks = self._transformer_blocks(encoder)
            if not blocks:
                raise RuntimeError(
                    "Could not locate MOMENT transformer blocks for partial "
                    "unfreezing. Set unfreeze_last_n=0 or freeze=true."
                )
            for block in blocks[-self.unfreeze_last_n :]:
                block.requires_grad_(True)

    @property
    def has_trainable_backbone(self) -> bool:
        return self.backbone is not None and any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )

    def _load_moment(self, trust_remote_code: bool) -> None:
        try:
            from momentfm import MOMENTPipeline  # type: ignore

            model_kwargs = {
                "task_name": "embedding",
                "freeze_encoder": self.freeze_backbone or self.unfreeze_last_n > 0,
                "freeze_embedder": self.freeze_backbone or self.unfreeze_last_n > 0,
                "enable_gradient_checkpointing": not self.freeze_backbone,
            }
            try:
                self.backbone = MOMENTPipeline.from_pretrained(
                    self.model_id,
                    model_kwargs=model_kwargs,
                    trust_remote_code=trust_remote_code,
                )
            except TypeError:
                # Older momentfm releases do not expose trust_remote_code.
                self.backbone = MOMENTPipeline.from_pretrained(
                    self.model_id,
                    model_kwargs=model_kwargs,
                )
            if hasattr(self.backbone, "init"):
                self.backbone.init()
            backbone_config = getattr(self.backbone, "config", None)
            self.expected_length = int(
                getattr(backbone_config, "seq_len", 0) or 0
            )
            backbone_dim = int(getattr(backbone_config, "d_model", 0) or 0)
            if backbone_dim > 0:
                self.proj = nn.Linear(backbone_dim, self.token_dim)
            self._configure_trainability()
        except Exception as exc:
            if not self.allow_local_fallback:
                raise RuntimeError(
                    "Could not initialize MOMENT. Install momentfm and make sure the "
                    f"checkpoint '{self.model_id}' is accessible. Set "
                    "allow_local_fallback=true only for smoke testing."
                ) from exc
            self.backbone = None

    @staticmethod
    def _pick_tensor(output: Any) -> Tensor:
        if torch.is_tensor(output):
            return output
        keys = (
            "encoder_embeddings",
            "patch_embeddings",
            "last_hidden_state",
            "embeddings",
            "features",
            "representation",
        )
        if isinstance(output, dict):
            for key in keys:
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            hidden = output.get("hidden_states")
            if isinstance(hidden, (list, tuple)) and hidden and torch.is_tensor(hidden[-1]):
                return hidden[-1]
        for key in keys:
            value = getattr(output, key, None)
            if torch.is_tensor(value):
                return value
        hidden = getattr(output, "hidden_states", None)
        if isinstance(hidden, (list, tuple)) and hidden and torch.is_tensor(hidden[-1]):
            return hidden[-1]
        if isinstance(output, (list, tuple)):
            for value in output:
                if torch.is_tensor(value) and value.ndim >= 2:
                    return value
        raise RuntimeError(
            "MOMENT returned an unsupported output object. Expected a tensor or an "
            "object containing embeddings/encoder_embeddings/last_hidden_state."
        )

    def _call_backbone(self, x_cf: Tensor, input_mask: Tensor) -> Any:
        assert self.backbone is not None
        calls = (
            lambda: self.backbone(
                x_enc=x_cf, input_mask=input_mask, reduction="none"
            ),
            lambda: self.backbone(x_enc=x_cf, input_mask=input_mask),
            lambda: self.backbone(x_enc=x_cf),
            lambda: self.backbone(x_cf, input_mask=input_mask),
            lambda: self.backbone(x_cf),
        )
        last_error: Optional[Exception] = None
        for call in calls:
            try:
                return call()
            except TypeError as exc:
                last_error = exc
        raise RuntimeError("Could not call the installed MOMENT model API.") from last_error

    def _normalize_shape(self, raw: Tensor, batch: int, channels: int) -> Tensor:
        # Standardize batch dimension first.
        if raw.shape[0] != batch and raw.ndim >= 3 and raw.shape[1] == batch:
            raw = raw.transpose(0, 1)
        if raw.shape[0] != batch:
            raise RuntimeError(
                f"MOMENT output batch dimension is {raw.shape[0]}, expected {batch}."
            )

        if raw.ndim == 2:  # [B, D]
            raw = raw.unsqueeze(1)
        elif raw.ndim == 3:
            # [B, D, N] occasionally appears. Keep [B, N, D] when recognizable.
            if raw.shape[-1] < raw.shape[1] and raw.shape[1] > 128:
                raw = raw.transpose(1, 2)
        elif raw.ndim == 4:
            # Supported forms: [B,C,N,D], [B,N,C,D], or [B,C,D,N].
            if raw.shape[1] == channels:
                lead_tokens = raw
            elif raw.shape[2] == channels:
                lead_tokens = raw.transpose(1, 2)
            else:
                # Flatten all token-like axes except feature dimension.
                raw = raw.flatten(1, -2)
                return raw
            if lead_tokens.shape[-1] < 16 and lead_tokens.shape[-2] > 32:
                lead_tokens = lead_tokens.transpose(-1, -2)
            lead_tokens = self.proj(lead_tokens)
            n_leads = lead_tokens.shape[1]
            lead_pos = self.lead_embeddings[:n_leads].view(1, n_leads, 1, -1)
            scores = self.lead_score(torch.tanh(lead_tokens + lead_pos)).squeeze(-1)
            weights = scores.softmax(dim=1)
            return torch.einsum("bcn,bcnd->bnd", weights, lead_tokens)
        else:
            raise RuntimeError(f"Unsupported MOMENT tensor rank: {raw.ndim}.")
        return self.proj(raw)

    def forward(self, x_cf: Tensor, input_mask: Optional[Tensor] = None) -> MomentOutput:
        batch, channels, length = x_cf.shape
        expected = int(getattr(self, "expected_length", 0) or 0)
        if expected > 0 and length != expected:
            x_cf = F.interpolate(x_cf, size=expected, mode="linear", align_corners=False)
            length = expected
        if input_mask is None or input_mask.shape[-1] != length:
            input_mask = torch.ones(
                batch, length, device=x_cf.device, dtype=x_cf.dtype
            )

        if self.backbone is None:
            tokens = self.local(x_cf)
            return MomentOutput(tokens=self.norm(tokens), raw=None)

        grad_enabled = self.training and not self.freeze_backbone
        with torch.set_grad_enabled(grad_enabled):
            output = self._call_backbone(x_cf, input_mask)
            raw = self._pick_tensor(output)
        tokens = self._normalize_shape(raw, batch, channels)
        tokens = self.norm(tokens)
        return MomentOutput(tokens=tokens, raw=output)

    def train(self, mode: bool = True) -> "MomentTokenEncoder":
        super().train(mode)
        if self.backbone is not None and not self.has_trainable_backbone:
            self.backbone.eval()
        return self


def align_token_count(source: Tensor, target_count: int) -> Tensor:
    """Interpolate token sequence to a target length."""
    if source.shape[1] == target_count:
        return source
    return F.interpolate(
        source.transpose(1, 2), size=target_count, mode="linear", align_corners=False
    ).transpose(1, 2)


def query_diversity_loss(queries: Tensor) -> Tensor:
    """Penalize duplicate learned-query outputs without forcing orthogonality."""
    q = F.normalize(queries, dim=-1)
    sim = torch.bmm(q, q.transpose(1, 2))
    n = sim.shape[-1]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye)
    return off_diag.square().mean() if off_diag.numel() else sim.new_zeros(())


def supervised_contrastive_alignment(
    first: Tensor,
    second: Tensor,
    labels: Optional[Tensor],
    temperature: float = 0.1,
) -> Tensor:
    """Symmetric cross-branch contrastive alignment.

    With labels, samples sharing the same class are positives. Without labels,
    only paired views of the same sample are positives.
    """
    if first.shape[0] < 2:
        return first.new_zeros(())
    z1 = F.normalize(first, dim=-1)
    z2 = F.normalize(second, dim=-1)
    logits = z1 @ z2.transpose(0, 1) / temperature
    batch = first.shape[0]
    if labels is None:
        targets = torch.arange(batch, device=first.device)
        return 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
        )

    labels = labels.view(-1)
    positive = labels[:, None].eq(labels[None, :]).to(logits.dtype)
    # Normalize each row so multiple same-class positives contribute equally.
    positive = positive / positive.sum(dim=1, keepdim=True).clamp_min(1.0)
    loss_12 = -(positive * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_21 = -(positive.t() * F.log_softmax(logits.t(), dim=1)).sum(dim=1).mean()
    return 0.5 * (loss_12 + loss_21)
