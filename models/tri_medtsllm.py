"""Combined BioMedCoOp + MOMENT + BLIP-2/Q-Former model for MedTsLLM.

Copy this file and ``tri_components.py`` into ``medtsllm4/models``.  The class
inherits the existing MedTsLLM implementation, so dataset loading, RevIN,
patching, reprogramming, prompt construction, LLM setup, and BioMedCoOp text
prototype construction continue to use the baseline repository's code.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .medtsllm import MedTsLLM
from .tri_components import (
    AttentionPool,
    GatedResidualFusion,
    LearnedQueryFormer,
    MomentTokenEncoder,
    align_token_count,
    cfg_get,
    query_diversity_loss,
    supervised_contrastive_alignment,
)


class TriMedTsLLM(MedTsLLM):
    """Unified sequence classifier using all three requested enhancements.

    Data path::

        MedTsLLM tokens -----\\
                              gated fusion -> Q-Former -> LLM -> BioMedCoOp
        MOMENT tokens -------/
    """

    supported_tasks = ["classification"]
    supported_modes = ["multivariate"]

    def __init__(self, config: Any, dataset: Any) -> None:
        if config.task != "classification":
            raise ValueError("TriMedTsLLM currently supports classification only.")
        super().__init__(config, dataset)

        combined = cfg_get(self.model_config, "combined", None)
        if combined is None:
            raise ValueError(
                "Missing [models.medtsllm.combined] in the TOML configuration."
            )
        if not self.use_biomedcoop or not hasattr(self, "bc_head"):
            raise ValueError(
                "TriMedTsLLM requires the BioMedCoOp head already implemented in "
                "medtsllm4. Enable [models.medtsllm.biomedcoop].enabled."
            )
        if not self.llm_enabled:
            raise ValueError("TriMedTsLLM requires models.medtsllm.llm.enabled=true.")

        q_dim = int(cfg_get(combined, "q_dim", 512))
        q_queries = int(cfg_get(combined, "q_queries", 32))
        q_depth = int(cfg_get(combined, "q_depth", 4))
        q_heads = int(cfg_get(combined, "q_heads", 8))
        dropout = float(cfg_get(combined, "dropout", 0.1))

        moment_cfg = cfg_get(combined, "moment", {})
        self.use_highres_windows = bool(
            cfg_get(moment_cfg, "use_highres_windows", True)
        )
        self.save_frozen_moment = bool(
            cfg_get(moment_cfg, "save_frozen_backbone", False)
        )
        self.moment_encoder = MomentTokenEncoder(
            input_channels=int(self.n_features),
            token_dim=q_dim,
            model_id=str(cfg_get(moment_cfg, "model_id", "AutonLab/MOMENT-1-base")),
            backend=str(cfg_get(moment_cfg, "backend", "moment")),
            freeze_backbone=bool(cfg_get(moment_cfg, "freeze", True)),
            trust_remote_code=bool(cfg_get(moment_cfg, "trust_remote_code", True)),
            allow_local_fallback=bool(
                cfg_get(moment_cfg, "allow_local_fallback", False)
            ),
            local_patch_size=int(cfg_get(moment_cfg, "local_patch_size", 16)),
            max_leads=int(cfg_get(moment_cfg, "max_leads", 64)),
            unfreeze_last_n=int(cfg_get(moment_cfg, "unfreeze_last_n", 0)),
        )

        self.med_to_q = nn.Sequential(nn.Linear(self.d_llm, q_dim), nn.LayerNorm(q_dim))
        self.fusion = GatedResidualFusion(q_dim, dropout)
        self.window_token_pool = AttentionPool(q_dim)
        self.window_pool = AttentionPool(q_dim)
        self.context_gate = nn.Linear(q_dim * 2, q_dim)
        self.context_norm = nn.LayerNorm(q_dim)

        self.qformer = LearnedQueryFormer(
            q_dim, q_queries, q_depth, q_heads, dropout
        )
        self.q_to_llm = nn.Sequential(
            nn.Linear(q_dim, self.d_llm), nn.LayerNorm(self.d_llm)
        )
        self.med_pool = AttentionPool(q_dim)
        self.moment_pool = AttentionPool(q_dim)
        self.query_pool = AttentionPool(q_dim)
        self.llm_pool = AttentionPool(self.d_llm)

        output_dim = self.n_classes if self.n_classes > 2 else 1
        self.med_aux_head = nn.Linear(q_dim, output_dim)
        self.moment_aux_head = nn.Linear(q_dim, output_dim)
        self.query_aux_head = nn.Linear(q_dim, output_dim)

        loss_cfg = cfg_get(combined, "loss", {})
        self.loss_weights = {
            "med_ce": float(cfg_get(loss_cfg, "med_ce", 0.15)),
            "moment_ce": float(cfg_get(loss_cfg, "moment_ce", 0.15)),
            "query_ce": float(cfg_get(loss_cfg, "query_ce", 0.10)),
            "alignment": float(cfg_get(loss_cfg, "alignment", 0.10)),
            "query_diversity": float(cfg_get(loss_cfg, "query_diversity", 0.01)),
            "biomedcoop": float(cfg_get(loss_cfg, "biomedcoop", 1.0)),
        }
        self.alignment_temperature = float(
            cfg_get(loss_cfg, "alignment_temperature", 0.1)
        )
        self._auxiliary_losses: dict[str, Tensor] = {}
        self.aux_loss: Optional[Tensor] = None
        self.last_fusion_gate: Optional[Tensor] = None
        self.last_cross_attention: Optional[list[Tensor]] = None

    def _restore_batch(self, tokens: Tensor, batch_size: int) -> Tensor:
        """Collapse optional independent-channel batch expansion."""
        if tokens.shape[0] == batch_size:
            return tokens
        if tokens.shape[0] % batch_size != 0:
            raise RuntimeError(
                "MedTsLLM token batch cannot be mapped to input samples: "
                f"tokens={tokens.shape[0]}, samples={batch_size}."
            )
        channels = tokens.shape[0] // batch_size
        return tokens.view(batch_size, channels, tokens.shape[1], tokens.shape[2]).mean(1)

    def _encode_prompts(self, inputs: dict[str, Any], dtype: torch.dtype) -> Tensor:
        """Use MedTsLLM's existing heterogeneous prompt-part API."""
        x_enc = inputs["x_enc"]
        batch_size = x_enc.size(0)
        prompts = self.build_prompt(inputs)
        if not prompts or not prompts[0]:
            return torch.zeros(
                batch_size, 0, self.d_llm, device=x_enc.device, dtype=dtype
            )
        encoded = [[self.encode_part(part) for part in prompt] for prompt in prompts]
        encoded = [torch.cat(parts, dim=1) for parts in encoded]
        max_len = max(item.size(1) for item in encoded)
        encoded = [self.pad_sequence(item, max_len) for item in encoded]
        return torch.cat(encoded, dim=0).to(device=x_enc.device, dtype=dtype)

    def _encode_moment(
        self, x_enc: Tensor, x_moment_windows: Optional[Tensor]
    ) -> Tensor:
        """Encode the main window and optional high-resolution context windows."""
        batch, _, channels = x_enc.shape
        main_cf = x_enc.transpose(1, 2).contiguous()
        window_count = 0
        combined_cf = main_cf
        if self.use_highres_windows and x_moment_windows is not None:
            if x_moment_windows.ndim != 4:
                raise ValueError(
                    "x_moment_windows must be [B,W,T,C], got "
                    f"{tuple(x_moment_windows.shape)}."
                )
            if x_moment_windows.shape[0] != batch or x_moment_windows.shape[-1] != channels:
                raise ValueError("x_moment_windows batch/channel dimensions do not match x_enc.")
            window_count = x_moment_windows.shape[1]
            windows_cf = (
                x_moment_windows.reshape(-1, x_moment_windows.shape[2], channels)
                .transpose(1, 2)
                .contiguous()
            )
            # MOMENT's adapter performs the final sequence-length interpolation.
            if windows_cf.shape[-1] != main_cf.shape[-1]:
                windows_cf = F.interpolate(
                    windows_cf, size=main_cf.shape[-1], mode="linear", align_corners=False
                )
            combined_cf = torch.cat([main_cf, windows_cf], dim=0)

        all_tokens = self.moment_encoder(combined_cf).tokens
        main_tokens = all_tokens[:batch]
        if window_count == 0:
            return main_tokens

        window_tokens = all_tokens[batch:]
        window_summaries = self.window_token_pool(window_tokens)
        window_summaries = window_summaries.view(batch, window_count, -1)
        global_context = self.window_pool(window_summaries)
        context = global_context.unsqueeze(1).expand(-1, main_tokens.shape[1], -1)
        gate = torch.sigmoid(self.context_gate(torch.cat([main_tokens, context], dim=-1)))
        return self.context_norm(main_tokens + gate * context)

    def _run_llm(self, prompt_tokens: Tensor, soft_queries: Tensor) -> Tensor:
        soft_queries = soft_queries.to(dtype=prompt_tokens.dtype)
        if self.llm.config.is_encoder_decoder:
            output = self.llm(
                inputs_embeds=prompt_tokens,
                decoder_inputs_embeds=soft_queries,
            ).last_hidden_state
            return output[:, -soft_queries.size(1) :].to(soft_queries.dtype)
        llm_input = torch.cat([prompt_tokens, soft_queries], dim=1)
        output = self.llm(inputs_embeds=llm_input).last_hidden_state
        return output[:, -soft_queries.size(1) :].to(soft_queries.dtype)

    def _branch_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.n_classes > 2:
            return F.cross_entropy(logits, labels.long())
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), labels.to(logits.dtype)
        )

    def _set_auxiliary_losses(
        self,
        med_repr: Tensor,
        moment_repr: Tensor,
        query_repr: Tensor,
        queries: Tensor,
        labels: Optional[Tensor],
    ) -> None:
        zero = query_repr.new_zeros(())
        bc_aux = getattr(self.bc_head, "aux_loss", None)
        if bc_aux is None:
            bc_aux = zero
        raw: dict[str, Tensor] = {
            "med_ce": zero,
            "moment_ce": zero,
            "query_ce": zero,
            "alignment": supervised_contrastive_alignment(
                med_repr, moment_repr, labels, self.alignment_temperature
            ),
            "query_diversity": query_diversity_loss(queries),
            "biomedcoop": bc_aux,
        }
        if labels is not None:
            raw["med_ce"] = self._branch_loss(self.med_aux_head(med_repr), labels)
            raw["moment_ce"] = self._branch_loss(
                self.moment_aux_head(moment_repr), labels
            )
            raw["query_ce"] = self._branch_loss(self.query_aux_head(query_repr), labels)
        weighted = {name: value * self.loss_weights[name] for name, value in raw.items()}
        weighted["total"] = torch.stack(tuple(weighted.values())).sum()
        self._auxiliary_losses = weighted
        self.aux_loss = weighted["total"]

    def forward(self, inputs: dict[str, Any]) -> Tensor:
        x_enc: Tensor = inputs["x_enc"]
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)
        if x_enc.ndim != 3:
            raise ValueError(f"x_enc must be [B,T,C], got {tuple(x_enc.shape)}.")
        if x_enc.size(-1) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} channels, got {x_enc.size(-1)}."
            )
        if self.device is None:
            self.device = x_enc.device

        batch = x_enc.shape[0]
        med_tokens = self._restore_batch(self.encode_ts(x_enc), batch)
        med_tokens = self.med_to_q(med_tokens)
        moment_tokens = self._encode_moment(
            x_enc, inputs.get("x_moment_windows")
        )
        moment_tokens = align_token_count(moment_tokens, med_tokens.shape[1])
        fused_tokens, gate = self.fusion(med_tokens, moment_tokens)

        queries, cross_attention = self.qformer(fused_tokens)
        soft_queries = self.q_to_llm(queries)
        prompt_tokens = self._encode_prompts(inputs, dtype=soft_queries.dtype)
        llm_query_tokens = self._run_llm(prompt_tokens, soft_queries)
        sample_repr = self.llm_pool(llm_query_tokens)

        if self._bc_prototypes is None:
            self._build_class_prototypes()
        prototypes = self._bc_prototypes.to(sample_repr.device)
        labels = inputs.get("labels") if self.training else None
        proto_logits = self.bc_head(sample_repr, prototypes, labels=labels)
        # medtsllm4's binary task uses BCEWithLogitsLoss and expects one score.
        logits = (
            proto_logits
            if self.n_classes > 2
            else proto_logits[:, 1] - proto_logits[:, 0]
        )

        med_repr = self.med_pool(med_tokens)
        moment_repr = self.moment_pool(moment_tokens)
        query_repr = self.query_pool(queries)
        self._set_auxiliary_losses(
            med_repr, moment_repr, query_repr, queries, labels
        )
        self.last_fusion_gate = gate.detach()
        self.last_cross_attention = [item.detach() for item in cross_attention]
        return logits

    def predict(self, inputs: dict[str, Any]) -> Tensor:
        return self.forward(inputs)

    def get_auxiliary_losses(self) -> dict[str, Tensor]:
        return self._auxiliary_losses

    def train(self, mode: bool = True) -> "TriMedTsLLM":
        super().train(mode)
        if self.llm_enabled and not self.lora_enabled:
            self.llm.eval()
        self.moment_encoder.train(mode)
        return self

    def state_dict(self) -> OrderedDict[str, Tensor]:
        """Save adapters plus trainable LLM/MOMENT parameters, not frozen copies."""
        state = nn.Module.state_dict(self)
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        for key in list(state.keys()):
            if key == "word_embeddings":
                del state[key]
            elif key.startswith("llm.") and key not in trainable:
                del state[key]
            elif (
                key.startswith("moment_encoder.backbone.")
                and not self.save_frozen_moment
                and key not in trainable
            ):
                del state[key]
        return OrderedDict(state)
