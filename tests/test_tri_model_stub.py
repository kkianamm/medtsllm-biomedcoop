"""End-to-end smoke test using a tiny stand-in for the external MedTsLLM base."""
from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class DummyLLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.block = nn.Linear(dim, dim)

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=self.block(inputs_embeds))


class DummyBioHead(nn.Module):
    def __init__(self, dim: int, classes: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.aux_loss = None
        self.classes = classes

    def forward(self, sample, prototypes, labels=None):
        class_vectors = prototypes.mean(1)
        logits = self.proj(sample) @ class_vectors.t()
        self.aux_loss = logits.square().mean() * 1e-4 if labels is not None else None
        return logits


class DummyMedTsLLM(nn.Module):
    supported_tasks = ["classification"]
    supported_modes = ["multivariate"]

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.model_config = config.models.medtsllm
        self.n_features = dataset.n_features
        self.n_classes = dataset.n_classes
        self.d_llm = 48
        self.use_biomedcoop = True
        self.llm_enabled = True
        self.lora_enabled = False
        self.device = None
        self.llm = DummyLLM(self.d_llm)
        self.bc_head = DummyBioHead(self.d_llm, self.n_classes)
        self._bc_prototypes = None
        self.med_encoder = nn.Linear(self.n_features, self.d_llm)

    def encode_ts(self, x):
        # Downsample time to a small token sequence.
        x = x[:, ::4, :]
        return self.med_encoder(x)

    def build_prompt(self, inputs):
        return [[] for _ in range(inputs["x_enc"].shape[0])]

    def encode_part(self, part):
        raise AssertionError("No prompt parts are used in this smoke test")

    def pad_sequence(self, seq, seq_len):
        return seq

    def _build_class_prototypes(self):
        self._bc_prototypes = torch.randn(
            self.n_classes, 3, self.d_llm, device=self.device
        )


def test_full_combined_forward_and_backward(monkeypatch):
    fake = types.ModuleType("models.medtsllm")
    fake.MedTsLLM = DummyMedTsLLM
    monkeypatch.setitem(sys.modules, "models.medtsllm", fake)
    sys.modules.pop("models.tri_medtsllm", None)
    module = importlib.import_module("models.tri_medtsllm")

    combined = AttrDict(
        q_dim=64,
        q_queries=8,
        q_depth=2,
        q_heads=8,
        dropout=0.0,
        moment=AttrDict(
            backend="local",
            freeze=True,
            allow_local_fallback=True,
            local_patch_size=8,
            max_leads=16,
            use_highres_windows=True,
        ),
        loss=AttrDict(),
    )
    med_cfg = AttrDict(combined=combined)
    config = AttrDict(
        task="classification",
        models=AttrDict(medtsllm=med_cfg),
    )
    dataset = SimpleNamespace(n_features=12, n_classes=5)
    model = module.TriMedTsLLM(config, dataset)
    model.train()

    x = torch.randn(3, 128, 12)
    windows = torch.randn(3, 2, 96, 12)
    labels = torch.tensor([0, 2, 4])
    logits = model({"x_enc": x, "x_moment_windows": windows, "labels": labels})
    assert logits.shape == (3, 5)
    assert model.aux_loss is not None and torch.isfinite(model.aux_loss)
    loss = torch.nn.functional.cross_entropy(logits, labels) + model.aux_loss
    loss.backward()
    assert model.qformer.query_tokens.grad is not None
