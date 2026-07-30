import torch

from models.tri_components import (
    GatedResidualFusion,
    LearnedQueryFormer,
    MomentTokenEncoder,
    align_token_count,
    query_diversity_loss,
    supervised_contrastive_alignment,
)


def test_combined_components_shapes():
    torch.manual_seed(7)
    batch, channels, length, dim = 4, 12, 250, 64
    x = torch.randn(batch, channels, length)
    moment = MomentTokenEncoder(
        input_channels=channels,
        token_dim=dim,
        backend="local",
        local_patch_size=10,
    )
    moment_tokens = moment(x).tokens
    med_tokens = torch.randn(batch, 31, dim)
    moment_tokens = align_token_count(moment_tokens, med_tokens.shape[1])
    fused, gate = GatedResidualFusion(dim)(med_tokens, moment_tokens)
    queries, maps = LearnedQueryFormer(
        dim=dim, num_queries=8, depth=2, heads=8
    )(fused)
    assert fused.shape == med_tokens.shape
    assert gate.shape == med_tokens.shape
    assert queries.shape == (batch, 8, dim)
    assert len(maps) == 2
    assert torch.isfinite(query_diversity_loss(queries))


def test_alignment_has_gradient():
    first = torch.randn(6, 32, requires_grad=True)
    second = torch.randn(6, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supervised_contrastive_alignment(first, second, labels)
    loss.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()
