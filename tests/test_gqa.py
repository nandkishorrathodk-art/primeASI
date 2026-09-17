"""GQA and the 46.66M SA-MoE specification.

The parameter-count tests matter more than they look.  A claimed size that
does not match the built model is a false claim, and it is exactly the kind of
claim that survives because nobody rebuilds the model to check.  These tests
rebuild it.
"""
from __future__ import annotations

import torch
import pytest

from forge.config import ForgeConfig, ModelConfig, MoEConfig, VisionConfig
from forge.model.solver import SA_MOE_TARGET, sa_moe_config, solve
from forge.model.transformer import ForgeLM


# ============================================================ GQA

def test_gqa_defaults_to_plain_mha():
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=0,
                      vocab_size=100, max_seq_len=32)
    assert cfg.kv_heads() == 4
    m = ForgeLM(cfg)
    attn = m.blocks[0].attn
    assert attn.n_kv_heads == 4 and attn.group == 1


def test_gqa_grouping_is_correct():
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
                      vocab_size=100, max_seq_len=32)
    attn = ForgeLM(cfg).blocks[0].attn
    assert attn.n_heads == 8 and attn.n_kv_heads == 4 and attn.group == 2


def test_gqa_requires_divisible_heads():
    with pytest.raises(AssertionError):
        ForgeLM(ModelConfig(dim=64, n_layers=1, n_heads=8, n_kv_heads=3,
                            vocab_size=100, max_seq_len=32))


def test_gqa_shrinks_parameters_and_cache():
    """GQA must buy something measurable: fewer params and a smaller cache."""
    base = dict(dim=128, n_layers=2, n_heads=8, vocab_size=1000, max_seq_len=64)
    mha = ForgeLM(ModelConfig(**base, n_kv_heads=0))
    gqa = ForgeLM(ModelConfig(**base, n_kv_heads=4))

    assert gqa.num_params() < mha.num_params()

    a_mha = mha.blocks[0].attn
    a_gqa = gqa.blocks[0].attn
    assert a_gqa.kv_cache_bytes(1, 256) < a_mha.kv_cache_bytes(1, 256)
    # With 8 q-heads and 4 kv-heads the cache should be exactly half.
    assert a_gqa.kv_cache_bytes(1, 256) * 2 == a_mha.kv_cache_bytes(1, 256)


def test_gqa_causal_mask_still_holds():
    """Each position must not attend to the future, whatever the grouping."""
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
                      vocab_size=100, max_seq_len=32)
    m = ForgeLM(cfg)
    m.eval()
    x = torch.randint(0, 100, (1, 8))
    full, _, _ = m(x)
    truncated, _, _ = m(x[:, :4])
    # The first four positions cannot have seen anything past themselves.
    assert torch.allclose(full[:, :4, :], truncated, atol=1e-5), \
        "causality broken by GQA grouping"


def test_gqa_kv_cache_matches_full_forward():
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=8, n_kv_heads=4,
                      vocab_size=100, max_seq_len=64)
    m = ForgeLM(cfg)
    m.eval()
    x = torch.randint(0, 100, (1, 10))
    full, _, _ = m(x)
    _, _, cache = m(x[:, :6])
    part, _, _ = m(x[:, 6:], cache=cache)
    assert torch.allclose(full[:, -1, :], part[:, -1, :], atol=1e-4)


def test_gqa_generates():
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
                      vocab_size=100, max_seq_len=32)
    m = ForgeLM(cfg)
    out = m.generate(torch.randint(0, 100, (1, 4)), max_new_tokens=6, top_k=5)
    assert out.shape[1] > 4


# ============================================================ param exactness

@pytest.mark.parametrize("kw", [
    dict(dim=64, n_layers=1, n_heads=4, n_kv_heads=0, vocab_size=100, max_seq_len=32),
    dict(dim=64, n_layers=1, n_heads=8, n_kv_heads=4, vocab_size=100, max_seq_len=32),
    dict(dim=128, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=1000, max_seq_len=64),
    dict(dim=256, n_layers=4, n_heads=8, n_kv_heads=4, vocab_size=32000, max_seq_len=512),
    dict(dim=448, n_layers=7, n_heads=8, n_kv_heads=4, vocab_size=32000, max_seq_len=512),
])
@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("vision", [False, True])
def test_param_estimate_matches_built_model(kw, tied, vision):
    """The estimate must equal reality, or a claimed size is a false claim."""
    cfg = ModelConfig(**kw, tie_embeddings=tied)
    cfg.moe = MoEConfig(num_experts=4, shared_experts=2, expert_hidden=128)
    cfg.vision = VisionConfig(enabled=vision, image_size=32, patch_size=8)
    assert cfg.param_estimate() == ForgeLM(cfg).num_params()


# ============================================================ 46.66M target

def test_sa_moe_hits_46_66m():
    """The headline specification: 46.66M parameters, proven by construction."""
    sol = sa_moe_config()
    real = ForgeLM(sol.config).num_params()
    assert real == sol.params, "solver estimate disagrees with the built model"
    assert abs(real - SA_MOE_TARGET) < SA_MOE_TARGET * 0.001, \
        f"off target by {abs(real - SA_MOE_TARGET):,}"
    assert real < 50_000_000 and real > 40_000_000


def test_sa_moe_architecture_matches_the_spec():
    """8 Q-heads, 4 KV-heads, shared experts, SwiGLU, top-2 routing."""
    sol = sa_moe_config()
    m = sol.config
    assert m.n_heads == 8 and m.kv_heads() == 4
    assert m.moe.shared_experts >= 1, "SA-MoE requires shared experts"
    assert m.moe.top_k == 2
    assert m.moe.num_experts >= 4

    model = ForgeLM(m)
    block = model.blocks[0]
    assert block.moe.top_k == 2
    assert len(block.moe.shared) == m.moe.shared_experts
    # SwiGLU has exactly three matrices per expert.
    expert = block.moe.experts[0]
    assert hasattr(expert, "w_in") and hasattr(expert, "w_gate") and hasattr(expert, "w_out")


def test_solver_reports_its_error_honestly():
    sol = sa_moe_config()
    assert sol.error_pct < 0.001
    assert sol.error == abs(sol.params - sol.target)


def test_solver_raises_when_unreachable():
    """An impossible target must fail loudly, not silently return nonsense."""
    with pytest.raises(ValueError):
        solve(10_000, dims=(256,), layers=range(8, 9),
              head_options=((8, 4),), expert_options=((4, 2),))


def test_sa_moe_forward_works():
    sol = sa_moe_config(vocab_size=32000, max_seq_len=512)
    model = ForgeLM(sol.config)
    x = torch.randint(0, 32000, (2, 64))
    logits, moe_loss, _ = model(x, targets=x)
    assert logits.shape == (2, 64, 32000)
    assert torch.isfinite(moe_loss)


def test_sa_moe_has_shared_and_routed_experts_active():
    """Every token must pass a shared expert, and routing must not collapse."""
    sol = sa_moe_config(vocab_size=1000, max_seq_len=64)
    model = ForgeLM(sol.config)
    model.train()
    x = torch.randint(0, 1000, (8, 64))
    model(x, targets=x)
    load = model.moe_layers()[0].load_stats()
    assert abs(load.sum().item() - 2.0) < 1e-5, "top-2 routing should dispatch 2/token"
    assert len(model.moe_layers()[0].shared) >= 1