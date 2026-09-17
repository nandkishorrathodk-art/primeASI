"""Central configuration for the Forge framework."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2
    expert_hidden: int = 256
    router_jitter: float = 0.01
    # Coefficient on the Switch-style load-balancing loss.  Tuned empirically:
    # too low and one expert starves, too high and routing stops being
    # input-dependent (every token goes everywhere, i.e. a dense model again).
    aux_loss_coef: float = 0.05
    # DeepSeek-style always-on experts that every token passes through.  They
    # carry the common patterns, leaving the routed experts to specialise.
    shared_experts: int = 1
    shared_hidden: int = 256
    # "token_choice": each token picks top_k experts (standard).
    # "expert_choice": each expert picks its own top-C tokens (perfectly
    #   balanced, but it looks at the whole batch at once, so it is NOT
    #   causal and must not be used for autoregressive decoding).
    routing: str = "token_choice"
    capacity_factor: float = 1.25


@dataclass
class VisionConfig:
    enabled: bool = True
    image_size: int = 32
    patch_size: int = 8
    channels: int = 3
    # number of learned summary tokens injected per image
    summary_tokens: int = 4


@dataclass
class ModelConfig:
    vocab_size: int = 260
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    # Grouped-Query Attention: fewer key/value heads than query heads.  With
    # n_heads=8 and n_kv_heads=4 there are 2 query heads per KV head, which
    # cuts KV cache size and attention parameters roughly in half while
    # keeping full query expressivity.  n_kv_heads = n_heads recovers plain
    # multi-head attention, so this is a strict generalisation.
    n_kv_heads: int = 0          # 0 means "same as n_heads" (plain MHA)
    # Measured on 4 CPU cores: 256 costs ~9.0k tok/s, 512 costs ~5.5k (3.3x
    # slower per step for 2x context).  128 could not fit even a compact
    # grammar prompt, so the model was asked a question it could not see.
    max_seq_len: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True
    moe: MoEConfig = field(default_factory=MoEConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    def param_estimate(self) -> int:
        """Exact parameter count for this configuration.

        Mirrors the modules that actually exist, including bias terms, so it
        can be used to *solve* for a target size rather than only to
        sanity-check one.  ``tests/test_gqa.py`` asserts this equals a built
        model's ``num_params()`` across many configurations; a hand-written
        estimate that silently disagrees with reality is how a claimed
        parameter count becomes a false claim.
        """
        d = self.dim
        m = self.moe
        n_kv = self.kv_heads()
        head_dim = d // self.n_heads

        per_layer = 0
        per_layer += d * self.n_heads * head_dim        # q projection
        per_layer += 2 * d * n_kv * head_dim            # k and v projections
        per_layer += d * d                              # attention output proj
        per_layer += 4 * d                              # two layernorms, w + b
        per_layer += d * m.num_experts                  # router (no bias)
        # Routed and shared experts have independent widths; using
        # expert_hidden for both silently mis-counts whenever they differ.
        routed_hidden = m.expert_hidden
        shared_hidden = m.shared_hidden or m.expert_hidden
        per_layer += m.num_experts * 3 * d * routed_hidden
        per_layer += m.shared_experts * 3 * d * shared_hidden

        total = self.vocab_size * d                     # token embedding
        if not self.tie_embeddings:
            total += self.vocab_size * d
        total += self.n_layers * per_layer
        total += 2 * d                                  # final layernorm, w + b
        total += self.max_seq_len * d                   # positional embedding

        if self.vision.enabled:
            total += self.vision_param_estimate()
        return total

    def vision_param_estimate(self) -> int:
        """Exact vision-tower count, biases included."""
        v = self.vision
        d = self.dim
        p = v.patch_size
        n_patches = (v.image_size // p) ** 2

        total = (p * p * v.channels) * d + d            # patch embed, w + b
        total += n_patches * d                          # positional
        # Two encoder blocks, matching vision._EncoderBlock.  MultiheadAttention
        # carries both in_proj_weight/bias and out_proj weight/bias; omitting
        # in_proj_bias leaves the estimate 3*d short per block.
        for _ in range(2):
            total += 2 * d                              # norm1
            total += 3 * d * d + 3 * d                  # in_proj weight + bias
            total += d * d + d                          # out_proj weight + bias
            total += 2 * d                              # norm2
            total += d * (2 * d) + (2 * d)              # mlp.0 weight + bias
            total += (2 * d) * d + d                    # mlp.2 weight + bias
        total += v.summary_tokens * d                   # summary tokens
        total += d * d + d                              # proj, w + b
        total += 2 * d                                  # final layernorm
        return total


@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 256
    steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup: int = 20
    log_every: int = 10
    eval_every: int = 50
    seed: int = 1337
    device: str = "cpu"
    out_dir: str = "runs/base"
    image_ratio: float = 0.25


@dataclass
class DistillConfig:
    """Teacher-model distillation settings.

    Teachers are optional: they are only contacted when credentials exist, and
    the framework always runs without them so training stays reproducible.
    """
    teachers: list[str] = field(default_factory=lambda: ["local"])
    temperature: float = 1.0
    alpha_kl: float = 0.5
    cache_dir: str = "runs/distill_cache"


@dataclass
class ForgeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ForgeConfig":
        model = ModelConfig(**{k: v for k, v in d["model"].items()
                               if k not in ("moe", "vision")})
        model.moe = MoEConfig(**d["model"].get("moe", {}))
        model.vision = VisionConfig(**d["model"].get("vision", {}))
        return ForgeConfig(
            model=model,
            train=TrainConfig(**d["train"]),
            distill=DistillConfig(**d["distill"]),
        )