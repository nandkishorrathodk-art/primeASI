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
    max_seq_len: int = 128
    dropout: float = 0.0
    tie_embeddings: bool = True
    moe: MoEConfig = field(default_factory=MoEConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    def param_estimate(self) -> int:
        """Rough parameter count, used for sanity checks on small machines."""
        d = self.dim
        per_layer = 4 * d * d
        m = self.moe
        per_layer += (m.num_experts + m.shared_experts) * 3 * d * m.expert_hidden
        vision = 0
        if self.vision.enabled:
            p = self.vision.patch_size
            vision = (p * p * self.vision.channels) * d + self.vision.summary_tokens * d
        return self.vocab_size * d + self.n_layers * per_layer + vision


@dataclass
class TrainConfig:
    batch_size: int = 16
    seq_len: int = 128
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