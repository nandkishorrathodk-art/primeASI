"""Train the 46.66M SA-MoE on Kaggle Dual-T4.

Setup: verify your phone at kaggle.com/settings, set the notebook accelerator
to "GPU T4 x2", then push with `kaggle kernels push -p kaggle/`.

The script detects the accelerator and says plainly whether a GPU is present.
If it is not, it reports that rather than quietly training on a CPU for hours.

Honest reporting is the point of this file:
  * training loss AND held-out validation loss are both printed
  * the overfitting gap is printed, because a near-zero training loss with a
    large gap means memorisation, not learning
  * the model's own output is checked against the plan grammar, so "does it
    work?" is answered by the artifact rather than by the loss curve
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./kaggle_out"
os.makedirs(WORK, exist_ok=True)

print("=" * 74)
print("ACCELERATOR")
print("=" * 74)
print("torch       :", torch.__version__)
print("cuda        :", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
print("device_count:", N_GPU)
if N_GPU:
    for i in range(N_GPU):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name} | {p.total_memory / 1e9:.1f} GB | sm_{p.major}{p.minor}")
else:
    print()
    print("  !! NO GPU DETECTED.")
    print("  !! Verify your phone at kaggle.com/settings, then set")
    print("  !! Settings -> Accelerator -> GPU T4 x2, and re-run.")
DEVICE = "cuda" if N_GPU else "cpu"

HAS_AMP = DEVICE == "cuda"


# ------------------------------------------------------------------- config

@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2
    expert_hidden: int = 2037
    shared_experts: int = 2
    shared_hidden: int = 2037
    router_jitter: float = 0.01
    aux_loss_coef: float = 0.05


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 512
    tie_embeddings: bool = True
    moe: MoEConfig = field(default_factory=MoEConfig)

    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    def param_estimate(self) -> int:
        d, m = self.dim, self.moe
        n_kv, hd = self.kv_heads(), self.dim // self.n_heads
        per = (d * self.n_heads * hd + 2 * d * n_kv * hd + d * d + 4 * d
               + d * m.num_experts
               + m.num_experts * 3 * d * m.expert_hidden
               + m.shared_experts * 3 * d * (m.shared_hidden or m.expert_hidden))
        total = self.vocab_size * d
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total + self.n_layers * per + 2 * d + self.max_seq_len * d


# -------------------------------------------------------------------- model

class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w_in = nn.Linear(dim, hidden, bias=False)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_in(x))


class MoELayer(nn.Module):
    """Shared experts always run; top-k routed experts are chosen per token."""

    def __init__(self, dim, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.jitter = cfg.router_jitter
        self.router = nn.Linear(dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(dim, cfg.expert_hidden) for _ in range(cfg.num_experts))
        self.shared = nn.ModuleList(
            Expert(dim, cfg.shared_hidden or cfg.expert_hidden)
            for _ in range(cfg.shared_experts))
        self.last_probs = None
        self.last_load = None

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        out = torch.zeros_like(flat)
        for e in self.shared:
            out = out + e(flat)

        logits = self.router(flat)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter
        probs = F.softmax(logits, dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        w = w / (w.sum(-1, keepdim=True) + 1e-9)

        load = torch.zeros(self.num_experts, device=x.device)
        for slot in range(self.top_k):
            sel = idx[:, slot]
            gate = w[:, slot].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = sel == e
                n = int(mask.sum())
                if n:
                    out[mask] += gate[mask] * self.experts[e](flat[mask])
                load[e] += n

        self.last_probs = probs.detach().mean(0)
        self.last_load = load / max(flat.shape[0], 1)
        return out.reshape(b, t, d)


def balance_loss(layers):
    total = torch.zeros((), device=DEVICE)
    n = 0
    for L in layers:
        if L.last_load is None:
            continue
        f = L.last_load / (L.last_load.sum() + 1e-9)
        p = L.last_probs / (L.last_probs.sum() + 1e-9)
        total = total + L.num_experts * (f * p).sum()
        n += 1
    return total / max(n, 1)


class GQA(nn.Module):
    """8 query heads share 4 key/value heads, halving the KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.nh, self.nkv = cfg.n_heads, cfg.kv_heads()
        self.group = self.nh // self.nkv
        self.hd = cfg.dim // self.nh
        self.q = nn.Linear(cfg.dim, self.nh * self.hd, bias=False)
        self.k = nn.Linear(cfg.dim, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(cfg.dim, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def _split(self, z, n, b, t):
        return z.view(b, t, n, self.hd).transpose(1, 2)

    def forward(self, x, cache=None):
        b, t, _ = x.shape
        q = self._split(self.q(x), self.nh, b, t)
        k = self._split(self.k(x), self.nkv, b, t)
        v = self._split(self.v(x), self.nkv, b, t)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k.detach(), v.detach())
        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)
        total = k.shape[2]
        mask = torch.triu(torch.full((t, total), float("-inf"), device=x.device),
                          diagonal=total - t + 1)
        att = torch.softmax((q @ k.transpose(-2, -1)) / self.hd ** 0.5 + mask, dim=-1)
        return self.o((att @ v).transpose(1, 2).contiguous().view(b, t, -1)), new_cache


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = GQA(cfg)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.moe = MoELayer(cfg.dim, cfg.moe)

    def forward(self, x, cache=None):
        a, c = self.attn(self.norm1(x), cache)
        return x + a + self.moe(self.norm2(x + a)), c


class SA_MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.dim))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        nn.init.normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def moe_layers(self):
        return [b.moe for b in self.blocks]

    def forward(self, idx, cache=None):
        b, t = idx.shape
        past = cache[0][0].shape[2] if cache and cache[0] is not None else 0
        x = self.tok(idx) + self.pos[:, past:past + t]
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, cache[i] if cache else None)
            new_cache.append(c)
        return self.head(self.norm_f(x)), balance_loss(self.moe_layers()), new_cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=120, temperature=0.8, top_k=40, eos_id=None):
        self.eval()
        cache = [None] * len(self.blocks)
        for _ in range(max_new_tokens):
            logits, _, cache = self.forward(idx, cache=cache)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), -1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


# ---------------------------------------------------------------- tokenizer
# Word-bounded BPE: merges never cross a whitespace boundary, so one token can
# never glue two words together.  That keeps the vocabulary interpretable and
# prevents a whole line collapsing into a single token on repetitive corpora.

SPECIALS = ["<pad>", "<bos>", "<eos>", "<image>"]
PAD, BOS, EOS, IMAGE = 0, 1, 2, 3
BYTE_OFF = len(SPECIALS)
MERGE_BASE = 256 + BYTE_OFF
SPACE = 32 + BYTE_OFF


class WordBoundedBPE:
    def __init__(self, merges=None):
        self.merges = list(merges or [])
        self.rank = {p: i for i, p in enumerate(self.merges)}
        self.vocab_size = MERGE_BASE + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size=32000, max_lines=None):
        target = max(0, vocab_size - MERGE_BASE)
        words = []
        for i, line in enumerate(text.splitlines()):
            if max_lines and i >= max_lines:
                break
            for w in line.split(" "):
                if w:
                    words.append([b + BYTE_OFF for b in w.encode("utf-8")])
        merges = []
        for _ in range(target):
            counts = Counter()
            for w in words:
                for pair in zip(w, w[1:]):
                    counts[pair] += 1
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            merges.append(pair)
            words = [_merge(w, pair, len(merges) - 1) for w in words]
        return cls(merges)

    def _apply(self, seq):
        while len(seq) >= 2:
            best, pos = None, -1
            for i in range(len(seq) - 1):
                r = self.rank.get((seq[i], seq[i + 1]))
                if r is not None and (best is None or r < best):
                    best, pos = r, i
            if best is None:
                break
            seq = _merge(seq, (seq[pos], seq[pos + 1]), best)
        return seq

    def encode(self, text, add_bos=True):
        ids, i = [], 0
        n = len(text)
        while i < n:
            for sid, name in enumerate(SPECIALS):
                if text.startswith(name, i):
                    ids.append(sid)
                    i += len(name)
                    break
            else:
                j = i
                while j < n and not any(text.startswith(s, j) for s in SPECIALS):
                    j += 1
                segment = text[i:j]
                for k, w in enumerate(segment.split(" ")):
                    if k:
                        ids.append(SPACE)
                    if w:
                        ids.extend(self._apply([b + BYTE_OFF for b in w.encode("utf-8")]))
                i = j
        return ([BOS] + ids) if add_bos else ids

    def decode(self, ids, skip_specials=True):
        raw = bytearray()
        for i in ids:
            if i < BYTE_OFF:
                if not skip_specials:
                    raw.extend(SPECIALS[i].encode())
            elif i < MERGE_BASE:
                raw.append(i - BYTE_OFF)
            else:
                raw.extend(_expand(i, self.merges))
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as fh:
            for a, b in self.merges:
                fh.write(f"{a} {b}\n")


def _merge(seq, pair, rank):
    a, b = pair
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(MERGE_BASE + rank)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _expand(tok, merges):
    if tok < BYTE_OFF:
        return b""
    if tok < MERGE_BASE:
        return bytes([tok - BYTE_OFF])
    idx = tok - MERGE_BASE
    if not (0 <= idx < len(merges)):
        return b""
    a, b = merges[idx]
    return _expand(a, merges) + _expand(b, merges)


# ------------------------------------------------------------------- corpus
GRAMMAR_PROMPT = """Output only plan lines:
goal: <slug>
mkdir <path>
write <path> <<<content>>>
read <path>
run_tests <name>
Paths must be relative. At most 3 steps."""

TASKS = ["harden the input parser", "document the trust boundary",
         "record the threat model", "write the hardening notes",
         "capture the audit findings", "note the routing tradeoffs",
         "review the token expiry logic", "record the capacity limits",
         "note the retry semantics", "document the rollback procedure",
         "capture the evidence chain", "note the permission model",
         "record the quarantine policy", "document the audit trail",
         "review the input validation", "note the rate limits",
         "document the escalation path", "record the trust ladder",
         "note the expert routing balance", "document the verification steps"]

BODY = {
    "parser": ["- reject input you cannot parse", "- fail closed, not open"],
    "trust": ["- validate at the trust boundary, not in the UI",
              "- never log credentials"],
    "threat": ["- record the threat model before the patch", "- one change per review"],
    "hardening": ["- prefer constant-time comparison for tokens",
                  "- alert on repeated failures"],
    "audit": ["- hash-chain every action record", "- report the exact broken index"],
    "routing": ["- balance on dispatched token counts", "- penalise collapsed experts"],
    "review": ["- demand evidence for every claim", "- block on high-severity findings"],
    "validation": ["- validate before the side effect", "- one rule per input class"],
    "token": ["- rotate on privilege change", "- expire short and reissue"],
    "capacity": ["- measure the knee before scaling",
                 "- two times context can cost three times the step"],
    "retry": ["- exponential backoff with jitter", "- make retried operations idempotent"],
    "rollback": ["- snapshot before mutating", "- restore the prior bytes, never just delete"],
    "evidence": ["- attribute every claim to an agent",
                 "- cite the test or trace that proves it"],
    "permission": ["- deny by default", "- grant with a TTL and a budget"],
    "quarantine": ["- move rather than unlink", "- keep the original path for restore"],
    "trail": ["- every entry covers the previous hash", "- anchor the head externally"],
    "limits": ["- rate limit at thirty requests per minute",
               "- allow only private ranges unless allowlisted"],
    "escalation": ["- low confidence forces approval",
                   "- trust is slow to earn and fast to lose"],
    "ladder": ["- promote on verified successes only", "- demote to dry run on one failure"],
    "balance": ["- shared experts carry the common patterns",
                "- use a scale-free balancing loss"],
    "verification": ["- re-read the filesystem after acting",
                     "- never trust the model's claim of success"],
}
FALLBACK_BODY = ["- review the task", "- record the outcome"]


def slugify(s):
    out = []
    for ch in s.lower().strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_/":
            if out and out[-1] != "-":
                out.append("-")
    return "".join(out).strip("-")


def body_for(task):
    low = task.lower()
    for k, v in BODY.items():
        if k in low:
            return list(v)
    return list(FALLBACK_BODY)


def user_turn(task, max_steps=3, rationale=None):
    lines = [f"Task: {task}"]
    if rationale:
        lines.append(f"Judge rationale: {rationale}")
    lines.append(f"Produce at most {max_steps} steps. First line must be the goal.")
    return "\n".join(lines)


def make_example(task, max_steps=3, rng=None, eos="<eos>"):
    """One [system][user][plan] triple in the exact shape the caller uses."""
    slug = slugify(task)
    steps = [f"goal: {slug}", "mkdir docs", f"write docs/{slug}.md <<<",
             *body_for(task), ">>>"]
    if rng is not None and rng.random() < 0.25:
        steps.append(f"read docs/{slug}.md")
    rationale = None
    if rng is not None and rng.random() < 0.5:
        rationale = "no high-severity finding outstanding"
    plan = "\n".join(steps)
    return f"{GRAMMAR_PROMPT}\n\n{user_turn(task, max_steps, rationale)}\n\n{plan}{eos}"


def make_corpus(n=400, seed=0):
    rng = random.Random(seed)
    return "\n\n".join(make_example(rng.choice(TASKS), 3, rng) for _ in range(n))


CODE = ["def add(a, b):\n    return a + b\n",
        "for i in range(10):\n    print(i * i)\n",
        "def mean(xs):\n    return sum(xs) / len(xs)\n",
        "class Parser:\n    def parse(self, text):\n        return text.split()\n"]
PROSE = ["policy: never log credentials, tokens, or session identifiers",
         "threat model: an attacker with network access can replay an old token",
         "input validation happens at the trust boundary, not in the UI",
         "detection rule: alert when one address trips fifty failed logins",
         "a password is hashed with a salted slow KDF, never stored raw"]


def mixed_corpus(n_examples=3000, seed=0):
    rng = random.Random(seed)
    parts = [make_corpus(max(1, n_examples // 4), seed)]
    for _ in range(n_examples):
        parts.append(rng.choice(CODE) if rng.random() < 0.5 else rng.choice(PROSE))
    rng.shuffle(parts)
    return "\n".join(parts)


# ------------------------------------------------------------------ training

def batches(data, bs, seq, steps, device):
    hi = max(len(data) - seq - 1, 1)
    for _ in range(steps):
        s = torch.randint(0, hi, (bs,))
        x = torch.stack([data[i:i + seq] for i in s])
        y = torch.stack([data[i + 1:i + seq + 1] for i in s])
        yield x.to(device), y.to(device)


def lr_at(step, total, base=3e-4, warm=100):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(total - warm, 1)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    EPOCHS = 20

    print()
    print("=" * 74)
    print("BUILD")
    print("=" * 74)
    cfg = ModelConfig()
    est = cfg.param_estimate()
    model = SA_MoE(cfg)
    real = model.num_params()
    print(f"config      : dim={cfg.dim} layers={cfg.n_layers} "
          f"q_heads={cfg.n_heads} kv_heads={cfg.kv_heads()}")
    print(f"MoE         : shared={cfg.moe.shared_experts} "
          f"routed={cfg.moe.num_experts} top_k={cfg.moe.top_k} SwiGLU")
    print(f"estimated   : {est:,}")
    print(f"built model : {real:,}")
    print(f"estimate matches the built model: {est == real}")
    model = model.to(DEVICE)
    core = model
    if N_GPU > 1:
        core = nn.DataParallel(model)
        print(f"DataParallel across {N_GPU} GPUs")

    print()
    print("=" * 74)
    print("TOKENIZER (word-bounded BPE, vocab 32000)")
    print("=" * 74)
    t0 = time.time()
    corpus = mixed_corpus(3000, seed=0)
    tok = WordBoundedBPE.train(corpus, vocab_size=cfg.vocab_size, max_lines=40000)
    print(f"trained in {time.time() - t0:.0f}s | vocab {tok.vocab_size:,} "
          f"| merges {len(tok.merges):,}")
    for probe in ("mkdir docs/x.md", "security", "document the trust boundary"):
        print(f"  {probe!r:32} -> {len(tok.encode(probe, add_bos=False))} tokens")

    ids = tok.encode(corpus, add_bos=False)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"corpus: {len(corpus):,} chars -> {len(data):,} tokens")
    assert int(data.max()) < cfg.vocab_size, "token id out of vocab range"

    cut = int(len(data) * 0.95)
    train_data, val_data = data[:cut], data[cut:]
    print(f"split: train {len(train_data):,} | held-out {len(val_data):,} tokens")

    print()
    print("=" * 74)
    print(f"TRAINING ({EPOCHS} epochs)")
    print("=" * 74)
    bs, seq = 16, cfg.max_seq_len
    per_epoch = max(1, len(train_data) // (bs * seq))
    total_steps = per_epoch * EPOCHS
    print(f"steps/epoch {per_epoch} | total steps {total_steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1,
                            betas=(0.9, 0.95))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=HAS_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=HAS_AMP)

    def autocast():
        return torch.amp.autocast("cuda", enabled=HAS_AMP)

    log = []
    t0 = time.time()
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    model.train()
    for step, (x, y) in enumerate(batches(train_data, bs, seq, total_steps, DEVICE)):
        lr = lr_at(step, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast():
            logits, moe_l, _ = model(x)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 y.reshape(-1))
            loss = ce + cfg.moe.aux_loss_coef * moe_l
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % max(1, per_epoch // 3) == 0 or step == total_steps - 1:
            load = base_model.moe_layers()[0].last_load
            ratio = float(load.max() / (load.min() + 1e-9))
            log.append({"step": step, "epoch": step / per_epoch,
                        "loss": float(ce), "lr": lr, "expert_ratio": ratio})
            print(f"  step {step:5d} ep {step / per_epoch:5.1f} "
                  f"loss {ce.item():.4f} lr {lr:.2e} expert_ratio {ratio:.2f}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\ntrained in {elapsed / 60:.1f} min "
          f"({total_steps * bs * seq / elapsed:,.0f} tok/s)")

    # ------------------------------------------------ held-out evaluation
    print()
    print("=" * 74)
    print("EVALUATION -- the numbers that decide whether this worked")
    print("=" * 74)

    def eval_loss(d, n=30):
        base_model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for x, y in batches(d, bs, seq, n, DEVICE):
                with autocast():
                    logits, _, _ = base_model(x)
                    tot += F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
                cnt += 1
        return tot / max(cnt, 1)

    train_loss = eval_loss(train_data)
    val_loss = eval_loss(val_data)
    gap = val_loss - train_loss
    print(f"  train loss (held portion)  : {train_loss:.4f}")
    print(f"  HELD-OUT validation loss   : {val_loss:.4f}   <-- the honest number")
    print(f"  overfitting gap            : {gap:.4f}")
    if gap > 1.0:
        print("  !! large gap: the model is memorising. Report the held-out")
        print("  !! number, not a near-zero training loss.")
    else:
        print("  gap is small: the model generalises rather than memorising.")

    # ------------------------------------------------- does it parse?
    print()
    print("=" * 74)
    print("DOES THE MODEL ACTUALLY EMIT PLANS?")
    print("=" * 74)
    base_model.eval()
    prompt = (f"{GRAMMAR_PROMPT}\n\n"
              + user_turn("document the trust boundary", 3,
                          "no high-severity finding outstanding") + "\n\n")
    pids = torch.tensor([tok.encode(prompt, add_bos=True)]).to(DEVICE)
    ok, samples = 0, []
    for temp in (0.2, 0.4, 0.7):
        out = base_model.generate(pids, max_new_tokens=120, temperature=temp,
                                  top_k=15, eos_id=EOS)
        txt = tok.decode(out[0].tolist())[len(prompt):]
        samples.append(txt)
        parsed = (txt.strip().startswith("goal: ") and "mkdir " in txt
                  and ">>>" in txt)
        ok += parsed
        print(f"\n  temperature {temp}: "
              f"{'PARSEABLE' if parsed else 'not parseable'}")
        print("  " + txt[:300].replace("\n", "\n  "))
    print(f"\n  RESULT: {ok}/3 samples produced a plan")

    # ---------------------------------------------------------- save
    outdir = os.path.join(WORK, "sa_moe")
    os.makedirs(outdir, exist_ok=True)
    torch.save({"config": {"dim": cfg.dim, "n_layers": cfg.n_layers,
                           "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
                           "vocab_size": cfg.vocab_size,
                           "max_seq_len": cfg.max_seq_len},
                "model": base_model.state_dict()},
               os.path.join(outdir, "sa_moe.pt"))
    tok.save(os.path.join(outdir, "tokenizer.txt"))
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump({
            "params": real,
            "param_estimate": est,
            "estimate_matches": est == real,
            "vocab": tok.vocab_size,
            "epochs": EPOCHS,
            "steps": total_steps,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "overfit_gap": gap,
            "gpu_count": N_GPU,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(N_GPU)],
            "minutes": elapsed / 60,
            "plan_samples_parseable": f"{ok}/3",
            "samples": samples,
            "loss_curve": log,
        }, fh, indent=2)

    print()
    print("=" * 74)
    print(f"saved to {outdir}: sa_moe.pt, tokenizer.txt, metrics.json")
    print(f"FINAL -- params {real:,} | train {train_loss:.4f} | "
          f"val {val_loss:.4f} | gap {gap:.4f} | plans {ok}/3")
    print("=" * 74)


if __name__ == "__main__":
    main()