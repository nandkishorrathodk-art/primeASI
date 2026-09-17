"""Training loops: language modelling, vision pretraining, joint multimodal."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from forge.config import ForgeConfig
from forge.data import ShapeDataset, corpus_to_tensor, make_corpus, sample_batch
from forge.model.transformer import ForgeLM
from forge.tokenizer import BPETokenizer, ByteTokenizer


def _default_tokenizer():
    return ByteTokenizer()


@dataclass
class StepLog:
    step: int
    loss: float
    ce: float
    moe: float
    lr: float
    tok_per_s: float
    expert_usage: list[float]


class Trainer:
    def __init__(
        self,
        cfg: ForgeConfig,
        model: Optional[ForgeLM] = None,
        tokenizer=None,
    ) -> None:
        self.cfg = cfg
        self.model = model or ForgeLM(cfg.model)
        # The tokenizer must travel with the weights.  A checkpoint whose
        # tokenizer is assumed rather than stored will decode its own output
        # wrongly the moment the vocabulary changes.
        self.tokenizer = tokenizer or _default_tokenizer()
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )
        self.logs: list[StepLog] = []
        self._tokens = 0

    def lr_at(self, step: int) -> float:
        """Linear warmup then cosine decay -- standard and stable on small runs."""
        t = self.cfg.train
        if step < t.warmup:
            return t.lr * (step + 1) / max(t.warmup, 1)
        progress = (step - t.warmup) / max(t.steps - t.warmup, 1)
        return t.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, step: int) -> float:
        lr = self.lr_at(step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def train_lm(
        self,
        text: Optional[str] = None,
        on_step: Optional[callable] = None,
    ) -> list[StepLog]:
        t = self.cfg.train
        torch.manual_seed(t.seed)
        text = text if text is not None else make_corpus()
        data = corpus_to_tensor(text, self.tokenizer)
        self.model.train()
        self.model.to(t.device)

        t0 = time.time()
        for step in range(t.steps):
            lr = self._set_lr(step)
            x, y = sample_batch(data, t.batch_size, t.seq_len, t.device)
            logits, moe_loss, _ = self.model(x, targets=y)
            loss, ce = self.model.loss(logits, y, moe_loss)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
            self.opt.step()

            self._tokens += x.numel()
            usage = self._expert_usage()
            log = StepLog(
                step=step, loss=loss.detach().item(), ce=ce.detach().item(), moe=moe_loss.detach().item(),
                lr=lr, tok_per_s=self._tokens / max(time.time() - t0, 1e-6),
                expert_usage=usage,
            )
            self.logs.append(log)
            if on_step:
                on_step(log)
            elif step % t.log_every == 0:
                print(
                    f"step {step:4d} loss {log.loss:.4f} ce {log.ce:.4f} "
                    f"moe {log.moe:.4f} lr {lr:.2e} {log.tok_per_s:,.0f} tok/s"
                )
        return self.logs

    def train_vision_classifier(
        self,
        steps: int = 200,
        batch_size: int = 16,
        n_classes: int = 4,
        on_step: Optional[callable] = None,
    ) -> list[StepLog]:
        """Pretrain the vision tower on shape brightness classification.

        A lightweight linear head sits on the summary tokens; gradients flow
        into the encoder.  The LM itself is frozen here so we can prove the
        vision path learns before wiring it into language modelling.
        """
        t = self.cfg.train
        ds = ShapeDataset(n=1500, size=self.cfg.model.vision.image_size, seed=t.seed)
        head = torch.nn.Linear(self.cfg.model.vision.summary_tokens * self.cfg.model.dim,
                               n_classes)
        params = list(self.model.vision.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.vision.parameters():
            p.requires_grad_(True)

        logs: list[StepLog] = []
        t0 = time.time()
        self._tokens = 0
        for step in range(steps):
            imgs, labels = ds.batch(batch_size)
            feats = self.model.vision(imgs).reshape(batch_size, -1)
            logits = head(feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc = (logits.argmax(-1) == labels).float().mean().item()
            log = StepLog(step=step, loss=loss.detach().item(), ce=loss.detach().item(), moe=0.0,
                          lr=1e-3, tok_per_s=batch_size / max(time.time() - t0, 1e-6),
                          expert_usage=[])
            logs.append(log)
            if on_step:
                on_step(log, acc)
            elif step % 20 == 0:
                print(f"vision step {step:4d} loss {loss:.4f} acc {acc:.3f}")
        self.logs.extend(logs)
        # Restore trainability of the whole model.
        for p in self.model.parameters():
            p.requires_grad_(True)
        return logs

    def _expert_usage(self) -> list[float]:
        """Fraction of tokens dispatched to each routed expert, averaged
        across layers.  Uses actual load, not router probability."""
        if not self.model.moe_layers():
            return []
        stats = [layer.load_stats().tolist() for layer in self.model.moe_layers()]
        n = len(stats[0])
        return [sum(s[i] for s in stats) / len(stats) for i in range(n)]

    def _dropped_tokens(self) -> float:
        vals = [l.last_info.dropped for l in self.model.moe_layers() if l.last_info]
        return sum(vals) / len(vals) if vals else 0.0

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "config": self.cfg.to_dict(),
            "model": self.model.state_dict(),
            "merges": list(getattr(self.tokenizer, "merges", [])),
            "extra": extra or {},
        }, path)

    @staticmethod
    def load(
        path: str, map_location: str = "cpu"
    ) -> tuple[ForgeLM, ForgeConfig]:
        model, cfg, _tok = Trainer.load_with_tokenizer(path, map_location)
        return model, cfg

    @staticmethod
    def load_with_tokenizer(
        path: str, map_location: str = "cpu"
    ) -> tuple[ForgeLM, ForgeConfig, object]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ForgeConfig.from_dict(ckpt["config"])
        model = ForgeLM(cfg.model)
        model.load_state_dict(ckpt["model"])
        model.eval()
        merges = ckpt.get("merges") or []
        tok = BPETokenizer([tuple(m) for m in merges]) if merges else _default_tokenizer()
        return model, cfg, tok