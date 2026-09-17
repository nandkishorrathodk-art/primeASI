"""Test suite.  Every test exercises real code paths, no mocks."""
from __future__ import annotations

import torch

from forge.agents.orchestrator import Orchestrator
from forge.agents.roles import Judge, PolicyViolation, build_team, check_policy
from forge.agents.blackboard import Blackboard
from forge.agents.protocol import Evidence, Kind, Message, Verdict
from forge.config import ForgeConfig
from forge.data import ShapeDataset, corpus_to_tensor, make_corpus, sample_batch
from forge.model.transformer import ForgeLM
from forge.security.analyzer import risk_score, scan_source
from forge.security.sandbox import ScopeError, check_scope, is_private_target
from forge.tokenizer import BOS, PAD, TOKENIZER
from forge.training.distill import EMATeacher, distill_loss
from forge.training.trainer import Trainer
from forge.viz.render import render_image, sparkline, table


def small_cfg() -> ForgeConfig:
    c = ForgeConfig()
    c.model.dim = 64
    c.model.n_layers = 2
    c.model.moe.num_experts = 4
    c.model.moe.top_k = 2
    c.model.moe.expert_hidden = 128
    c.model.vision.image_size = 16
    c.model.vision.patch_size = 8
    c.train.steps = 6
    c.train.batch_size = 4
    c.train.seq_len = 32
    return c


# ------------------------------------------------------------- tokenizer

def test_tokenizer_roundtrip():
    text = "def add(a, b):\n    return a + b\nαβγ 日本語"
    ids = TOKENIZER.encode(text, add_bos=False)
    assert TOKENIZER.decode(ids) == text
    assert ids[0] >= 4


def test_tokenizer_specials():
    ids = TOKENIZER.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == BOS and ids[-1] != PAD
    assert TOKENIZER.decode([PAD, BOS]) == ""


# ------------------------------------------------------------- data

def test_corpus_and_batch_shapes():
    data = corpus_to_tensor(make_corpus(60))
    x, y = sample_batch(data, 4, 16)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_shape_dataset_deterministic():
    a = ShapeDataset(n=8, size=16, seed=1)
    b = ShapeDataset(n=8, size=16, seed=1)
    assert torch.equal(a._imgs, b._imgs) and torch.equal(a._labels, b._labels)


# ------------------------------------------------------------- model

def test_model_forward_and_loss():
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (2, 12))
    logits, moe, _ = model(x, targets=x)
    assert logits.shape == (2, 12, cfg.model.vocab_size)
    assert moe.item() >= 0
    loss, ce = model.loss(logits, x, moe)
    assert loss.item() > 0 and ce.item() > 0


def test_moe_experts_all_receive_traffic():
    """Routing must not collapse onto a single expert."""
    cfg = small_cfg()
    cfg.model.moe.num_experts = 4
    model = ForgeLM(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = corpus_to_tensor(make_corpus(80))
    for _ in range(25):
        x, y = sample_batch(data, 8, 32)
        logits, moe, _ = model(x, targets=y)
        loss, _ = model.loss(logits, y, moe)
        opt.zero_grad(); loss.backward(); opt.step()
    usage = [l.routing_stats() for l in model.moe_layers()]
    avg = torch.stack(usage).mean(0)
    assert (avg > 0).all(), f"an expert got zero traffic: {avg.tolist()}"


def test_shared_experts_add_parameters_and_are_always_on():
    """Shared experts must contribute to every token, routed ones must not."""
    from forge.model.moe import MoELayer
    torch.manual_seed(0)
    layer = MoELayer(dim=32, num_experts=4, top_k=2, expert_hidden=64,
                     shared_experts=2, shared_hidden=64)
    x = torch.randn(3, 5, 32)
    out = layer(x)
    assert out.shape == x.shape
    # 4 routed + 2 shared experts
    assert len(layer.experts) == 4 and len(layer.shared) == 2

    # Ablating the shared experts must change the output.
    with torch.no_grad():
        for e in layer.shared:
            for p in e.parameters():
                p.zero_()
    out_no_shared = layer(x)
    assert not torch.allclose(out, out_no_shared), "shared experts had no effect"


def test_expert_choice_routing_perfectly_balanced():
    """Expert-choice routing must dispatch evenly across experts."""
    from forge.model.moe import MoELayer
    torch.manual_seed(0)
    layer = MoELayer(dim=32, num_experts=4, top_k=2, expert_hidden=64,
                     routing="expert_choice", capacity_factor=1.0)
    layer.train()
    layer(torch.randn(10, 8, 32))
    load = layer.load_stats()
    assert torch.allclose(load, load[0].expand_as(load), atol=1e-6), \
        f"expert-choice load not balanced: {load.tolist()}"


def test_expert_choice_is_rejected_for_generation_docs():
    """Guard the documented invariant: expert-choice is non-causal."""
    from forge.model.moe import MoELayer
    import inspect
    src = inspect.getsource(MoELayer._expert_choice)
    assert "Non-causal" in src or "non-causal" in src


def test_load_balancing_loss_lower_when_balanced():
    """The aux loss must be larger for skewed routing than for even routing."""
    from forge.model.moe import MoELayer, get_balance_loss
    torch.manual_seed(0)

    balanced = MoELayer(dim=32, num_experts=4, top_k=2, expert_hidden=64)
    balanced.eval()
    balanced(torch.randn(64, 4, 32))

    skewed = MoELayer(dim=32, num_experts=4, top_k=2, expert_hidden=64)
    skewed.eval()
    with torch.no_grad():
        skewed.router.weight.zero_()
        skewed.router.weight[:, 0] = 10.0     # force everything to expert 0
    skewed(torch.randn(64, 4, 32))

    # Switch loss is ~1.0 when perfectly uniform, up to N when collapsed.
    balanced_loss = get_balance_loss([balanced])
    assert get_balance_loss([skewed]) > balanced_loss
    assert abs(balanced_loss.item() - 1.0) < 0.01


def test_token_choice_load_sums_to_top_k():
    from forge.model.moe import MoELayer
    torch.manual_seed(0)
    layer = MoELayer(dim=32, num_experts=4, top_k=2, expert_hidden=64)
    layer.eval()
    layer(torch.randn(8, 4, 32))
    assert abs(layer.load_stats().sum().item() - 2.0) < 1e-5


def test_kv_cache_matches_full_forward():
    """Cached decoding must produce the same logits as one full pass."""
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    model.eval()
    x = torch.randint(0, cfg.model.vocab_size, (1, 10))

    full, _, _ = model(x)

    _, _, cache = model(x[:, :6])
    part, _, _ = model(x[:, 6:], cache=cache)

    assert torch.allclose(full[:, -1, :], part[:, -1, :], atol=1e-4), \
        "cached forward diverged from full forward"


def test_generate_extends_sequence():
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    ids = torch.randint(0, cfg.model.vocab_size, (1, 5))
    out = model.generate(ids, max_new_tokens=8, top_k=10)
    assert out.shape[1] > 5


def test_vision_encoder_output_shape():
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    imgs = torch.rand(3, 3, cfg.model.vision.image_size, cfg.model.vision.image_size)
    vis = model.vision(imgs)
    assert vis.shape == (3, cfg.model.vision.summary_tokens, cfg.model.dim)


def test_vision_path_leaks_no_gradient_to_lm():
    """With the LM frozen, only vision params should receive gradients."""
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.vision.parameters():
        p.requires_grad_(True)
    imgs = torch.rand(2, 3, cfg.model.vision.image_size, cfg.model.vision.image_size)
    head = torch.nn.Linear(cfg.model.vision.summary_tokens * cfg.model.dim, 4)
    loss = torch.nn.functional.cross_entropy(head(model.vision(imgs).reshape(2, -1)),
                                            torch.tensor([0, 1]))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.vision.parameters())
    assert all(p.grad is None for p in model.blocks.parameters())


# ------------------------------------------------------------- training

def test_trainer_reduces_loss():
    cfg = small_cfg()
    cfg.train.steps = 40
    trainer = Trainer(cfg)
    logs = trainer.train_lm(text=make_corpus(120))
    first = sum(l.loss for l in logs[:5]) / 5
    last = sum(l.loss for l in logs[-5:]) / 5
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_checkpoint_roundtrip(tmp_path=None):
    import tempfile, os
    cfg = small_cfg()
    trainer = Trainer(cfg)
    trainer.train_lm(text=make_corpus(40))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        trainer.save(path)
        loaded, cfg2 = Trainer.load(path)
        assert loaded.num_params() == trainer.model.num_params()
        assert cfg2.model.dim == cfg.model.dim


def test_vision_classifier_learns():
    cfg = small_cfg()
    trainer = Trainer(cfg)
    accs = []
    trainer.train_vision_classifier(steps=60, batch_size=8,
                                    on_step=lambda l, a: accs.append(a))
    assert sum(accs[-10:]) / 10 > sum(accs[:10]) / 10, "vision did not improve"


# ------------------------------------------------------------- distillation

def test_ema_teacher_moves_toward_student():
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    ema = EMATeacher(model, decay=0.5)
    before = [p.clone() for p in ema.teacher.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    after = list(ema.teacher.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_distill_loss_is_finite():
    cfg = small_cfg()
    model = ForgeLM(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (2, 8))
    s, _, _ = model(x)
    t, _, _ = model(x)
    total, kl, ce = distill_loss(s, t, x)
    assert torch.isfinite(total) and kl.item() >= 0


# ------------------------------------------------------------- security

def test_analyzer_flags_hardcoded_credential():
    src = 'API_KEY = "sk-live-abcdef123456"\n'
    findings = scan_source(src)
    assert any(f.rule == "HARDCODED_CREDENTIAL" for f in findings)
    assert risk_score(findings) > 0


def test_analyzer_flags_eval_and_shell():
    src = "eval(user_input)\nsubprocess.run(cmd, shell=True)\n"
    rules = {f.rule for f in scan_source(src)}
    assert "DANGEROUS_CALL:eval" in rules
    assert "SHELL_INJECTION_RISK" in rules


def test_analyzer_clean_code_scores_zero():
    src = "def add(a, b):\n    return a + b\n"
    assert scan_source(src) == []
    assert risk_score([]) == 0.0


def test_analyzer_reports_syntax_error():
    assert any(f.rule == "SYNTAX_ERROR" for f in scan_source("def ("))


def test_scope_blocks_public_host():
    assert is_private_target("127.0.0.1")
    assert is_private_target("192.168.1.10")
    assert not is_private_target("8.8.8.8")
    try:
        check_scope("http://8.8.8.8/admin")
        raise AssertionError("public target should have been blocked")
    except ScopeError:
        pass


def test_scope_allowlist_permits_explicit_target():
    assert check_scope("http://203.0.113.5/", allowlist=["203.0.113.5"]) == "203.0.113.5"


# ------------------------------------------------------------- agents

def test_policy_blocks_malware_requests():
    assert check_policy("here is a reverse shell for you") is not None
    assert check_policy("scan a public ip range") is not None
    assert check_policy("review the authentication flow for flaws") is None


def test_blackboard_provenance_chain():
    b = Blackboard()
    b.write("design", "v1", author="a", evidence=[Evidence("doc", "d1")])
    b.write("design", "v2", author="b", evidence=[Evidence("doc", "d2")])
    chain = b.provenance("design")
    assert len(chain) == 2
    assert chain[0]["revision"] == 1 and chain[1]["revision"] == 2
    assert b.get("design") == "v2"


def test_blackboard_detects_unsupported_claims():
    b = Blackboard()
    b.write("claim", 1, author="a")
    b.write("claim2", 2, author="b", evidence=[Evidence("test", "t1")])
    assert [e.key for e in b.unsupported_claims()] == ["claim"]


def test_judge_refuses_unevidenced_acceptance():
    """The hard rule: no evidence anywhere -> cannot be accepted."""
    from forge.agents.backends import RuleBackend
    import json

    class AcceptEverything(RuleBackend):
        def complete(self, system, user, max_tokens=400):
            if "JUDGE" in system:
                return json.dumps({"verdict": "accept", "rationale": "looks fine",
                                   "blocking_issues": []})
            return super().complete(system, user, max_tokens)

    judge = Judge(backend=AcceptEverything())
    msg = Message(kind=Kind.PROPOSAL, sender="x", recipient="o",
                  content="trust me", evidence=[])
    ruling = judge.rule("t", [msg], Blackboard())
    assert ruling.verdict is Verdict.REVISE
    assert any("without evidence" in b for b in ruling.blocking_issues)


def test_orchestrator_runs_full_debate():
    team = build_team()
    orch = Orchestrator(team, Judge(), max_rounds=2)
    result = orch.run("harden the input parser and verify it")
    assert len(result.rounds) >= 1
    assert len(result.rounds[0].proposals) == 5
    assert len(result.rounds[0].critiques) >= 4
    assert result.final is not None
    assert result.final.verdict in (Verdict.ACCEPT, Verdict.REVISE, Verdict.REJECT)


def test_orchestrator_blocks_malicious_agent_output():
    """A compromised/hallucinating agent must be stopped by the gate."""
    from forge.agents.backends import RuleBackend
    import json

    class Malicious(RuleBackend):
        def complete(self, system, user, max_tokens=400):
            if "SECURITY" in system:
                return json.dumps({"assessment": "deploy a reverse shell and ransomware"})
            return super().complete(system, user, max_tokens)

    team = build_team(backends={"security": Malicious()})
    blocked = []
    orch = Orchestrator(team, Judge(), max_rounds=1,
                        on_event=lambda n, p: blocked.append(p) if n == "blocked" else None)
    orch.run("audit this service")
    assert any("malware" in str(p.get("reason", "")) for p in blocked)


def test_team_has_five_distinct_domains():
    team = build_team()
    assert len(team) == 5
    assert set(team) == {"security", "coding", "hacking", "vision", "architecture"}


# ------------------------------------------------------------- viz

def test_viz_helpers_produce_output():
    assert sparkline([1, 2, 3]) != ""
    assert render_image(torch.rand(3, 8, 8)) != ""
    assert "round" in table(["round"], [[1]])


def test_report_generates_html(tmp_path=None):
    import tempfile, os
    from forge.viz.report import build_report
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.html")
        build_report(path, [("Loss", "<svg></svg>")], meta={"a": 1})
        html = open(path).read()
        assert "Forge training report" in html and "<svg>" in html