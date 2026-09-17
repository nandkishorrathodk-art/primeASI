"""Pipeline tests: model -> agents -> judge -> channel -> kernel -> disk.

Real filesystem, real checkpoints, no mocks anywhere.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from forge.agents.act import ActionChannel, GRAMMAR_PROMPT
from forge.agents.backends import CleanBackend, MalformedBackend, RuleBackend
from forge.agents.local import build_team_backends, resolve_backend
from forge.agents.orchestrator import Orchestrator
from forge.agents.protocol import Evidence, Kind, Message, Ruling, Verdict
from forge.agents.roles import Judge, build_team
from forge.config import ForgeConfig
from forge.control.kernel import ControlKernel, Plan, Step
from forge.control.scope import Op
from forge.control.trust import Level
from forge.data import make_corpus
from forge.training.trainer import Trainer


@pytest.fixture()
def sandbox():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.realpath(d)


@pytest.fixture()
def kernel(sandbox):
    k = ControlKernel([sandbox], approve=lambda d, s: True)
    for op in (Op.WRITE, Op.MKDIR, Op.DELETE, Op.APPEND):
        k.trust.seed(op, Level.AUTO)
    return k


def accept_ruling() -> Ruling:
    return Ruling(Verdict.ACCEPT, "evidenced work", [], 0.8)


# ============================================================ backend wiring

def test_resolve_backend_local_falls_back_without_checkpoint(sandbox):
    """A missing checkpoint must not take down the loop."""
    b = resolve_backend("local", checkpoint=os.path.join(sandbox, "nope.pt"))
    assert isinstance(b, RuleBackend)


def test_resolve_backend_local_uses_real_checkpoint(sandbox):
    cfg = ForgeConfig()
    cfg.model.dim = 32
    cfg.model.n_layers = 1
    cfg.model.moe.num_experts = 2
    cfg.model.moe.expert_hidden = 32
    cfg.model.vision.image_size = 16
    cfg.model.vision.patch_size = 8
    cfg.train.steps = 3
    cfg.train.batch_size = 2
    cfg.train.seq_len = 16

    trainer = Trainer(cfg)
    trainer.train_lm(text=make_corpus(20))
    ckpt = os.path.join(sandbox, "m.pt")
    trainer.save(ckpt)

    backend = resolve_backend("local", checkpoint=ckpt)
    assert backend.name == "local"
    out = backend.complete("system", "user", max_tokens=8)
    assert isinstance(out, str) and len(out) > 0


def test_build_team_backends_covers_all_domains():
    backends = build_team_backends("rules")
    assert set(backends) == {"security", "coding", "hacking", "vision", "architecture"}


# ============================================================ the channel

def test_channel_refuses_non_accept_ruling(kernel, sandbox):
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    for verdict in (Verdict.REVISE, Verdict.REJECT):
        res = ch.act("do a thing", Ruling(verdict, "not ready", [], 0.5))
        assert not res.acted, f"{verdict.value} must not be allowed to act"
    assert kernel.ledger.history() == [], "nothing should have touched the machine"


def test_channel_acts_on_accept_and_verifies(kernel, sandbox):
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    res = ch.act("document the boundary", accept_ruling())

    assert res.acted
    assert res.steps_executed >= 1
    assert res.verified == res.steps_executed
    assert not res.used_fallback
    assert res.model_parsed
    assert not res.rolled_back


def test_channel_falls_back_visibly_when_model_output_is_garbage(kernel, sandbox):
    ch = ActionChannel(kernel, sandbox, backend=MalformedBackend())
    res = ch.act("tidy the docs", accept_ruling())

    assert res.used_fallback, "unparseable model output should trigger fallback"
    assert not res.model_parsed
    assert kernel.audit.find("fallback"), "fallback must be recorded, not silent"
    assert res.acted
    # The fallback is deterministic local rules, so it must not touch a model.
    assert "docs/tidy-the-docs.md" in res.model_text


def test_channel_abandons_when_neither_model_nor_fallback_parses(kernel, sandbox):
    """If even the deterministic fallback cannot produce a plan, nothing runs."""
    ch = ActionChannel(
        kernel, sandbox,
        backend=MalformedBackend(),
        fallback_plan=lambda task: "this is not grammar either",
    )
    res = ch.act("hopeless", accept_ruling())
    assert not res.acted
    assert kernel.audit.find("abandoned")
    assert kernel.ledger.history() == []


def test_channel_records_result_on_the_blackboard(kernel, sandbox):
    from forge.agents.blackboard import Blackboard

    board = Blackboard()
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    ch.act("record it", accept_ruling(), blackboard=board)
    assert "action_result" in board.keys()
    entry = board.entry("action_result")
    assert entry.evidence, "the action result must carry evidence"


def test_channel_respects_max_steps(kernel, sandbox):
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend(), max_steps=1)
    res = ch.act("only one step", accept_ruling())
    assert res.steps_executed <= 1


def test_channel_cannot_escape_the_sandbox(kernel, sandbox):
    """Even a plan naming an absolute path outside must not land."""
    outside = os.path.realpath(tempfile.mkdtemp())

    class Escaping(CleanBackend):
        def complete(self, system, user, max_tokens=160):
            if system == GRAMMAR_PROMPT:
                return (f"goal: escape\n"
                        f"write {outside}/pwned.txt <<<gotcha>>>")
            return super().complete(system, user, max_tokens)

    ch = ActionChannel(kernel, sandbox, backend=Escaping())
    res = ch.act("try to escape", accept_ruling())
    assert not os.path.exists(os.path.join(outside, "pwned.txt"))
    assert not res.verified or res.rolled_back or not res.acted
    os.rmdir(outside)


# ============================================================ orchestrator wiring

def test_orchestrator_without_channel_does_not_touch_machine(sandbox):
    orch = Orchestrator(build_team(), Judge(), max_rounds=1)
    result = orch.run("audit the service")
    assert result.channel is None
    assert not os.path.exists(os.path.join(sandbox, "docs"))


def test_orchestrator_with_channel_acts_only_after_accept(sandbox, kernel):
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    orch = Orchestrator(build_team(), Judge(), max_rounds=3, channel=ch)
    result = orch.run("harden the input parser")

    if result.accepted:
        assert result.channel is not None
        assert result.summary()["acted"] is True
    else:
        assert result.channel is None
        assert kernel.ledger.history() == [], \
            "a non-accepted run must leave the machine untouched"


def test_orchestrator_act_on_accept_false_is_respected(sandbox, kernel):
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    orch = Orchestrator(build_team(), Judge(), max_rounds=3, channel=ch,
                        act_on_accept=False)
    result = orch.run("harden the input parser")
    assert result.channel is None
    assert kernel.ledger.history() == []


def test_orchestrator_reports_action_in_summary(kernel, sandbox):
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    orch = Orchestrator(build_team(), Judge(), max_rounds=3, channel=ch)
    result = orch.run("document the trust boundary")
    summary = result.summary()
    assert "acted" in summary
    if summary["acted"]:
        assert summary["action"]["executed"] >= 1


def test_rule_judge_is_not_a_constant():
    """The rules judge must actually deliberate, or the channel is dead code."""
    b = RuleBackend()
    clean = b.complete("You are the JUDGE.", "transcript: x->orchestrator proposal")
    dirty = b.complete("You are the JUDGE.",
                       'theatre: severity: high finding from critic '
                       'x->orchestrator proposal')
    assert '"accept"' in clean
    assert '"revise"' in dirty


def test_accepted_debate_cannot_reach_secrets_even_when_judge_approves(sandbox, kernel):
    """The judge approves the *work*; it does not approve the *target*.

    A debate can be entirely well-formed and still name a secrets file.  The
    kernel must refuse that on its own authority, with no involvement from the
    judge at all.
    """
    from forge.agents.backends import ScriptedBackend

    class WantsSecrets(ScriptedBackend):
        def __init__(self):
            super().__init__("wants-secrets",
                             "goal: leak\nwrite .env <<<API_KEY=stolen>>>")

    ch = ActionChannel(kernel, sandbox, backend=WantsSecrets())
    res = ch.act("record the env contents", accept_ruling())

    assert not os.path.exists(os.path.join(sandbox, ".env"))
    assert not res.verified or not res.acted or res.rolled_back


def test_accepted_debate_cannot_use_shell_even_when_judge_approves(sandbox, kernel):
    from forge.agents.backends import ScriptedBackend

    class Shellish(ScriptedBackend):
        def __init__(self):
            super().__init__("shellish",
                             "goal: run\nrun_tests rm -rf / --no-preserve-root")

    ch = ActionChannel(kernel, sandbox, backend=Shellish())
    res = ch.act("run the tests", accept_ruling())
    # The grammar layer must reject it before the ledger ever sees it.  This
    # was a real gap: run_tests accepted arbitrary argument text, so
    # "rm -rf /" parsed as a command name.  The ledger refused it later, but
    # a defence layer that only the inner layer enforces is a weaker design.
    assert not res.model_parsed, "a non-allowlisted command must not parse"
    assert res.used_fallback
    assert kernel.audit.find("fallback"), "the rejection must be recorded"
    assert os.path.exists(sandbox), "sandbox must be intact"

    # And the ledger layer still refuses it independently, if called directly.
    from forge.control.scope import ScopeDenial
    with pytest.raises(ScopeDenial):
        kernel.ledger.run_tests("rm -rf /")


def test_full_chain_is_auditable_end_to_end(kernel, sandbox):
    """Every stage of the chain must leave a trace in the same audit log."""
    ch = ActionChannel(kernel, sandbox, backend=CleanBackend())
    orch = Orchestrator(build_team(), Judge(), max_rounds=3, channel=ch)
    result = orch.run("harden the input parser")

    log = kernel.audit
    assert log.verify().ok, "audit chain must be intact after a full run"
    assert log.n() > 0
    if result.accepted and result.channel and result.channel.acted:
        assert log.find("channel"), "channel activity must be logged"
        assert log.find("action:write") or log.find("action:mkdir")


def test_auto_rollback_does_not_undo_earlier_legitimate_work(kernel, sandbox):
    """Regression: rollback once unwound the whole ledger, not just this run."""
    good = os.path.join(sandbox, "keep.txt")
    kernel.execute(kernel.plan("good work", [
        Step(Op.WRITE, good, content="keep me", expect_content_after="keep me"),
    ]), simulate_first=False)
    assert open(good).read() == "keep me"

    bad = os.path.join(sandbox, "bad.txt")
    kernel.execute(kernel.plan("bad work", [
        Step(Op.WRITE, bad, content="x", expect_content_after="mismatch"),
    ]), simulate_first=False)

    assert os.path.exists(good), "earlier legitimate work was rolled back"
    assert open(good).read() == "keep me"
    assert not os.path.exists(bad), "failed run left residue"


# ============================================================ auto-rollback

def test_kernel_auto_rolls_back_on_verification_failure(kernel, sandbox):
    """A plan that mutates but fails verification must not leave residue."""
    target = os.path.join(sandbox, "residue.txt")
    plan = kernel.plan("lie about content", [
        Step(Op.WRITE, target, content="actual bytes",
             expect_content_after="a claim that is false"),
    ])
    outcome = kernel.execute(plan, simulate_first=False)

    assert outcome.failed_verifications
    assert outcome.rolled_back >= 1
    assert not os.path.exists(target), "failed plan left residue behind"
    assert not outcome.success
    assert kernel.audit.find("auto_rollback")


def test_kernel_no_rollback_when_everything_verifies(kernel, sandbox):
    target = os.path.join(sandbox, "clean.txt")
    plan = kernel.plan("honest", [
        Step(Op.WRITE, target, content="ok", expect_content_after="ok"),
    ])
    outcome = kernel.execute(plan, simulate_first=False)
    assert outcome.success
    assert outcome.rolled_back == 0
    assert os.path.exists(target)


def test_kernel_rollback_restores_prior_content(kernel, sandbox):
    """Overwriting then failing must restore the original bytes, not delete."""
    target = os.path.join(sandbox, "existing.txt")
    kernel.ledger.write(target, "ORIGINAL")

    plan = kernel.plan("bad overwrite", [
        Step(Op.WRITE, target, content="REPLACED",
             expect_content_after="neither of these"),
    ])
    kernel.execute(plan, simulate_first=False)
    assert open(target).read() == "ORIGINAL", "rollback did not restore originals"