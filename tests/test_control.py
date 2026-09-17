"""Control kernel test suite.  Real filesystem, real code paths, no mocks."""
from __future__ import annotations

import os
import tempfile

import pytest

from forge.control.actions import ALLOWED_COMMANDS, ActionLedger, Transaction
from forge.control.audit import GENESIS, AuditChain
from forge.control.kernel import ControlKernel, Plan, Step
from forge.control.scope import (
    CapabilitySet, Op, PathGuard, ScopeDenial, default_policy,
)
from forge.control.trust import Level, TrustLadder
from forge.control.world import WorldDigest


@pytest.fixture()
def sandbox():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.realpath(d)


@pytest.fixture()
def kernel(sandbox):
    """Kernel whose operator approves everything, for happy-path tests."""
    k = ControlKernel([sandbox], approve=lambda decision, step: True,
                      audit_path=os.path.join(sandbox, "audit.log"))
    for op in (Op.WRITE, Op.MKDIR, Op.DELETE, Op.APPEND, Op.RUN_TESTS):
        k.trust.grant_level(op, Level.AUTO)
    return k


# ------------------------------------------------------------ path guard

def test_pathguard_rejects_traversal(sandbox):
    guard = PathGuard([sandbox])
    with pytest.raises(ScopeDenial):
        guard.resolve(os.path.join(sandbox, "..", "..", "etc", "passwd"))


def test_pathguard_rejects_symlink_escape(sandbox):
    """The classic bypass: a symlink pointing outside the sandbox."""
    outside = os.path.realpath(tempfile.mkdtemp())
    link = os.path.join(sandbox, "escape")
    os.symlink(outside, link)
    guard = PathGuard([sandbox])
    with pytest.raises(ScopeDenial) as exc:
        guard.resolve(os.path.join(link, "victim"))
    assert "escapes the sandbox" in str(exc.value)
    os.rmdir(outside)


def test_pathguard_rejects_null_byte(sandbox):
    guard = PathGuard([sandbox])
    with pytest.raises(ScopeDenial):
        guard.resolve(sandbox + "/x\x00y")


def test_pathguard_accepts_inside(sandbox):
    guard = PathGuard([sandbox])
    assert guard.contains(os.path.join(sandbox, "a", "b.txt"))
    assert guard.contains(sandbox)


# ------------------------------------------------------------ capabilities

def test_empty_capability_set_denies_everything(sandbox):
    guard = PathGuard([sandbox])
    caps = CapabilitySet(guard)          # no grants at all
    for op in (Op.READ, Op.WRITE, Op.DELETE, Op.MKDIR):
        with pytest.raises(ScopeDenial):
            caps.check(op, os.path.join(sandbox, "f"))


def test_capability_expiry_denies(sandbox):
    guard = PathGuard([sandbox])
    caps = CapabilitySet(guard)
    caps.grant(Op.WRITE, [sandbox], "op", ttl_seconds=-1)   # already expired
    with pytest.raises(ScopeDenial) as exc:
        caps.check(Op.WRITE, os.path.join(sandbox, "f"))
    assert "expired" in str(exc.value)


def test_capability_budget_exhausts(sandbox):
    guard = PathGuard([sandbox])
    caps = CapabilitySet(guard)
    cap = caps.grant(Op.WRITE, [sandbox], "op", ttl_seconds=60, budget=2)
    target = os.path.join(sandbox, "f")
    caps.check(Op.WRITE, target); caps.spend(cap)
    caps.check(Op.WRITE, target); caps.spend(cap)
    with pytest.raises(ScopeDenial):
        caps.check(Op.WRITE, target)


def test_denial_carries_constructive_suggestion(sandbox):
    guard = PathGuard([sandbox])
    caps = CapabilitySet(guard)
    with pytest.raises(ScopeDenial) as exc:
        caps.check(Op.WRITE, os.path.join(sandbox, "f"))
    assert exc.value.suggestion, "a denial must suggest a permitted alternative"


def test_revoke_removes_capability(sandbox):
    guard = PathGuard([sandbox])
    caps = CapabilitySet(guard)
    cap = caps.grant(Op.WRITE, [sandbox], "op")
    assert caps.check(Op.WRITE, os.path.join(sandbox, "f")).id == cap.id
    assert caps.revoke(cap.id) is True
    with pytest.raises(ScopeDenial):
        caps.check(Op.WRITE, os.path.join(sandbox, "f"))


def test_capability_scoped_to_subdir_not_parent(sandbox):
    guard = PathGuard([sandbox])
    caps = CapabilitySet(guard)
    sub = os.path.join(sandbox, "sub")
    os.makedirs(sub)
    caps.grant(Op.WRITE, [sub], "op")
    caps.check(Op.WRITE, os.path.join(sub, "ok.txt"))         # allowed
    with pytest.raises(ScopeDenial):
        caps.check(Op.WRITE, os.path.join(sandbox, "outside.txt"))


# ------------------------------------------------------------ audit chain

def test_audit_chain_grows_and_verifies():
    chain = AuditChain()
    for i in range(5):
        chain.append("agent", f"act{i}", {"i": i})
    result = chain.verify()
    assert result.ok and result.n_records == 5
    assert chain.head() != GENESIS


def test_audit_detects_content_tampering():
    chain = AuditChain()
    for i in range(4):
        chain.append("agent", f"act{i}", {"i": i})
    chain.tamper(2, "act2_forged")
    result = chain.verify()
    assert not result.ok
    assert result.first_bad_index == 2
    assert "modified" in result.reason


def test_audit_detects_link_break():
    chain = AuditChain()
    for i in range(4):
        chain.append("agent", f"act{i}")
    chain.records()[3].prev_hash = "f" * 64     # sever the chain
    result = chain.verify()
    assert not result.ok
    assert result.first_bad_index == 3


def test_audit_persists_and_reloads(sandbox):
    path = os.path.join(sandbox, "a.log")
    c1 = AuditChain(path)
    c1.append("agent", "one", {"x": 1})
    c1.append("agent", "two", {"x": 2})
    c2 = AuditChain(path)
    assert c2.n() == 2
    assert c2.verify().ok
    assert c2.head() == c1.head()


# ------------------------------------------------------------ ledger / undo

def test_write_then_undo_restores_original(sandbox):
    guard = PathGuard([sandbox])
    caps = default_policy(guard)
    ledger = ActionLedger(caps, guard)
    target = os.path.join(sandbox, "f.txt")

    ledger.write(target, "original")
    ledger.write(target, "changed")
    assert open(target).read() == "changed"

    ledger.undo_last()
    assert open(target).read() == "original", "undo did not restore content"


def test_write_new_file_undo_removes_it(sandbox):
    guard = PathGuard([sandbox])
    ledger = ActionLedger(caps := default_policy(guard), guard)
    target = os.path.join(sandbox, "new.txt")
    ledger.write(target, "hi")
    assert os.path.exists(target)
    ledger.undo_last()
    assert not os.path.exists(target)


def test_delete_quarantines_never_erases(sandbox):
    guard = PathGuard([sandbox])
    ledger = ActionLedger(default_policy(guard), guard)
    target = os.path.join(sandbox, "keepme.txt")
    ledger.write(target, "precious data")
    res = ledger.delete(target)

    assert res.ok
    assert not os.path.exists(target), "file still in workspace"
    assert os.path.exists(str(res.value)), "bytes were not retained"

    # And it comes back intact.
    ledger.undo_last()
    assert os.path.exists(target)
    assert open(target).read() == "precious data"


def test_append_undo_truncates_exactly(sandbox):
    guard = PathGuard([sandbox])
    ledger = ActionLedger(default_policy(guard), guard)
    target = os.path.join(sandbox, "log.txt")
    ledger.write(target, "base")
    ledger.append(target, "-extra")
    assert open(target).read() == "base-extra"
    ledger.undo_last()
    assert open(target).read() == "base"


def test_transaction_rolls_back_all_steps(sandbox):
    guard = PathGuard([sandbox])
    ledger = ActionLedger(default_policy(guard), guard)
    a = os.path.join(sandbox, "a.txt")
    b = os.path.join(sandbox, "b.txt")

    with pytest.raises(RuntimeError):
        with Transaction(ledger, "multi") as _:
            ledger.write(a, "1")
            ledger.write(b, "2")
            raise RuntimeError("third step failed")

    assert not os.path.exists(a), "step 1 not rolled back"
    assert not os.path.exists(b), "step 2 not rolled back"


def test_run_tests_rejects_unlisted_command(sandbox):
    guard = PathGuard([sandbox])
    ledger = ActionLedger(default_policy(guard), guard)
    with pytest.raises(ScopeDenial) as exc:
        ledger.run_tests("rm -rf /")
    assert "not allowlisted" in str(exc.value)


def test_allowed_commands_are_shell_free_and_immutable():
    """The real guarantee is not character-counting: it is that no shell is
    involved and the vectors are constants the caller cannot influence."""
    import inspect
    from forge.control import actions

    for group in ALLOWED_COMMANDS.values():
        for argv in group:
            assert isinstance(argv, list)
            assert all(isinstance(t, str) for t in argv)
            assert argv[0].startswith("python"), "only the interpreter is invoked"

    # shell=False must be passed, and the argv comes from the constant.
    src = inspect.getsource(actions.ActionLedger.run_tests)
    assert "shell=False" in src
    assert "ALLOWED_COMMANDS[command]" in src


def test_run_tests_rejects_arbitrary_callable_and_garbage(sandbox):
    guard = PathGuard([sandbox])
    ledger = ActionLedger(default_policy(guard), guard)
    for bad in ("rm -rf /", "", "python3 -c 'print(1)'", "run_tests; rm -rf /"):
        with pytest.raises(ScopeDenial):
            ledger.run_tests(bad)


# ------------------------------------------------------------ trust ladder

def test_trust_starts_at_dry_run_and_denies():
    t = TrustLadder()
    d = t.decide(Op.WRITE, confidence=1.0)
    assert not d.allow and d.level is Level.DRY_RUN


def test_trust_promotes_only_after_consecutive_successes():
    t = TrustLadder(promote_after=3)
    for _ in range(2):
        t.observe(Op.WRITE, success=True)
    assert t.record_for(Op.WRITE).level is Level.DRY_RUN
    t.observe(Op.WRITE, success=True)
    assert t.record_for(Op.WRITE).level is Level.APPROVAL
    for _ in range(3):
        t.observe(Op.WRITE, success=True)
    assert t.record_for(Op.WRITE).level is Level.AUTO


def test_trust_demotes_immediately_on_failure():
    t = TrustLadder(promote_after=1)
    t.observe(Op.WRITE, success=True)
    t.observe(Op.WRITE, success=True)
    assert t.record_for(Op.WRITE).level is Level.AUTO
    t.observe(Op.WRITE, success=False)
    assert t.record_for(Op.WRITE).level is Level.DRY_RUN, \
        "trust must be slow to earn and instant to lose"


def test_low_confidence_escalates_even_at_auto():
    t = TrustLadder(confidence_floor=0.6)
    t.grant_level(Op.WRITE, Level.AUTO)
    d = t.decide(Op.WRITE, confidence=0.3)
    assert d.needs_approval and d.level is Level.APPROVAL
    assert "confidence" in d.reason


def test_trust_levels_are_independent_per_op():
    t = TrustLadder(promote_after=1)
    t.observe(Op.WRITE, success=True)
    assert t.record_for(Op.WRITE).level is Level.APPROVAL
    assert t.record_for(Op.DELETE).level is Level.DRY_RUN


# ------------------------------------------------------------ world digest

def test_world_digest_respects_depth_and_reports_truncation(sandbox):
    deep = os.path.join(sandbox, "a", "b", "c", "d", "e")
    os.makedirs(deep)
    open(os.path.join(deep, "f.txt"), "w").write("x")

    digest = WorldDigest([sandbox], max_depth=2)
    digest.build()
    stats = digest.stats()
    assert stats["nodes"] > 0
    assert stats["truncated_dirs"], "deep tree should report truncation"


def test_world_digest_skips_caches(sandbox):
    os.makedirs(os.path.join(sandbox, "__pycache__"))
    os.makedirs(os.path.join(sandbox, "src"))
    text = WorldDigest([sandbox]).summary()
    assert "__pycache__" not in text
    assert "src" in text


def test_world_digest_flags_secrets_by_name(sandbox):
    open(os.path.join(sandbox, ".env"), "w").write("X=1")
    text = WorldDigest([sandbox]).summary()
    assert "secrets" in text.lower()


# ------------------------------------------------------------ kernel

def test_kernel_full_cycle_creates_verifies_and_audits(kernel, sandbox):
    target = os.path.join(sandbox, "hello.txt")
    plan = kernel.plan("write a greeting", [
        Step(Op.WRITE, target, content="hello world",
             expect_absent_before=True, expect_content_after="hello world"),
    ])
    outcome = kernel.execute(plan)

    assert outcome.success, f"run failed: {outcome.summary()}"
    assert open(target).read() == "hello world"
    assert outcome.verification.ok
    assert kernel.audit.n() >= 4


def test_kernel_verifies_against_disk_not_the_claim(kernel, sandbox, monkeypatch):
    """If the world disagrees with the claim, the step must not count."""
    target = os.path.join(sandbox, "x.txt")
    plan = kernel.plan("write", [
        Step(Op.WRITE, target, content="actual",
             expect_content_after="a lie about the content"),
    ])
    outcome = kernel.execute(plan)
    assert outcome.executed and not outcome.executed[0].verified
    assert not outcome.success


def test_kernel_simulation_blocks_out_of_scope_before_acting(kernel, sandbox):
    outside = os.path.realpath(tempfile.mkdtemp())
    plan = kernel.plan("escape", [Step(Op.WRITE, os.path.join(outside, "evil.txt"),
                                       content="x")])
    outcome = kernel.execute(plan)
    assert outcome.skipped and not outcome.executed
    assert not os.path.exists(os.path.join(outside, "evil.txt"))
    os.rmdir(outside)


def test_kernel_simulation_blocks_secrets_paths(kernel, sandbox):
    plan = kernel.plan("steal env", [
        Step(Op.WRITE, os.path.join(sandbox, ".env"), content="LEAKED=1"),
    ])
    outcome = kernel.execute(plan)
    assert outcome.skipped and not outcome.executed
    assert not os.path.exists(os.path.join(sandbox, ".env"))


def test_kernel_without_approval_skips_untrusted_steps(sandbox):
    """A kernel with no human attached must not self-approve."""
    k = ControlKernel([sandbox])       # default approve = deny
    target = os.path.join(sandbox, "nope.txt")
    plan = k.plan("try", [Step(Op.WRITE, target, content="x")])
    outcome = k.execute(plan, simulate_first=False)
    assert outcome.skipped, "untrusted step should have required approval"
    assert not os.path.exists(target)


def test_kernel_delete_verified_as_quarantined(kernel, sandbox):
    target = os.path.join(sandbox, "gone.txt")
    kernel.ledger.write(target, "data")
    plan = kernel.plan("remove", [Step(Op.DELETE, target)])
    outcome = kernel.execute(plan)
    assert outcome.success
    assert not os.path.exists(target)


def test_kernel_rollback_restores_plan_effects(kernel, sandbox):
    a = os.path.join(sandbox, "r1.txt")
    b = os.path.join(sandbox, "r2.txt")
    plan = kernel.plan("two writes", [
        Step(Op.WRITE, a, content="A"),
        Step(Op.WRITE, b, content="B"),
    ])
    kernel.execute(plan)
    assert os.path.exists(a) and os.path.exists(b)

    undone = kernel.undo_all()
    assert undone >= 2
    assert not os.path.exists(a) and not os.path.exists(b)


def test_kernel_revoke_everything_is_available_and_logged(kernel, sandbox):
    n = kernel.revoke_everything()
    assert n > 0
    assert kernel.audit.find("revoke_all")
    plan = kernel.plan("after revoke", [
        Step(Op.WRITE, os.path.join(sandbox, "z.txt"), content="x")])
    outcome = kernel.execute(plan)
    assert outcome.skipped


def test_kernel_status_reports_audit_integrity(kernel):
    status = kernel.status()
    assert status["audit_ok"] is True
    assert status["roots"]
    assert isinstance(status["trust"], list)


def test_kernel_plan_refusal_is_audited(kernel, sandbox):
    plan = kernel.plan("bad", [Step(Op.WRITE, os.path.join(sandbox, ".env"),
                                    content="x")])
    kernel.execute(plan)
    assert kernel.audit.find("plan_refused")


def test_trust_cannot_bootstrap_itself(sandbox):
    """A class at DRY_RUN never executes, so it can never promote itself.

    This is the bootstrap problem, and it is intentional: an agent must not be
    able to grant itself permission by repeatedly trying.
    """
    k = ControlKernel([sandbox], approve=lambda d, s: True)
    for i in range(10):
        k.execute(k.plan(f"attempt{i}", [
            Step(Op.WRITE, os.path.join(sandbox, f"n{i}.txt"), content="x")]),
            simulate_first=False)
    rec = k.trust.record_for(Op.WRITE)
    assert rec.level is Level.DRY_RUN
    assert rec.successes == 0, "nothing should have executed"


def test_trust_promotes_over_real_kernel_runs(sandbox):
    """End-to-end: seeded by a human, then autonomy is genuinely earned."""
    k = ControlKernel([sandbox], approve=lambda d, s: True)
    k.trust.seed(Op.WRITE, Level.APPROVAL)         # the human's decision

    for i in range(4):                              # promote_after default is 3
        k.execute(k.plan(f"run{i}", [
            Step(Op.WRITE, os.path.join(sandbox, f"f{i}.txt"), content="x",
                 expect_content_after="x")]),
            simulate_first=False)

    assert k.trust.record_for(Op.WRITE).level is Level.AUTO, \
        "verified successes should have promoted write to auto"


def test_trust_demotes_after_a_real_verification_failure(sandbox):
    """A step the world contradicts must cost the class its autonomy."""
    k = ControlKernel([sandbox], approve=lambda d, s: True)
    k.trust.seed(Op.WRITE, Level.AUTO)

    k.execute(k.plan("lie", [
        Step(Op.WRITE, os.path.join(sandbox, "x.txt"), content="actual",
             expect_content_after="not what was written")]),
        simulate_first=False)

    assert k.trust.record_for(Op.WRITE).level is Level.DRY_RUN, \
        "a failed verification must demote the class immediately"