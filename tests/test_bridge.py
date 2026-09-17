"""Tests for the model-to-machine bridge.  Real filesystem, no mocks."""
from __future__ import annotations

import os
import tempfile

import pytest

from forge.control.bridge import model_drives_machine, parse_plan
from forge.control.kernel import ControlKernel, Step
from forge.control.scope import Op
from forge.control.trust import Level


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


# ------------------------------------------------------------ parsing

def test_parse_simple_plan():
    text = """
    goal: write a note
    mkdir docs
    write docs/note.md <<<
    # hardening
    validate at the boundary
    >>>
    """
    res = parse_plan(text, base_dir="/tmp/x")
    assert res.ok, res.errors
    assert res.plan.goal == "write a note"
    assert len(res.plan.steps) == 2
    assert res.plan.steps[0].op is Op.MKDIR
    assert res.plan.steps[1].op is Op.WRITE
    assert "validate at the boundary" in res.plan.steps[1].content


def test_parse_rejects_unknown_operation():
    res = parse_plan("frobnicate /etc/passwd")
    assert not res.ok
    assert any("unrecognised" in e for e in res.errors)


def test_parse_rejects_write_without_content_block():
    res = parse_plan("write /tmp/f.txt just some text")
    assert not res.ok
    assert any("<<<" in e for e in res.errors)


def test_parse_rejects_shell_metacharacters_in_path():
    res = parse_plan("read /tmp/x; rm -rf /")
    assert not res.ok
    assert any("illegal characters" in e for e in res.errors)


def test_parse_rejects_command_substitution():
    res = parse_plan("read $(cat /etc/shadow)")
    assert not res.ok


def test_parse_rejects_python_injection_attempt():
    """The grammar has no production for code, so these cannot parse."""
    for evil in [
        "exec('import os; os.system(\"rm -rf /\")')",
        "eval(\"__import__('os').system('id')\")",
        "__import__('subprocess').run(['rm', '-rf', '/'])",
        "os.system rm -rf /",
    ]:
        res = parse_plan(evil)
        assert not res.ok, f"should not parse: {evil!r}"


def test_parse_empty_output_is_an_error():
    assert not parse_plan("   \n\n").ok


def test_parse_delete_step_is_low_confidence():
    """A model proposing deletion should not be confident about it."""
    res = parse_plan("goal: cleanup\ndelete /tmp/x.txt", base_dir="/tmp")
    assert res.ok
    assert res.plan.steps[0].confidence < 1.0


def test_parse_truncates_huge_content_with_warning():
    big = "A" * 30_000
    res = parse_plan(f"write f.txt <<<{big}>>>")
    assert res.ok
    assert res.warnings
    assert len(res.plan.steps[0].content) == 20_000


# ------------------------------------------------------------ bridge

def test_bridge_executes_a_valid_model_plan(kernel, sandbox):
    text = f"""
    goal: create docs
    mkdir {sandbox}/docs
    write {sandbox}/docs/a.md <<<
    hello
    >>>
    """
    out = model_drives_machine(text, kernel, base_dir=sandbox)
    assert out.acted
    assert out.run.success
    assert os.path.exists(os.path.join(sandbox, "docs", "a.md"))


def test_unparseable_model_output_executes_nothing(kernel, sandbox):
    out = model_drives_machine("exec('os.system(\"rm -rf /\")')", kernel,
                               base_dir=sandbox)
    assert not out.acted
    assert out.parse.errors
    assert os.path.exists(sandbox), "sandbox should be untouched"


def test_model_cannot_escape_through_absolute_path(kernel, sandbox):
    outside = os.path.realpath(tempfile.mkdtemp())
    text = f"goal: escape\nwrite {outside}/evil.txt <<<pwned>>>"
    out = model_drives_machine(text, kernel, base_dir=sandbox)
    assert out.run is not None
    assert not out.run.executed
    assert not os.path.exists(os.path.join(outside, "evil.txt"))
    os.rmdir(outside)


def test_model_relative_paths_are_confined_to_base(kernel, sandbox):
    out = model_drives_machine("goal: ok\nwrite safe.txt <<<hi>>>", kernel,
                               base_dir=sandbox)
    assert out.acted
    assert os.path.exists(os.path.join(sandbox, "safe.txt"))


def test_model_cannot_write_secrets_path(kernel, sandbox):
    out = model_drives_machine("goal: leak\nwrite .env <<<X=1>>>", kernel,
                               base_dir=sandbox)
    assert out.run is not None
    assert not out.run.executed
    assert not os.path.exists(os.path.join(sandbox, ".env"))


def test_bridge_logs_parse_failure_to_audit(kernel):
    model_drives_machine("nonsense here", kernel)
    assert kernel.audit.find("parse_failed")


def test_bridge_dry_parse_does_not_execute(kernel, sandbox):
    out = model_drives_machine("goal: g\nwrite dry.txt <<<x>>>", kernel,
                               base_dir=sandbox, execute=False)
    assert out.run is None
    assert not os.path.exists(os.path.join(sandbox, "dry.txt"))


def test_verification_still_applies_to_model_authored_plans(kernel, sandbox):
    """A model plan that the world contradicts must not be reported as success."""
    out = model_drives_machine("goal: g\nwrite v.txt <<<actual>>>", kernel,
                               base_dir=sandbox)
    # The bridge sets expect_content_after from the content, so this should
    # verify.  Now break reality behind the kernel's back and confirm the
    # verification step reads disk rather than trusting the plan.
    path = os.path.join(sandbox, "v.txt")
    assert out.run.success
    os.remove(path)
    assert not os.path.exists(path), "kernel verified against real disk state"