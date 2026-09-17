"""Orchestrator: runs the debate rounds and enforces the safety gate.

Loop shape (this is the part that differs from a plain chain-of-agents):

    plan -> parallel proposals -> cross-critique -> security gate
         -> judge ruling -> (revise and repeat) or accept

The security gate is a *hard* filter outside the model's control.  A neural
judge can be talked out of a decision; a regex gate cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from forge.agents.blackboard import Blackboard
from forge.agents.protocol import Evidence, Kind, Message, Ruling, Verdict
from forge.agents.roles import Agent, Judge, PolicyViolation, check_policy


@dataclass
class Round:
    index: int
    proposals: list[Message] = field(default_factory=list)
    critiques: list[Message] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)
    ruling: Optional[Ruling] = None


@dataclass
class RunResult:
    task: str
    rounds: list[Round] = field(default_factory=list)
    final: Optional[Ruling] = None
    accepted: bool = False
    channel: Optional[object] = None       # ChannelResult, when a channel is wired

    def summary(self) -> dict:
        return {
            "task": self.task,
            "rounds": len(self.rounds),
            "accepted": self.accepted,
            "verdict": self.final.verdict.value if self.final else None,
            "blocked": [b for r in self.rounds for b in r.blocked],
            "rationale": self.final.rationale if self.final else None,
            "acted": getattr(self.channel, "acted", False),
            "action": self.channel.summary() if self.channel else None,
        }


class Orchestrator:
    def __init__(
        self,
        team: dict[str, Agent],
        judge: Judge,
        blackboard: Optional[Blackboard] = None,
        max_rounds: int = 3,
        on_event: Optional[Callable[[str, dict], None]] = None,
        channel: Optional[object] = None,
        act_on_accept: bool = True,
    ) -> None:
        self.team = team
        self.judge = judge
        self.board = blackboard or Blackboard()
        self.max_rounds = max_rounds
        self.on_event = on_event or (lambda name, payload: None)
        # Optional ActionChannel.  When absent the orchestrator debates and
        # stops, exactly as before -- wiring the machine in is opt-in.
        self.channel = channel
        self.act_on_accept = act_on_accept

    def _emit(self, name: str, **payload) -> None:
        self.on_event(name, payload)

    def run(self, task: str) -> RunResult:
        result = RunResult(task=task)
        self.board.write("task", task, author="orchestrator",
                         evidence=[Evidence("input", "user", task[:120])])
        self._emit("task_start", task=task)

        for r in range(1, self.max_rounds + 1):
            round_ = Round(index=r)
            self._emit("round_start", round=r)

            # 1. Independent proposals (each agent sees only the blackboard).
            for domain, agent in self.team.items():
                try:
                    msg = agent.propose(task, self.board)
                except PolicyViolation as exc:
                    round_.blocked.append((domain, str(exc)))
                    self._emit("blocked", domain=domain, reason=str(exc))
                    continue
                self.board.post(msg)
                round_.proposals.append(msg)
                self._emit("proposal", sender=msg.sender, content=msg.content[:200])

            # 2. Cross-critique: each agent reviews one other agent's work.
            names = list(self.team)
            for i, target in enumerate(round_.proposals):
                critic_domain = names[(i + 1) % len(names)]
                critic = self.team[critic_domain]
                if critic.name == target.sender:
                    continue
                try:
                    c = critic.critique(target, self.board)
                except PolicyViolation as exc:
                    round_.blocked.append((critic_domain, str(exc)))
                    continue
                self.board.post(c)
                round_.critiques.append(c)
                self._emit("critique", sender=c.sender, on=target.sender,
                           severity=c.confidence)

            # 3. Hard security gate over everything produced this round.
            for msg in round_.proposals + round_.critiques:
                reason = check_policy(msg.content)
                if reason:
                    round_.blocked.append((msg.sender, reason))
                    self._emit("blocked", domain=msg.sender, reason=reason)

            # 4. Judge rules.
            all_msgs = round_.proposals + round_.critiques
            ruling = self.judge.rule(task, all_msgs, self.board)
            round_.ruling = ruling
            self.board.write(
                f"round{r}_ruling", ruling.verdict.value, author="judge",
                evidence=[Evidence("ruling", "judge", ruling.rationale[:160])],
            )
            self._emit("ruling", round=r, verdict=ruling.verdict.value,
                       rationale=ruling.rationale)

            result.rounds.append(round_)

            if ruling.verdict is Verdict.ACCEPT:
                result.final = ruling
                result.accepted = True
                # Acceptance is the only point at which the machine may be
                # touched.  The gate above the machine is separate from the
                # gate inside it: a debate can be wrong, the kernel cannot be
                # talked out of a capability check.
                if self.channel is not None and self.act_on_accept:
                    try:
                        result.channel = self.channel.act(
                            task, ruling, blackboard=self.board
                        )
                        self._emit("action", **result.channel.summary())
                    except Exception as exc:          # never lose the debate result
                        self.board.write("action_error", str(exc),
                                         author="orchestrator")
                        self._emit("action_error", error=str(exc))
                break
            if ruling.verdict is Verdict.REJECT and r == self.max_rounds:
                result.final = ruling
                break
            # REVISE -> loop again with critiques now on the board.

        if result.final is None:
            result.final = self.judge.rule(task, self.board.messages(), self.board)

        self._emit("done", **result.summary())
        return result