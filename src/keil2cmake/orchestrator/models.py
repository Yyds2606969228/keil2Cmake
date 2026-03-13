from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ArtifactRecord:
    kind: str
    path: str
    role: str
    producer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommandExecution:
    command: list[str]
    cwd: str
    executed: bool
    success: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowGoal:
    name: str
    desired_outcome: str
    success_criteria: list[str]
    constraints: list[str] = field(default_factory=list)
    owner: str = 'software-loop-orchestrator'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowSignal:
    source: str
    summary: str
    details: str = ""
    severity: str = 'info'
    related_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProposedAction:
    title: str
    description: str
    tool_hint: str | None = None
    risk: str = 'low'
    expected_outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DirectionDecision:
    workflow_phase: str
    focus_domain: str
    summary: str
    reasoning: str
    next_action: str
    handoff_skill: str | None = None
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentWorkItem:
    owner_action: str
    workflow_phase: str
    focus_domain: str
    goal: str
    summary: str
    diagnostics: str
    constraints: list[str]
    candidate_files: dict[str, str] = field(default_factory=dict)
    suggested_actions: list[ProposedAction] = field(default_factory=list)
    validation_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRepairResult:
    applied: bool
    summary: str
    files_touched: list[str] = field(default_factory=list)
    strategy: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowCheckpoint:
    workflow_phase: str
    focus_domain: str
    summary: str
    status: str
    next_action: str | None = None
    handoff_skill: str | None = None
    at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
