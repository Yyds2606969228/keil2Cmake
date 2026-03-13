from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import AgentRepairResult, AgentWorkItem, DirectionDecision, ProposedAction, WorkflowCheckpoint, WorkflowGoal, WorkflowSignal, utc_now_iso


@dataclass(slots=True)
class OrchestratorState:
    project_root: str = ""
    uvprojx_path: str = ""
    project_name: str = ""
    workflow_phase: str = 'idle'
    phase_status: str = 'idle'
    focus_domain: str = 'unknown'
    artifact_consistency: str = "unknown"
    build_preset: str = "build"
    configure_preset: str = "keil2cmake"
    last_error: str | None = None
    debug_ready: bool = False
    current_goal: WorkflowGoal | None = None
    current_signal: WorkflowSignal | None = None
    active_work_item: AgentWorkItem | None = None
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    planned_actions: list[ProposedAction] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    pending_action: str | None = None
    handoff_skill: str | None = None
    agent_iterations: list[AgentRepairResult] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    history: list[WorkflowCheckpoint] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    def transition(
        self,
        workflow_phase: str,
        focus_domain: str,
        summary: str,
        *,
        status: str,
        artifact_consistency: str | None = None,
        decision: DirectionDecision | None = None,
        goal: WorkflowGoal | None = None,
        signal: WorkflowSignal | None = None,
        planned_actions: list[ProposedAction] | None = None,
        work_item: AgentWorkItem | None = None,
        agent_iteration: AgentRepairResult | None = None,
        success_criteria: list[str] | None = None,
        constraints: list[str] | None = None,
        completed_action: str | None = None,
        **context: Any,
    ) -> None:
        self.workflow_phase = workflow_phase
        self.phase_status = status
        self.focus_domain = focus_domain
        self.updated_at = utc_now_iso()
        if artifact_consistency is not None:
            self.artifact_consistency = artifact_consistency
        if decision is not None:
            self.pending_action = decision.next_action
            self.handoff_skill = decision.handoff_skill
            self.success_criteria = list(decision.success_criteria)
        if goal is not None:
            self.current_goal = goal
            self.success_criteria = list(goal.success_criteria)
            self.constraints = list(goal.constraints)
        if signal is not None:
            self.current_signal = signal
            if signal.severity == 'error':
                self.last_error = signal.details or signal.summary
        if planned_actions is not None:
            self.planned_actions = list(planned_actions)
        if work_item is not None:
            self.active_work_item = work_item
        if agent_iteration is not None:
            self.agent_iterations.append(agent_iteration)
        if success_criteria is not None:
            self.success_criteria = list(success_criteria)
        if constraints is not None:
            self.constraints = list(constraints)
        if completed_action and completed_action not in self.completed_actions:
            self.completed_actions.append(completed_action)
        self.history.append(
            WorkflowCheckpoint(
                workflow_phase=workflow_phase,
                focus_domain=focus_domain,
                summary=summary,
                status=status,
                next_action=self.pending_action,
                handoff_skill=self.handoff_skill,
            )
        )
        if context:
            self.context.update(context)

    def snapshot(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "uvprojx_path": self.uvprojx_path,
            "project_name": self.project_name,
            "workflow_phase": self.workflow_phase,
            "phase_status": self.phase_status,
            "focus_domain": self.focus_domain,
            "artifact_consistency": self.artifact_consistency,
            "build_preset": self.build_preset,
            "configure_preset": self.configure_preset,
            "last_error": self.last_error,
            "debug_ready": self.debug_ready,
            "current_goal": self.current_goal.to_dict() if self.current_goal else None,
            "current_signal": self.current_signal.to_dict() if self.current_signal else None,
            "active_work_item": self.active_work_item.to_dict() if self.active_work_item else None,
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
            "planned_actions": [item.to_dict() for item in self.planned_actions],
            "completed_actions": list(self.completed_actions),
            "pending_action": self.pending_action,
            "handoff_skill": self.handoff_skill,
            "agent_iterations": [item.to_dict() for item in self.agent_iterations],
            "context": dict(self.context),
            "history": [item.to_dict() for item in self.history],
            "updated_at": self.updated_at,
        }
