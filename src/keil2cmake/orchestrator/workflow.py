from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from ..compiler.clangd import generate_clangd_config
from ..compiler.debug import generate_debug_templates, generate_openocd_files
from ..compiler.presets import generate_cmake_presets
from ..compiler.toolchains import generate_toolchains
from ..keil.config import get_cmake_path
from ..keil.device import detect_cpu_architecture
from ..keil.uvprojx import parse_uvprojx
from ..project_gen import generate_cmake_structure
from .artifacts import ArtifactRegistry
from .models import AgentRepairResult, AgentWorkItem, CommandExecution, DirectionDecision, WorkflowSignal
from .planner import DirectionPlanner
from .state import OrchestratorState
from .validation import ValidationExecutor

Runner = Callable[[list[str], str], CommandExecution]
RepairAgent = Callable[[AgentWorkItem], AgentRepairResult | None]


def default_runner(command: list[str], cwd: str) -> CommandExecution:
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandExecution(
        command=command,
        cwd=cwd,
        executed=True,
        success=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class GlobalOrchestrator:
    def __init__(self, runner: Runner | None = None) -> None:
        self.artifacts = ArtifactRegistry()
        self.planner = DirectionPlanner()
        self.state = OrchestratorState()
        self.validation_executor = ValidationExecutor()
        self._runner = runner or default_runner

    def bootstrap_project(self, uvprojx_path: str, output_dir: str) -> dict[str, Any]:
        project_data = parse_uvprojx(uvprojx_path)
        project_root = os.path.abspath(output_dir)
        os.makedirs(project_root, exist_ok=True)

        generate_cmake_structure(project_data, project_root)
        generate_toolchains(project_data, project_root)
        generate_cmake_presets(project_root, project_data)
        generate_clangd_config(
            project_root,
            detect_cpu_architecture(project_data['device']),
            project_data.get('use_microlib'),
        )
        generate_debug_templates(project_root)
        debug_files = generate_openocd_files(
            project_root,
            project_data.get('device', ''),
            project_data.get('debugger', ''),
            overwrite=False,
        )

        self.state.project_root = project_root
        self.state.uvprojx_path = project_data['uvprojx_path']
        self.state.project_name = project_data['project_name']
        self.state.context.update(
            {
                'device': project_data.get('device', ''),
                'debugger': project_data.get('debugger', ''),
                'project_root': project_root,
            }
        )
        self.artifacts.register(
            kind="source",
            path=project_data['uvprojx_path'],
            role="uvprojx",
            producer="bootstrap_project",
            metadata={"project_name": project_data['project_name']},
        )
        self.artifacts.register(
            kind="build-config",
            path=os.path.join(project_root, 'CMakeLists.txt'),
            role="cmake_lists",
            producer="bootstrap_project",
        )
        self.artifacts.register(
            kind="build-config",
            path=os.path.join(project_root, 'CMakePresets.json'),
            role="cmake_presets",
            producer="bootstrap_project",
        )
        self.artifacts.register(
            kind="debug-config",
            path=debug_files['openocd_cfg'],
            role="openocd_cfg",
            producer="bootstrap_project",
            metadata={
                "interface": debug_files['openocd_interface'],
                "target": debug_files['openocd_target'],
                "transport": debug_files['openocd_transport'],
            },
        )
        goal = self.planner.create_primary_goal(project_data['project_name'])
        decision = self.planner.after_bootstrap()
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            decision.summary,
            status='completed',
            decision=decision,
            goal=goal,
            artifact_consistency=self._evaluate_artifact_consistency(stage='project_generation'),
            completed_action='bootstrap_project',
            device=project_data.get('device', ''),
            debugger=project_data.get('debugger', ''),
        )
        return self.snapshot()

    def configure_project(
        self,
        *,
        preset: str = 'keil2cmake',
        execute: bool = False,
        repair_agent: RepairAgent | None = None,
        max_repair_attempts: int = 1,
    ) -> dict[str, Any]:
        self._require_project_root()
        cmake = get_cmake_path() or 'cmake'
        command = [cmake, '--preset', preset]
        execution = self._execute(command, self.state.project_root, execute=execute)
        self.state.configure_preset = preset
        summary = execution.stderr or execution.stdout or ''
        decision = self.planner.after_configure(success=execution.success if execute else True, summary=summary)
        signal = self._build_signal('configure', summary, severity='error' if execute and not execution.success else 'info')
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            '已生成 configure 命令。' if not execute else decision.summary,
            status='completed' if execution.success or not execute else 'failed',
            decision=decision,
            signal=signal,
            artifact_consistency=self._evaluate_artifact_consistency(stage='configure'),
            completed_action='configure_project' if execution.success or not execute else None,
            last_execution=execution.to_dict(),
        )
        if execute and not execution.success:
            return self._handle_failed_execution(
                owner_action='configure',
                summary=summary or 'configure failed',
                repair_agent=repair_agent,
                max_attempts=max_repair_attempts,
                execution=execution,
            )
        return self.snapshot(execution=execution)

    def build_project(
        self,
        *,
        preset: str = 'build',
        execute: bool = False,
        repair_agent: RepairAgent | None = None,
        max_repair_attempts: int = 1,
    ) -> dict[str, Any]:
        self._require_project_root()
        cmake = get_cmake_path() or 'cmake'
        command = [cmake, '--build', '--preset', preset]
        execution = self._execute(command, self.state.project_root, execute=execute)
        self.state.build_preset = preset
        if execute and execution.success:
            elf_path = os.path.join(self.state.project_root, 'build', f"{Path(self.state.project_root).name}.elf")
            self.artifacts.register(
                kind='firmware',
                path=elf_path,
                role='elf_candidate',
                producer='build_project',
                metadata={'preset': preset},
            )
        summary = execution.stderr or execution.stdout or ''
        decision = self.planner.after_build(success=execution.success if execute else True, summary=summary)
        signal = self._build_signal('build', summary, severity='error' if execute and not execution.success else 'info')
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            '已生成 build 命令。' if not execute else decision.summary,
            status='completed' if execution.success or not execute else 'failed',
            decision=decision,
            signal=signal,
            artifact_consistency=self._evaluate_artifact_consistency(stage='build'),
            completed_action='build_project' if execution.success or not execute else None,
            last_execution=execution.to_dict(),
        )
        if execute and not execution.success:
            return self._handle_failed_execution(
                owner_action='build',
                summary=summary or 'build failed',
                repair_agent=repair_agent,
                max_attempts=max_repair_attempts,
                execution=execution,
            )
        return self.snapshot(execution=execution)

    def prepare_debug_session(
        self,
        *,
        mcu: str,
        debugger: str,
        elf_path: str | None = None,
        svd_path: str | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        self._require_project_root()
        debug_files = generate_openocd_files(self.state.project_root, mcu, debugger, overwrite=overwrite)
        if elf_path:
            self.artifacts.register(
                kind='firmware',
                path=os.path.abspath(elf_path),
                role='elf',
                producer='prepare_debug_session',
            )
        if svd_path:
            self.artifacts.register(
                kind='metadata',
                path=os.path.abspath(svd_path),
                role='svd',
                producer='prepare_debug_session',
            )
        self.artifacts.register(
            kind='debug-config',
            path=debug_files['openocd_cfg'],
            role='openocd_cfg',
            producer='prepare_debug_session',
            metadata={'interface': debug_files['openocd_interface'], 'target': debug_files['openocd_target']},
        )
        self.state.debug_ready = True
        decision = self.planner.after_debug_prepare()
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            decision.summary,
            status='completed',
            decision=decision,
            artifact_consistency=self._evaluate_artifact_consistency(stage='debug_preparation'),
            completed_action='prepare_debug_session',
            device=mcu,
            debugger=debugger,
        )
        return self.snapshot()

    def report_runtime_issue(self, summary: str) -> dict[str, Any]:
        decision = self.planner.runtime_issue(summary)
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            decision.summary,
            status='blocked',
            decision=decision,
            signal=self._build_signal('runtime', summary, severity='error'),
            artifact_consistency=self._evaluate_artifact_consistency(stage='runtime_triage'),
        )
        return self.snapshot()

    def mark_triaged(self, summary: str) -> dict[str, Any]:
        decision = self.planner.after_triage()
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            summary,
            status='completed',
            decision=decision,
            artifact_consistency=self._evaluate_artifact_consistency(stage='runtime_triage'),
            completed_action='mark_triaged',
        )
        return self.snapshot()

    def mark_analysis_complete(self, summary: str) -> dict[str, Any]:
        decision = self.planner.after_analysis()
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            summary,
            status='completed',
            decision=decision,
            artifact_consistency=self._evaluate_artifact_consistency(stage='root_cause_analysis'),
            completed_action='mark_analysis_complete',
        )
        return self.snapshot()

    def mark_regression(self, *, passed: bool, summary: str) -> dict[str, Any]:
        decision = self.planner.after_regression(success=passed, summary=summary)
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            summary,
            status='completed' if passed else 'failed',
            decision=decision,
            artifact_consistency=self._evaluate_artifact_consistency(stage='regression_validation'),
            completed_action='mark_regression' if passed else None,
        )
        return self.snapshot()

    def run_agentic_repair_loop(
        self,
        repair_agent: RepairAgent,
        *,
        max_attempts: int = 1,
        owner_action: str,
    ) -> dict[str, Any]:
        if max_attempts < 1:
            raise ValueError('max_attempts must be at least 1.')

        if self.state.active_work_item is None:
            raise ValueError('agentic repair loop requires an active work item.')

        agentic_repairs: list[AgentRepairResult] = []
        validation_execution: CommandExecution | None = None

        for attempt in range(1, max_attempts + 1):
            work_item = self._build_agent_work_item(owner_action=owner_action, attempt=attempt)
            result = repair_agent(work_item) or AgentRepairResult(
                applied=False,
                summary='repair agent returned no actionable result',
            )
            agentic_repairs.append(result)
            self.state.transition(
                'execute',
                self.state.focus_domain,
                f'agentic repair attempt {attempt}: {result.summary}',
                status='running' if result.applied else 'failed',
                work_item=work_item,
                agent_iteration=result,
                artifact_consistency=self._evaluate_artifact_consistency(stage=owner_action),
            )

            if not result.applied:
                self.state.active_work_item = None
                return self.snapshot(
                    agentic_repairs=agentic_repairs,
                    validation_execution=None,
                    execution=None,
                )

            validation_execution = self._run_validation_action_for_stage(owner_action)
            if validation_execution.success:
                return self._complete_recovery(
                    owner_action=owner_action,
                    success=True,
                    summary=validation_execution.stderr or validation_execution.stdout or result.summary or 'agentic repair validated',
                    agentic_repairs=agentic_repairs,
                    validation_execution=validation_execution,
                )

            self.state.last_error = validation_execution.stderr or validation_execution.stdout or 'validation failed'
            if attempt < max_attempts:
                self.state.transition(
                    'reflect',
                    self.state.focus_domain,
                    f'validation failed after attempt {attempt}; continue with updated diagnostics.',
                    status='blocked',
                    artifact_consistency=self._evaluate_artifact_consistency(stage=owner_action),
                )

        return self._complete_recovery(
            owner_action=owner_action,
            success=False,
            summary=self.state.last_error or 'agentic repair exhausted without a successful validation pass',
            agentic_repairs=agentic_repairs,
            validation_execution=validation_execution,
        )

    def snapshot(
        self,
        execution: CommandExecution | None = None,
        agentic_repairs: list[AgentRepairResult] | None = None,
        validation_execution: CommandExecution | None = None,
    ) -> dict[str, Any]:
        return self._snapshot(
            execution=execution,
            agentic_repairs=agentic_repairs,
            validation_execution=validation_execution,
        )

    def _snapshot(
        self,
        *,
        execution: CommandExecution | None = None,
        agentic_repairs: list[AgentRepairResult] | None = None,
        validation_execution: CommandExecution | None = None,
    ) -> dict[str, Any]:
        data = {
            'state': self.state.snapshot(),
            'artifacts': self.artifacts.grouped(),
        }
        if execution is not None:
            data['execution'] = execution.to_dict()
        if agentic_repairs is not None:
            data['agentic_repairs'] = [item.to_dict() for item in agentic_repairs]
        if validation_execution is not None:
            data['validation_execution'] = validation_execution.to_dict()
        return data


    def _handle_failed_execution(
        self,
        *,
        owner_action: str,
        summary: str,
        repair_agent: RepairAgent | None,
        max_attempts: int,
        execution: CommandExecution,
    ) -> dict[str, Any]:
        work_item = self._build_agent_work_item(owner_action=owner_action, attempt=1, diagnostics=summary)
        self.state.transition(
            'decide',
            self.state.focus_domain,
            f'{owner_action} failed; direction orchestration has prepared an agent work item.',
            status='blocked',
            work_item=work_item,
            planned_actions=list(work_item.suggested_actions),
            artifact_consistency=self._evaluate_artifact_consistency(stage=owner_action),
        )
        if repair_agent is None:
            return self.snapshot(execution=execution)
        cycle_snapshot = self.run_agentic_repair_loop(repair_agent, max_attempts=max_attempts, owner_action=owner_action)
        cycle_snapshot['execution'] = execution.to_dict()
        return cycle_snapshot

    def _complete_recovery(
        self,
        *,
        owner_action: str,
        success: bool,
        summary: str,
        agentic_repairs: list[AgentRepairResult],
        validation_execution: CommandExecution | None,
    ) -> dict[str, Any]:
        decision = self.planner.after_build(success=True) if owner_action == 'build' and success else None
        if owner_action == 'configure' and success:
            decision = self.planner.after_configure(success=True)
        if not success:
            decision = DirectionDecision(
                workflow_phase='reflect',
                focus_domain=self.state.focus_domain,
                summary=summary,
                reasoning='the latest repair cycle did not reach the desired outcome; the next step is to reassess direction or escalate to a human.',
                next_action='reassess_strategy',
                success_criteria=['a clearer direction or escalation decision is produced'],
            )
        self.state.active_work_item = None
        self.state.transition(
            decision.workflow_phase,
            decision.focus_domain,
            summary,
            status='completed' if success else 'failed',
            decision=decision,
            artifact_consistency=self._evaluate_artifact_consistency(stage=owner_action),
            success_criteria=list(decision.success_criteria),
            completed_action='agentic_recovery' if success else None,
        )
        if success:
            self.state.last_error = None
        return self.snapshot(agentic_repairs=agentic_repairs, validation_execution=validation_execution)

    def _execute(self, command: list[str], cwd: str, *, execute: bool) -> CommandExecution:
        if not execute:
            return CommandExecution(command=command, cwd=cwd, executed=False, success=True, exit_code=None)
        return self._runner(command, cwd)

    def _build_signal(self, source: str, summary: str, *, severity: str) -> WorkflowSignal:
        return WorkflowSignal(source=source, summary=summary or f'{source} event', details=summary, severity=severity)

    def _build_agent_work_item(
        self,
        *,
        owner_action: str,
        attempt: int,
        diagnostics: str | None = None,
    ) -> AgentWorkItem:
        diagnostics_text = diagnostics or self.state.last_error or ''
        candidate_files = self._collect_candidate_files(diagnostics_text)
        suggested_actions = self.planner.create_work_item(
            owner_action=owner_action,
            focus_domain=self.state.focus_domain,
            diagnostics=diagnostics_text,
            validation_action=f're-run {owner_action}_project(execute=True)' if owner_action in {'build', 'configure'} else f're-run {owner_action}',
        )
        return AgentWorkItem(
            owner_action=owner_action,
            workflow_phase=self.state.workflow_phase,
            focus_domain=self.state.focus_domain,
            goal=(self.state.current_goal.desired_outcome if self.state.current_goal else 'restore workflow progress'),
            summary=f'Attempt {attempt}: use the latest diagnostics to choose one minimal corrective action.',
            diagnostics=diagnostics_text,
            constraints=list(self.state.constraints) + [
                'prefer one reversible edit per iteration',
                'validate immediately after the edit',
            ],
            candidate_files=candidate_files,
            suggested_actions=suggested_actions,
            validation_action=f're-run {owner_action}_project(execute=True)' if owner_action in {'build', 'configure'} else f're-run {owner_action}',
            metadata={
                'attempt': attempt,
                'project_root': self.state.project_root,
                'pending_action': self.state.pending_action,
            },
        )

    def _collect_candidate_files(self, diagnostics: str) -> dict[str, str]:
        candidates = ['CMakeLists.txt', 'cmake/user/keil2cmake_user.cmake']
        possible_files = re.findall(r'([A-Za-z]:[^:\r\n]+|[^:\r\n]+\.(?:cmake|c|cc|cpp|cxx|h|hpp|txt|json))', diagnostics)
        for item in possible_files:
            normalized = item.replace('\\', '/')
            if normalized not in candidates:
                candidates.append(normalized)
        snapshots: dict[str, str] = {}
        root = Path(self.state.project_root)
        for item in candidates:
            for resolved in (root / item, root.parent / item, Path(item)):
                if resolved.exists() and resolved.is_file():
                    content = resolved.read_text(encoding='utf-8', errors='replace')
                    snapshots[item] = content[:6000] + ('\n...<truncated>...' if len(content) > 6000 else '')
                    break
        return snapshots

    def _evaluate_artifact_consistency(self, *, stage: str) -> str:
        openocd_cfg = self.artifacts.latest('openocd_cfg')
        elf = self.artifacts.latest('elf') or self.artifacts.latest('elf_candidate')
        svd = self.artifacts.latest('svd')

        if stage in {'project_generation', 'configure'}:
            return 'pending'
        if stage == 'build':
            return 'consistent' if elf is not None else 'pending'
        if stage == 'debug_preparation':
            return 'consistent' if openocd_cfg is not None and elf is not None else 'pending'
        if stage in {'runtime_triage', 'root_cause_analysis', 'regression_validation', 'closed_loop_complete'}:
            if openocd_cfg is None or elf is None:
                return 'inconsistent'
            return 'consistent' if svd is not None else 'pending'
        return 'unknown'

    def _require_project_root(self) -> None:
        if not self.state.project_root:
            raise ValueError('project_root is not initialized. Call bootstrap_project() first.')

    def _run_validation_action_for_stage(self, stage: str) -> CommandExecution:
        return self.validation_executor.run(
            stage=stage,
            project_root=self.state.project_root,
            build_preset=self.state.build_preset,
            configure_preset=self.state.configure_preset,
            execute_command=self._execute,
        )
