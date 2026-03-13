from __future__ import annotations

import re

from .models import DirectionDecision, ProposedAction, WorkflowGoal


class DirectionPlanner:
    _FOCUS_PATTERNS: dict[str, tuple[str, ...]] = {
        'engineering': ('uvprojx', 'template', 'project', 'path', 'generation', 'cmake error'),
        'build': ('cmake', 'preset', 'toolchain', 'compile', 'compiler', 'link', 'undefined reference', 'error:'),
        'artifact': ('elf', 'bin', 'hex', 'svd', 'artifact', 'mismatch', 'openocd.cfg'),
        'debug': ('openocd', 'connect', 'halt', 'reset', 'flash', 'breakpoint', 'watchpoint'),
        'runtime': ('hardfault', 'assert', 'crash', 'deadlock', 'wdt', 'stall', 'uart', 'fault'),
        'validation': ('regression', 'flaky', 'iteration', 'sample', 'timeout', 'pass', 'fail'),
    }

    def classify_focus(self, summary: str, *, fallback: str = 'unknown') -> str:
        lowered = summary.lower()
        for focus, patterns in self._FOCUS_PATTERNS.items():
            if any(re.search(re.escape(pattern), lowered) for pattern in patterns):
                return focus
        return fallback

    def create_primary_goal(self, project_name: str) -> WorkflowGoal:
        return WorkflowGoal(
            name='close software delivery loop',
            desired_outcome=f'establish a reproducible build-debug-validation loop for {project_name}',
            success_criteria=[
                'configure can complete with a stable generated project',
                'build can produce a usable firmware artifact',
                'debug preparation can hand off consistent artifacts',
            ],
            constraints=[
                'keep actions auditable and easy to validate',
                'prefer small, reversible edits',
                'ask for help only when risk or ambiguity blocks safe progress',
            ],
        )

    def after_bootstrap(self) -> DirectionDecision:
        return DirectionDecision(
            workflow_phase='execute',
            focus_domain='engineering',
            summary='工程骨架已建立，下一步进入 configure 以验证生成输入。',
            reasoning='方向编排优先确保工程基线可执行，而不是预先枚举所有后续步骤。',
            next_action='configure_project',
            success_criteria=['configure completes with a consistent generated project'],
        )

    def after_configure(self, *, success: bool, summary: str = '') -> DirectionDecision:
        if success:
            return DirectionDecision(
                workflow_phase='execute',
                focus_domain='build',
                summary='configure 已打通，下一步应通过 build 验证构建链路。',
                reasoning='一旦配置稳定，最直接的方向是验证能否产出可用工件。',
                next_action='build_project',
                success_criteria=['build produces a usable firmware artifact'],
            )
        return DirectionDecision(
            workflow_phase='decide',
            focus_domain=self.classify_focus(summary, fallback='engineering'),
            summary='configure 失败，应围绕配置输入与生成物一致性重新规划下一步。',
            reasoning='方向编排在失败时先收束到问题域，再把细节修复交给 Agent 或工具。',
            next_action='stabilize_configure_inputs',
            success_criteria=['configure can be rerun successfully'],
        )

    def after_build(self, *, success: bool, summary: str = '') -> DirectionDecision:
        if success:
            return DirectionDecision(
                workflow_phase='handoff',
                focus_domain='artifact',
                summary='build 已产生产物，下一步应准备调试工件并完成 skill 交接。',
                reasoning='方向编排在构建成功后不继续停留在 build，而是转向工件一致性与调试接入。',
                next_action='prepare_debug_session',
                handoff_skill='openocd-core-operations',
                success_criteria=['debug preparation yields consistent artifacts and configs'],
            )
        return DirectionDecision(
            workflow_phase='decide',
            focus_domain=self.classify_focus(summary, fallback='build'),
            summary='build 失败，应先围绕失败域收集证据并制定下一轮最小动作。',
            reasoning='方向编排只决定问题域和下一步方向，不直接内嵌具体补丁脚本。',
            next_action='stabilize_build_inputs',
            success_criteria=['build can be rerun successfully'],
        )

    def after_debug_prepare(self) -> DirectionDecision:
        return DirectionDecision(
            workflow_phase='handoff',
            focus_domain='debug',
            summary='调试工件已齐备，应交接到调试控制 skill 建立最小控制闭环。',
            reasoning='方向编排负责确定交接方向，而不是侵入式执行底层调试命令。',
            next_action='connect_and_halt',
            handoff_skill='openocd-core-operations',
            success_criteria=['target can be connected and controlled reproducibly'],
        )

    def runtime_issue(self, summary: str) -> DirectionDecision:
        return DirectionDecision(
            workflow_phase='collect',
            focus_domain=self.classify_focus(summary, fallback='runtime'),
            summary='运行期异常应优先冻结现场并抓取证据。',
            reasoning='方向编排在运行期问题中先导向取证，而不是直接修改代码。',
            next_action='freeze_and_capture',
            handoff_skill='openocd-triage-and-capture',
            success_criteria=['runtime evidence is captured with enough fidelity for analysis'],
        )

    def after_triage(self) -> DirectionDecision:
        return DirectionDecision(
            workflow_phase='decide',
            focus_domain='runtime',
            summary='已完成取证，应进入根因分析与假设收敛。',
            reasoning='证据收集后，方向编排转向分析而不是继续采样。',
            next_action='explain_root_cause',
            handoff_skill='openocd-deep-debug-analysis',
            success_criteria=['root cause hypotheses are explicit and testable'],
        )

    def after_analysis(self) -> DirectionDecision:
        return DirectionDecision(
            workflow_phase='verify',
            focus_domain='validation',
            summary='根因已形成，应转为自动化验证与回归确认。',
            reasoning='方向编排在分析后自动把目标切到验证闭环。',
            next_action='run_regression',
            handoff_skill='openocd-automation-validation',
            success_criteria=['regression confirms the hypothesis and fix'],
        )

    def after_regression(self, *, success: bool, summary: str = '') -> DirectionDecision:
        if success:
            return DirectionDecision(
                workflow_phase='done',
                focus_domain='validation',
                summary='回归通过，本轮软件闭环完成。',
                reasoning='成功路径应显式闭环，而不是停留在验证阶段。',
                next_action='deliver_result',
                success_criteria=['result can be delivered with evidence'],
            )
        return DirectionDecision(
            workflow_phase='reflect',
            focus_domain=self.classify_focus(summary, fallback='validation'),
            summary='回归失败，应回到失败样本与假设反思。',
            reasoning='方向编排在验证失败后先反思方向，再决定下一轮分析或修复。',
            next_action='inspect_failed_samples',
            handoff_skill='openocd-deep-debug-analysis',
            success_criteria=['a revised hypothesis or failure explanation is available'],
        )

    def create_work_item(
        self,
        *,
        owner_action: str,
        focus_domain: str,
        diagnostics: str,
        validation_action: str,
    ) -> list[ProposedAction]:
        return [
            ProposedAction(
                title='read diagnostics and inspect the smallest relevant files',
                description='Use the raw failure log to choose one minimal edit path instead of assuming a fixed patch recipe.',
                tool_hint='read-file / search / edit',
                risk='low',
                expected_outcome='a single candidate fix is identified with explicit rationale',
            ),
            ProposedAction(
                title='apply one minimal fix and immediately validate',
                description=f'Prefer one bounded workspace edit, then {validation_action}.',
                tool_hint='edit-workspace / rerun-command',
                risk='medium',
                expected_outcome='either the failure disappears or the next diagnostic becomes more specific',
            ),
        ]