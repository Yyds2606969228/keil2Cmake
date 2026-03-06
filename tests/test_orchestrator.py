# -*- coding: utf-8 -*-

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from keil2cmake.orchestrator import GlobalOrchestrator
from keil2cmake.orchestrator.models import AgentRepairResult, CommandExecution
from keil2cmake.orchestrator.state import OrchestratorState


def _write_minimal_uvprojx(path: str) -> None:
    content = '''
<Project>
  <Targets>
    <Target>
      <TargetName>demo</TargetName>
      <Groups>
        <Group>
          <GroupName>Src</GroupName>
          <Files>
            <File>
              <FileName>main.c</FileName>
              <FileType>1</FileType>
              <FilePath>../Core/main.c</FilePath>
            </File>
          </Files>
        </Group>
      </Groups>
      <TargetOption>
        <TargetCommonOption>
          <Device>STM32F103C8</Device>
          <OutputDirectory>build/</OutputDirectory>
        </TargetCommonOption>
        <TargetArmAds>
          <Cads>
            <Optim>1</Optim>
            <VariousControls>
              <IncludePath>../Core/Inc</IncludePath>
              <Define>USE_HAL_DRIVER,STM32F103xB</Define>
              <MiscControls></MiscControls>
            </VariousControls>
          </Cads>
          <Aads>
            <VariousControls>
              <MiscControls></MiscControls>
            </VariousControls>
          </Aads>
          <LDads>
            <VariousControls>
              <MiscControls></MiscControls>
            </VariousControls>
          </LDads>
        </TargetArmAds>
      </TargetOption>
    </Target>
  </Targets>
</Project>
'''.strip()
    Path(path).write_text(content, encoding='utf-8')


class TestGlobalOrchestrator(unittest.TestCase):
    def test_state_snapshot_contains_direction_fields(self) -> None:
        state = OrchestratorState()
        state.transition(
            'decide',
            'build',
            'direction decided for build recovery',
            status='blocked',
            constraints=['do not modify application source', 're-run build immediately'],
            success_criteria=['build succeeds'],
        )

        snapshot = state.snapshot()

        self.assertEqual(snapshot['workflow_phase'], 'decide')
        self.assertEqual(snapshot['focus_domain'], 'build')
        self.assertEqual(snapshot['constraints'][0], 'do not modify application source')
        self.assertEqual(snapshot['success_criteria'], ['build succeeds'])
        self.assertEqual(snapshot['agent_iterations'], [])

    def test_bootstrap_project_registers_artifacts_and_route(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / 'demo'
            mdk = project_dir / 'MDK-ARM'
            core = project_dir / 'Core'
            inc = core / 'Inc'
            mdk.mkdir(parents=True, exist_ok=True)
            inc.mkdir(parents=True, exist_ok=True)
            (core / 'main.c').write_text('int main(void){return 0;}', encoding='utf-8')
            uvprojx = mdk / 'demo.uvprojx'
            _write_minimal_uvprojx(str(uvprojx))

            orchestrator = GlobalOrchestrator()
            snapshot = orchestrator.bootstrap_project(str(uvprojx), str(project_dir / 'generated'))

            self.assertEqual(snapshot['state']['workflow_phase'], 'execute')
            self.assertEqual(snapshot['state']['phase_status'], 'completed')
            self.assertEqual(snapshot['state']['focus_domain'], 'engineering')
            self.assertEqual(snapshot['state']['pending_action'], 'configure_project')
            self.assertIn('cmake_lists', snapshot['artifacts'])
            self.assertIn('openocd_cfg', snapshot['artifacts'])

    def test_build_and_debug_routing(self) -> None:
        executions = []

        def fake_runner(command, cwd):
            executions.append((command, cwd))
            return CommandExecution(command=command, cwd=cwd, executed=True, success=True, exit_code=0)

        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / 'demo'
            mdk = project_dir / 'MDK-ARM'
            core = project_dir / 'Core'
            inc = core / 'Inc'
            mdk.mkdir(parents=True, exist_ok=True)
            inc.mkdir(parents=True, exist_ok=True)
            (core / 'main.c').write_text('int main(void){return 0;}', encoding='utf-8')
            uvprojx = mdk / 'demo.uvprojx'
            _write_minimal_uvprojx(str(uvprojx))

            orchestrator = GlobalOrchestrator(runner=fake_runner)
            orchestrator.bootstrap_project(str(uvprojx), str(project_dir / 'generated'))
            configured = orchestrator.configure_project(execute=True)
            built = orchestrator.build_project(execute=True)
            prepared = orchestrator.prepare_debug_session(mcu='STM32F103C8', debugger='stlink')

            self.assertEqual(configured['state']['workflow_phase'], 'execute')
            self.assertEqual(configured['state']['pending_action'], 'build_project')
            self.assertEqual(built['state']['workflow_phase'], 'handoff')
            self.assertEqual(built['state']['artifact_consistency'], 'consistent')
            self.assertEqual(built['state']['pending_action'], 'prepare_debug_session')
            self.assertEqual(prepared['state']['workflow_phase'], 'handoff')
            self.assertEqual(prepared['state']['handoff_skill'], 'openocd-core-operations')
            self.assertTrue(executions)

    def test_runtime_to_regression_routes(self) -> None:
        orchestrator = GlobalOrchestrator()
        runtime_issue = orchestrator.report_runtime_issue('uart stalled after resume')
        triaged = orchestrator.mark_triaged('snapshot captured')
        analyzed = orchestrator.mark_analysis_complete('root cause linked to invalid RCC state')
        regressed = orchestrator.mark_regression(passed=False, summary='failed at iteration 12')

        self.assertEqual(runtime_issue['state']['workflow_phase'], 'collect')
        self.assertEqual(runtime_issue['state']['focus_domain'], 'runtime')
        self.assertEqual(runtime_issue['state']['handoff_skill'], 'openocd-triage-and-capture')
        self.assertEqual(triaged['state']['phase_status'], 'completed')
        self.assertEqual(triaged['state']['handoff_skill'], 'openocd-deep-debug-analysis')
        self.assertEqual(analyzed['state']['workflow_phase'], 'verify')
        self.assertEqual(analyzed['state']['handoff_skill'], 'openocd-automation-validation')
        self.assertEqual(regressed['state']['workflow_phase'], 'reflect')
        self.assertEqual(regressed['state']['pending_action'], 'inspect_failed_samples')

    def test_build_failure_creates_agent_work_item(self) -> None:
        def failing_runner(command, cwd):
            return CommandExecution(
                command=command,
                cwd=cwd,
                executed=True,
                success=False,
                exit_code=1,
                stderr='Core/main.c:10:10: fatal error: missing_board.h: No such file or directory',
            )

        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / 'demo'
            mdk = project_dir / 'MDK-ARM'
            core = project_dir / 'Core'
            inc = core / 'Inc'
            mdk.mkdir(parents=True, exist_ok=True)
            inc.mkdir(parents=True, exist_ok=True)
            (core / 'main.c').write_text('int main(void){return 0;}', encoding='utf-8')
            uvprojx = mdk / 'demo.uvprojx'
            _write_minimal_uvprojx(str(uvprojx))

            orchestrator = GlobalOrchestrator(runner=failing_runner)
            orchestrator.bootstrap_project(str(uvprojx), str(project_dir / 'generated'))
            snapshot = orchestrator.build_project(execute=True)

            self.assertFalse(snapshot['execution']['success'])
            self.assertEqual(snapshot['state']['workflow_phase'], 'decide')
            self.assertEqual(snapshot['state']['pending_action'], 'stabilize_build_inputs')
            self.assertIsNotNone(snapshot['state']['active_work_item'])
            self.assertIn('Core/main.c', snapshot['state']['active_work_item']['candidate_files'])

    def test_build_failure_can_be_repaired_by_agentic_source_edit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / 'demo'
            mdk = project_dir / 'MDK-ARM'
            core = project_dir / 'Core'
            inc = core / 'Inc'
            mdk.mkdir(parents=True, exist_ok=True)
            inc.mkdir(parents=True, exist_ok=True)
            source_file = core / 'main.c'
            source_file.write_text('board_handle_t board = 0;\nint main(void){return board;}\n', encoding='utf-8')
            uvprojx = mdk / 'demo.uvprojx'
            _write_minimal_uvprojx(str(uvprojx))

            def runner(command, cwd):
                content = source_file.read_text(encoding='utf-8')
                if 'board_handle_t' in content:
                    return CommandExecution(
                        command=command,
                        cwd=cwd,
                        executed=True,
                        success=False,
                        exit_code=1,
                        stderr="Core/main.c:1:1: error: unknown type name 'board_handle_t'",
                    )
                return CommandExecution(command=command, cwd=cwd, executed=True, success=True, exit_code=0)

            def repair_agent(observation):
                self.assertIn("unknown type name 'board_handle_t'", observation.diagnostics)
                self.assertIn('Core/main.c', observation.candidate_files)
                updated = observation.candidate_files['Core/main.c'].replace('board_handle_t', 'int')
                source_file.write_text(updated, encoding='utf-8')
                return AgentRepairResult(
                    applied=True,
                    summary='replaced unresolved board_handle_t with int in the failing translation unit',
                    files_touched=['Core/main.c'],
                    strategy='edit the failing source file directly based on compiler diagnostics',
                )

            orchestrator = GlobalOrchestrator(runner=runner)
            orchestrator.bootstrap_project(str(uvprojx), str(project_dir / 'generated'))
            snapshot = orchestrator.build_project(execute=True, repair_agent=repair_agent, max_repair_attempts=2)

            self.assertFalse(snapshot['execution']['success'])
            self.assertEqual(snapshot['state']['workflow_phase'], 'handoff')
            self.assertEqual(snapshot['state']['pending_action'], 'prepare_debug_session')
            self.assertEqual(snapshot['agentic_repairs'][0]['files_touched'], ['Core/main.c'])
            self.assertTrue(snapshot['validation_execution']['success'])
            self.assertNotIn('board_handle_t', source_file.read_text(encoding='utf-8'))

    def test_configure_failure_can_be_repaired_by_agentic_workspace_edit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / 'demo'
            mdk = project_dir / 'MDK-ARM'
            core = project_dir / 'Core'
            inc = core / 'Inc'
            mdk.mkdir(parents=True, exist_ok=True)
            inc.mkdir(parents=True, exist_ok=True)
            (core / 'main.c').write_text('int main(void){return 0;}\n', encoding='utf-8')
            uvprojx = mdk / 'demo.uvprojx'
            _write_minimal_uvprojx(str(uvprojx))

            def runner(command, cwd):
                content = cmake_lists.read_text(encoding='utf-8')
                if 'broken_command()' in content:
                    return CommandExecution(
                        command=command,
                        cwd=cwd,
                        executed=True,
                        success=False,
                        exit_code=1,
                        stderr='CMake Error at CMakeLists.txt:2 (broken_command):\n  Unknown CMake command "broken_command".',
                    )
                return CommandExecution(command=command, cwd=cwd, executed=True, success=True, exit_code=0)

            orchestrator = GlobalOrchestrator(runner=runner)
            orchestrator.bootstrap_project(str(uvprojx), str(project_dir / 'generated'))
            cmake_lists = Path(orchestrator.state.project_root) / 'CMakeLists.txt'
            cmake_lists.write_text(cmake_lists.read_text(encoding='utf-8') + '\nbroken_command()\n', encoding='utf-8')

            def repair_agent(observation):
                self.assertIn('Unknown CMake command', observation.diagnostics)
                updated = observation.candidate_files['CMakeLists.txt'].replace('\nbroken_command()\n', '\n')
                cmake_lists.write_text(updated, encoding='utf-8')
                return AgentRepairResult(
                    applied=True,
                    summary='removed unsupported CMake directive from generated CMakeLists',
                    files_touched=['CMakeLists.txt'],
                    strategy='edit generated build configuration based on configure diagnostics',
                )

            snapshot = orchestrator.configure_project(execute=True, repair_agent=repair_agent, max_repair_attempts=2)

            self.assertFalse(snapshot['execution']['success'])
            self.assertEqual(snapshot['state']['workflow_phase'], 'execute')
            self.assertEqual(snapshot['state']['pending_action'], 'build_project')
            self.assertTrue(snapshot['validation_execution']['success'])
            self.assertEqual(snapshot['agentic_repairs'][0]['files_touched'], ['CMakeLists.txt'])