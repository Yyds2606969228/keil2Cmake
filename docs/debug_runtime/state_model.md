# Session State Model

本文档包含两类状态：

1. `openocd-mcp` 会话态
2. `keil2cmake orchestrator` 全局编排态

## Session Fields

- `debugger_connected: bool`
- `serial_connected: bool`
- `target_state: unknown|connected|running|halted`
- `flash_mode: bool`

## Runtime State

- `breakpoints`: id-indexed entries (`bp` / `wp`)
- `tasks`: id-indexed script tasks
- `trigger_history`: serial cross-trigger hit history with timestamp

## Transition Rules

1. `connect_debugger` -> `debugger_connected=true`, `target_state=connected`
2. `control_target("halt"|"step")` -> `target_state=halted`, emits `TARGET_HALTED`
3. `control_target("resume"|"run"|"reset"|"init")` -> `target_state=running`
4. `serial_set_trigger` hit -> emits `TARGET_HALTED` (if action is halt), writes trigger timestamp
5. `WATCHPOINT_HIT` log event -> emits `WATCHPOINT_HIT` to task runtime
6. `emergency_stop` -> force-cancels tasks, verifies breakpoint/watchpoint clearance, resets+halts target, disconnects all channels

## Orchestrator State Fields

- `project_root: str`
- `uvprojx_path: str`
- `project_name: str`
- `workflow_phase: idle|understand|collect|decide|execute|verify|reflect|handoff|done`
- `phase_status: idle|completed|failed|blocked|running`
- `focus_domain: unknown|engineering|build|artifact|debug|runtime|validation`
- `artifact_consistency: unknown|pending|consistent|inconsistent`
- `build_preset: str`
- `configure_preset: str`
- `last_error: str | null`
- `debug_ready: bool`
- `current_goal: object | null`
- `current_signal: object | null`
- `active_work_item: object | null`
- `success_criteria: list[str]`
- `constraints: list[str]`
- `planned_actions: list[object]`
- `completed_actions: list[str]`
- `pending_action: str | null`
- `handoff_skill: str | null`
- `agent_iterations: list[object]`

说明：
- `current_goal` 用于表达当前总目标与成功条件。
- `current_signal` 用于承接最新事实或失败现场。
- `active_work_item` 用于把开放式修复任务交给 Agent，而不是给固定补丁脚本。
- `planned_actions` 是方向建议，不是硬编码步骤清单。

## Orchestrator Transition Rules

1. `bootstrap_project()` -> 建立 `current_goal`，进入 `workflow_phase=execute`，方向指向 `configure_project`
2. `configure_project()` 成功 -> 方向切换到 `build_project`
3. `configure_project()` 失败 -> 进入 `workflow_phase=decide`，生成 `active_work_item`
4. `build_project()` 成功 -> 进入 `workflow_phase=handoff`，方向切换到 `prepare_debug_session`
5. `build_project()` 失败 -> 进入 `workflow_phase=decide`，生成 `active_work_item`
6. `run_agentic_repair_loop()` -> 进入 `execute -> verify -> reflect` 回环，直到成功或需要重新评估方向
7. `prepare_debug_session()` -> 进入 `handoff`，把方向交给 `openocd-core-operations`
8. `report_runtime_issue()` -> 进入 `collect`，把方向交给 `openocd-triage-and-capture`
9. `mark_triaged()` -> 进入 `decide`，把方向交给 `openocd-deep-debug-analysis`
10. `mark_analysis_complete()` -> 进入 `verify`，把方向交给 `openocd-automation-validation`
11. `mark_regression(passed=True)` -> 进入 `done`
12. `mark_regression(passed=False)` -> 进入 `reflect`

## Agentic Recovery Constraints

1. Recovery loop 必须以原始诊断和候选文件为输入，而不是盲改代码。
2. 单轮只应提交一组最小、可解释的变更。
3. 每轮编辑后必须立即触发验证。
4. 连续 2~3 轮无明显进展，应转入 `reflect` 并请求人工介入或重新规划目标。

## 方向编排的错误处理

当前版本不再把 build/configure 失败优先压缩成固定问题类型。默认策略是：

1. 保留原始 `stderr/stdout`
2. 生成 `WorkflowSignal`
3. 根据信号确定 `focus_domain`
4. 生成 `AgentWorkItem`
5. 把下一步方向交给 Agent
6. 让验证执行层负责立即重跑
