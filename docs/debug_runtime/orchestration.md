# Global Orchestration Flow

当前全局编排已按 **方向编排（direction orchestration）** 重构，不再把总工作流实现为“固定阶段状态机 + 规则补丁器”。

新的定位是：

- 用 **目标** 约束方向
- 用 **信号** 承接现场事实
- 用 **决策** 给出下一步方向
- 用 **Agent 工作项** 驱动开放式执行
- 用 **验证与反思** 控制回环

## 目标

该编排层负责把软件侧闭环组织成以下方向性回路：

1. 明确当前总目标与成功条件
2. 根据最新信号判断当前关注域（engineering / build / artifact / debug / runtime / validation）
3. 给出下一步方向，而不是写死细粒度步骤
4. 在失败时生成 Agent 工作项，让 Agent 自主搜集信息、修改工作区并立即验证
5. 在成功、失败、反思、交接之间保持闭环可追溯

## 核心模块

- `src/keil2cmake/orchestrator/artifacts.py`
- `src/keil2cmake/orchestrator/models.py`
- `src/keil2cmake/orchestrator/state.py`
- `src/keil2cmake/orchestrator/planner.py`
- `src/keil2cmake/orchestrator/validation.py`
- `src/keil2cmake/orchestrator/workflow.py`

## 当前能力边界

当前版本提供：

- 全局目标、信号、决策与工作项快照
- configure / build / debug preparation / triage / analysis / regression 的方向编排
- 面向 Agent 的开放式修复回路（不依赖固定补丁脚本）
- 工件注册表与一致性评估
- 独立验证执行层，用于统一 `configure/build` 的回环验证

当前版本尚未提供：

- 跨多轮 session 的持久化恢复
- 运行期异常的自动化方向重规划器
- 人机协作式批准节点的结构化协议

## 方向编排原则

该层遵循 `AGENTS.MD` 中的 `Agent编排工作流指导原则`：

- **目标导向**：编排层先定义当前目标和成功条件，再让 Agent 自主决定路径
- **松耦合流程约束**：保留 `understand -> collect -> decide -> execute -> verify -> reflect -> handoff -> done` 的流程骨架，但不把阶段内部动作写死
- **工具即能力**：构建、调试、取证、验证只是能力，编排层只决定何时、为何调用它们
- **反思优先于硬编码补丁**：失败后先形成工作项与候选文件，再让 Agent 决定最小动作

## Agent 工作项模型

当 `configure` 或 `build` 失败时，编排层会直接生成 `AgentWorkItem`。其中包含：

- 当前 owner action（如 `configure` / `build`）
- 当前工作流相位与关注域
- 当前总目标
- 原始诊断日志
- 约束条件
- 候选文件快照
- 建议动作（仅作方向建议，不是固定脚本）
- 验证动作

因此，错误识别的主逻辑已从“规则式问题分类”转为“保留原始诊断 -> Agent 自主判断 -> 立即验证”。

## 当前状态字段

上层 Agent 当前应重点读取：

- `workflow_phase`
- `phase_status`
- `focus_domain`
- `artifact_consistency`
- `current_goal`
- `current_signal`
- `active_work_item`
- `success_criteria`
- `constraints`
- `planned_actions`
- `completed_actions`
- `pending_action`
- `handoff_skill`
- `agent_iterations`

## 面向 LLM 的入口定位

该编排层的主要消费者仍然是 Skill 驱动的 LLM / Agent，而不是人工 CLI。

推荐入口：

- `skills/software-loop-orchestrator/SKILL.md`

其中：

- `orchestrator` 提供目标、方向、工作项与回环状态
- Skill 提供全局方向判断原则
- Tool 层只负责执行原子能力

## 与双入口 EXE 分发的关系

采用双入口 EXE 分发时，这一层依然有效：

- `Keil2Cmake.exe` 仍然是主入口与方向编排承载者
- `openocd-mcp.exe` 是被 handoff 的 MCP 服务入口
- 编排层只决定何时进入 `debug/runtime/validation` 域，而不是自己变成 MCP 服务本体

因此，双入口解决的是进程与分发形态，不会削弱当前方向编排模型。
