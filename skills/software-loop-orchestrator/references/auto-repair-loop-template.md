# Agent 自主修复回路模板

## 1. 目标 (Objective)
在证据充分且风险可控时，为软件侧闭环提供一轮最小 Agent 自主修复，并立即回到可验证阶段。

## 2. 触发条件 (Trigger)
- 当前失败现象可复述
- 当前关注域已收束
- 当前存在候选文件与原始诊断
- 当前存在可立即执行的验证动作

## 3. 输入 (Input)
- `workflow_phase`
- `phase_status`
- `focus_domain`
- `artifact_consistency`
- `last_error`
- `current_goal`
- `active_work_item`

## 4. 步骤 (Steps)
1. 归纳失败事实，不做猜测性扩大解释。
2. 根据当前工作项选择一个最小动作。
3. 只修改最相关的一组文件。
4. 立即执行验证动作。
5. 比较修复前后信号。
6. 决定继续、反思、交接或停止。

## 5. 输出 (Output)
- `goal`
- `constraints`
- `candidate_files`
- `agent_edit_strategy`
- `validation_action`
- `repair_result`
- 下一阶段动作

还应补充：
- `files_touched`
- `new_signal`
- `reflection`

说明：
- `candidate_files` 用于给 Agent 提供最小上下文，而不是提前决定补丁类型。
- `validation_action` 用于告诉 Agent “改完立刻验证什么”，保证严格约束落在回路上。
- `reflection` 用于说明为什么继续当前方向、切换方向，或请求人工介入。

## 6. 停止条件 (Stop Conditions)
- 连续 2~3 轮无明显进展
- 需要高风险操作
- 根因骨发生漂移，说明原判断不成立
- 当前证据不足以继续自动修复

## 7. 示例 (Example)
```text
workflow_phase    : decide
focus_domain      : build
goal              : restore a reproducible build path
candidate_files   : Core/main.c, CMakeLists.txt
agent_edit_strategy: inspect the failing translation unit and apply one reversible edit
validation_action : re-run build_project(execute=True)
repair_result     : build passed, next -> prepare_debug_session
```
