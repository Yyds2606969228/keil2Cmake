# 方向相位矩阵

## 1. 目标 (Objective)
提供方向编排各相位的统一定义，避免 LLM 在 `understand / collect / decide / execute / verify / reflect / handoff / done` 之间混淆职责。

## 2. 输入 (Input)
- 当前目标描述
- 当前工件状态
- 最新错误或异常现象

## 3. 步骤 (Steps)
1. 判断当前处于哪个 `workflow_phase`。
2. 结合 `focus_domain` 与已完成动作判断是否需要继续推进、回流或交接。
3. 根据相位与关注域决定下一动作或下一 skill。

## 4. 输出 (Output)
- 当前 `workflow_phase`
- 推荐下一动作
- 推荐 skill 或执行域

## 5. 风险 (Risks)
- 相位判定错误会导致高风险动作提前执行。
- 把 `build` 问题误判成 `runtime`，会浪费调试时间。

## 6. 示例 (Example)
```text
workflow_phase : decide
focus_domain   : build
next_action    : stabilize_build_inputs
handoff_skill  : none
```