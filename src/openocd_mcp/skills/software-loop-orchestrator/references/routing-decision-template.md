# 方向决策模板

## 1. 目标 (Objective)
统一跨 skill 交接时的最小信息集，避免只给结论、不带目标、信号和成功条件。

## 2. 输入 (Input)
- 当前相位
- 已执行动作
- 当前信号
- 下游目标 skill

## 3. 步骤 (Steps)
1. 概括当前所处相位与总目标。
2. 列出已完成动作。
3. 列出关键工件与信号。
4. 说明为何需要转交给目标 skill。

## 4. 输出 (Output)
- 结构化交接摘要
- 推荐 skill
- 推荐下一动作
- 若需要主动修复，还应给出 `current_goal`、`active_work_item`、`validation_action`

## 5. 风险 (Risks)
- 不带证据的交接会导致下游重复劳动。
- 不说明未完成动作会导致流程断裂。

## 6. 示例 (Example)
```text
当前相位：collect
已完成：build / debug_prepare / halt
信号：PC=0x08001234, UART tail="HardFault"
路由：openocd-triage-and-capture
原因：需要先冻结现场并生成标准化证据包
```

若场景属于可主动修复的 build / engineering / artifact 问题，则输出不应只停留在“交给某 skill”，还应明确：
- 当前总目标是什么
- 当前最小工作项是什么
- Agent 应先读哪些文件
- 改完后立刻验证什么