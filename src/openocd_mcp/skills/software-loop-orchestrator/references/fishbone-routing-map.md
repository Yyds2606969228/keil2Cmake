# 方向分因映射图

## 1. 目标 (Objective)
使用“现象 -> 关注域 -> 下一方向”的映射方式，帮助 LLM 先判断问题主要落在哪个域，再决定下一步动作。

## 2. 输入 (Input)
- 当前目标摘要
- 最新失败现象或错误信息
- 当前工件一致性状态

## 3. 步骤 (Steps)
1. 先识别当前总目标和最新信号。
2. 再根据错误文本、工件状态和现场现象判断主要关注域。
3. 将关注域映射到下一动作或下游 skill。

## 4. 输出 (Output)
- `focus_domain`
- 推荐方向决策
- 推荐下一动作

## 5. 风险 (Risks)
- 只看流程相位不看现场信号，会导致方向失真。
- 只看错误文本不看目标与工件状态，会导致局部最优修复。

## 6. 示例 (Example)
```text
workflow_phase       : decide
focus_domain         : build
reason               : undefined reference + toolchain preset mismatch
next_action          : stabilize_build_inputs
handoff_skill        : none
```