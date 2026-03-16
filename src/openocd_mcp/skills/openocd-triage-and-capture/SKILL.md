---
name: openocd-triage-and-capture
description: Skill for first-response fault triage, volatile state preservation, and dual-channel evidence capture across OpenOCD and serial logs.
---

# OpenOCD 异常分诊与现场捕获指南 (Triage & Capture)

## 1. 角色定位

该 skill 是故障现场的“第一响应层”，负责在问题刚发生时**先保全现场，再形成初步判断**。

主责包括：
- 立即冻结目标，防止证据继续流失
- 同步采集 OpenOCD 与串口等多通道信息
- 对异常进行初步分流，例如 HardFault、WDT、死循环、卡死
- 生成可直接交给深度分析的证据包

该 skill 不负责：
- 进行完整根因分析
- 长时间脚本验证
- 替代 `core` 执行高风险恢复或刷写

## 2. 何时使用

当出现以下信号时，应优先使用本 skill：

- 用户报告“板子卡死”“串口停了”“看门狗反复复位”
- 目标已进入 HardFault、BusFault、MemManage 或疑似异常模式
- 需要将 CPU 状态与 UART/RTT 日志拼接成同一时间线
- 需要保全挥发性现场，避免复位后丢失第一现场

若用户已提供完整快照并要求解释根因，应直接路由到深度分析 skill。

## 3. 标准流程

### 3.1 冻结优先

1. 第一动作是 `halt` 或等价冻结动作。
2. 在未明确要求前，禁止立刻 `reset`、`resume` 或清现场。
3. 记录冻结时刻的 target state、PC、SP、LR。

### 3.2 多通道取证

优先采集以下数据：
- CPU 核心寄存器，如 `r0-r12`、`sp`、`lr`、`pc`、`xpsr`
- 异常相关系统寄存器，如 `CFSR`、`HFSR`、`BFAR`、`MMFAR`、`SHCSR`
- 串口或 RTT 的最近一段缓冲日志
- 若已知问题相关外设，补充最小必要的寄存器快照

### 3.3 快照组装

将证据整理为统一结构：
1. 时间点
2. 冻结状态
3. 寄存器快照
4. 异常寄存器解释
5. 串口上下文
6. 初步结论

### 3.4 初步分诊

分诊结论必须满足：
- 基于事实，而非经验猜测
- 指出最可能的异常类型
- 指出仍然缺失的证据
- 明确是否需要升级到深度分析

## 4. 输出要求

输出至少应包含：
- `summary`：当前分诊结论
- `current_state`：是否 halted、异常模式、目标是否可继续交互
- `evidence`：寄存器值、地址、串口片段、时间顺序
- `hypothesis`：初步异常类型与依据
- `next_action`：建议进入深度分析、补抓某个寄存器，或由 core 执行恢复
- `risk_or_limit`：例如日志不完整、现场可能已被 WDT 改写

必须避免两种输出：
- 只有寄存器，没有解释
- 只有解释，没有原始值

## 5. 路由规则

- 若需要还原调用链、解析 ELF/SVD、分析 RTOS 或多核问题，路由到 `openocd-deep-debug-analysis`
- 若分诊后发现其实只是链路问题、刷写问题或恢复问题，交回 `openocd-core-operations`
- 若需要把本次分诊动作固定成脚本并重复执行，交给 `openocd-automation-validation`

交接必须附带：
- 冻结时刻信息
- 原始寄存器快照
- 串口/RTT 关键片段
- 已确认与未确认结论

## 6. 共享规范

本 skill 严格继承并遵守：
- [COMMON-NORMS.md](../shared/COMMON-NORMS.md)
- [SKILL-STYLE-GUIDE.md](../shared/SKILL-STYLE-GUIDE.md)

特别强调三条：
1. 先冻结，后解释。
2. 先保全现场，后考虑恢复。
3. 不把初步分诊写成最终根因。

## 7. 参考资料
- [证据组装模板 (evidence-template.md)](references/evidence-template.md)
- [异常排查执行手册 (fault-runbook.md)](references/fault-runbook.md)
- [双通道协同陷阱与最佳实践 (gotchas-and-best-practices.md)](references/gotchas-and-best-practices.md)
