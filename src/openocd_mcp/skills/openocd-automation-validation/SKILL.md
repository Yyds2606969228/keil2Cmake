---
name: openocd-automation-validation
description: Skill for deterministic sandbox automation, repeatable regression validation, scripted evidence capture, and machine-readable pass/fail reporting.
---

# OpenOCD 自动化与回归验证指南 (Automation & Validation)

## 1. 角色定位

该 skill 负责把人工调试动作固化为**可重复、可统计、可追责**的自动化流程。

主责包括：
- 在受限 Python 沙箱中执行调试脚本
- 将人工排查步骤改写为回归用例
- 统一统计通过/失败、超时、波动与失败样本
- 为深度分析提供自动抓取的失败快照

它不负责：
- 在没有明确目标时替代故障分诊
- 直接解释复杂根因
- 绕过安全门禁执行高风险操作

## 2. 何时使用

当任务满足以下任一条件时，应优先使用本 skill：

- 需要把一次性调试动作重复执行几十次到上千次
- 需要机器可读的 `pass/fail` 结果，而不是人工口头判断
- 需要带超时、重试、统计和失败样本保留的验证流程
- 需要把修复方案固化为可回归的测试脚本
- 需要在问题复现时自动抓取寄存器、内存或日志快照

## 3. 标准流程

### 3.1 脚本前置检查

运行前确认：
- 脚本输入参数明确
- 成功条件和失败条件明确
- 允许访问的能力边界明确
- 超时与最大迭代次数明确

### 3.2 执行流程

1. **脚本骨架加载**
	- 使用统一入口函数
	- 统一结果结构
	- 统一异常处理与清理逻辑
2. **沙箱边界校验**
	- 确认未触碰禁止网络、文件系统或未授权调试能力
	- 确认脚本不会无限循环
3. **任务执行**
	- 启动任务并记录开始时间、参数、上下文摘要
	- 按步骤执行控制动作和断言
4. **结果汇总**
	- 汇总通过数、失败数、失败样本、平均耗时
	- 对失败样本保留最小必要快照

### 3.3 失败处理

出现以下情况必须终止任务并输出失败：
- 超时
- 未捕获异常
- 断言失败
- 会话失联且无法恢复

终止后必须说明：
- 失败轮次
- 失败动作
- 最后可见证据

## 4. 输出要求

输出至少应包含：
- `summary`：本轮自动化验证的总体结论
- `current_state`：任务状态、执行次数、超时状态
- `evidence`：断言结果、失败日志、寄存器或内存快照摘要
- `next_action`：是否建议进入深度分析、补抓证据或回归更多轮
- `risk_or_limit`：例如样本量不足、存在 flaky 行为、当前脚本覆盖范围有限
- `raw_output`：脚本原始日志或框架输出

建议附加字段：
- `passed`
- `failed`
- `iterations`
- `duration_ms`
- `failed_samples`

## 5. 路由规则

- 若脚本发现稳定失败样本，应交给 `openocd-deep-debug-analysis` 做根因解释
- 若脚本失败源于连接、刷写、恢复或权限边界问题，应交回 `openocd-core-operations`
- 若需要先冻结并补抓第一现场，而不是继续回归，应交给 `openocd-triage-and-capture`

交接时应附带：
- 失败轮次
- 触发条件
- 失败快照
- 基线与偏差

## 6. 共享规范

本 skill 严格继承并遵守：
- [COMMON-NORMS.md](../shared/COMMON-NORMS.md)
- [SKILL-STYLE-GUIDE.md](../shared/SKILL-STYLE-GUIDE.md)

特别强调三条：
1. 自动化必须可停止。
2. 回归必须有标准。
3. 失败样本必须可回放或可交接。

## 7. 参考资料
- [回归测试模板 (regression-template.md)](references/regression-template.md)
- [标准任务脚本骨架 (task-script-skeleton.py)](references/task-script-skeleton.py)
