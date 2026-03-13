---
name: openocd-deep-debug-analysis
description: Skill for expert-level root-cause analysis using ELF, SVD, RTOS introspection, stack and heap forensics, peripheral decoding, and multicore correlation.
---

# OpenOCD 深度调试与根因分析指南 (Deep Debug Analysis)

## 1. 角色定位

该 skill 负责把“已经采集到的现场”转化为“可信的根因链”。

主责包括：
- 基于 ELF 还原函数、行号、符号与调用链
- 基于 SVD 解码外设寄存器语义
- 对栈、堆、任务、锁、核间状态进行结构化分析
- 把多个证据碎片拼成可验证的根因假设

本 skill 不应替代：
- `core` 的会话与刷写控制
- `triage` 的第一现场保全
- `automation` 的批量验证与回归执行

## 2. 何时使用

当满足以下任一条件时，应使用本 skill：

- 现场已经拿到，但原因仍不明确
- 需要把 PC/SP/LR 映射回函数、源码位置或异常栈帧
- 需要解释外设寄存器位域含义，而不是只看十六进制值
- 需要分析 RTOS 调度、死锁、优先级反转、任务饥饿
- 需要分析多核同步、核间状态不一致、共享资源竞争
- 需要判断栈溢出、堆损坏或内存踩踏

## 3. 标准流程

### 3.1 输入归一化

先确认可用输入：
- triage 交接快照
- `.elf` 文件
- `.svd` 文件
- 目标架构与内存布局
- 相关外设或 RTOS 背景

若关键输入缺失，应先说明缺失项对结论的影响范围。

### 3.2 三层映射分析

1. **指令层**
   - 根据 PC、LR、xPSR 判断崩溃点或停机点
   - 必要时结合反汇编确认当前指令语义
2. **符号层**
   - 用 ELF 把地址映射到函数、源码位置、局部上下文
   - 还原调用链与潜在返回路径
3. **硬件层**
   - 用 SVD 解码寄存器位域
   - 审查时钟、复位、GPIO 复用、外设使能、错误标志

### 3.3 结构化推理

根因推理应遵循：
1. 列出已知事实
2. 列出候选解释
3. 用证据逐项排除
4. 保留最小充分解释

### 3.4 验证建议

对每个主要结论，应附至少一个验证建议，例如：
- 再读取某个寄存器位
- 切换某个外设配置后复测
- 用自动化脚本重复触发并统计结果

## 4. 输出要求

输出至少应包含：
- `summary`：一句话说明最可能根因
- `current_state`：输入证据是否完整、分析前提是否成立
- `evidence`：地址、寄存器、符号映射、任务状态、位域解码
- `hypothesis`：根因链与排除过程
- `next_action`：建议验证动作或修复动作
- `risk_or_limit`：例如 ELF 版本不匹配、SVD 不完整、现场已被污染

所有关键判断应尽量包含：
- 地址
- 原始值
- 解释后的语义
- 为什么与结论相关

## 5. 路由规则

- 若需要按方案重新连接、恢复、刷写或改写目标，交回 `openocd-core-operations`
- 若需要批量重复验证、统计通过率或自动抓取失败样本，交给 `openocd-automation-validation`
- 若发现当前现场并不完整，应要求 `openocd-triage-and-capture` 重新补抓关键证据

交接时必须写明：
- 当前主结论
- 支撑证据
- 尚未验证的假设
- 推荐的最小下一步

## 6. 共享规范

本 skill 严格继承并遵守：
- [COMMON-NORMS.md](../shared/COMMON-NORMS.md)
- [SKILL-STYLE-GUIDE.md](../shared/SKILL-STYLE-GUIDE.md)

特别强调三条：
1. 深度分析不等于大胆猜测。
2. 每个结论都应有证据链。
3. 不能把验证建议当成既成事实。

## 7. 参考资料
- [Cortex-M 异常栈解析参考 (cortex-m-fault-stack.md)](references/cortex-m-fault-stack.md)
- [外设验证清单 (peripheral-checklist.md)](references/peripheral-checklist.md)
- [ELF/SVD 解析报告模板 (svd-decode-report-template.md)](references/svd-decode-report-template.md)
- [RTOS 任务级分诊清单 (rtos-triage-checklist.md)](references/rtos-triage-checklist.md)
