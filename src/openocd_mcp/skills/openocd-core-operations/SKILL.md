---
name: openocd-core-operations
description: Unified entry skill for safe session bring-up, target control, inspection, controlled flash operations, and recovery on unknown boards.
---

# OpenOCD 核心操作与控制指南 (Core Operations)

## 1. 角色定位

该 skill 是 `openocd-mcp` 的默认入口，负责建立与目标芯片之间的**最小安全控制闭环**。

主责包括：
- 建立或恢复调试会话
- 控制目标执行状态，例如 `halt`、`resume`、`step`
- 读取寄存器、内存、目标状态
- 在授权前提下执行有限写入、断点、观察点和刷写动作
- 在异常或退出时做安全收尾

本 skill 的定位是“**安全控制层**”，不是“根因分析层”。

它负责：
- 保证目标可控
- 保证动作边界清楚
- 保证高风险操作可审计

它不负责：
- 对复杂异常做深度解释
- 对 RTOS、多核、SVD/ELF 做复杂推理
- 对长流程测试做批量编排

## 2. 何时使用

当任务满足以下任一条件时，应优先使用本 skill：

- 首次连接未知目标板，需要确认链路是否可用
- 需要立即 `halt` 目标，阻止继续跑飞或覆盖现场
- 需要读取寄存器、内存、PC、SP、target state 等基础信息
- 需要设置断点、观察点或最小范围写入
- 需要执行受控刷写、校验镜像或恢复调试会话
- 需要在结束前做统一清理与退出

如果问题已经进入“为什么会崩”“为什么调度异常”“为什么外设失效”这类解释阶段，应转交给更专门的 skill。

## 3. 标准流程

### 3.1 前置检查

在任何控制动作之前，先确认：
- 调试器与目标供电关系清楚
- 接口类型明确，例如 SWD 或 JTAG
- 当前是否允许改变目标运行状态
- 是否存在高风险动作审批要求

### 3.2 最小安全流程

1. **状态探测连接**
   - 优先采用无损连接，例如 `init`、`reset init`
   - 若失败，降低适配器频率后重试
   - 若仍失败，只报告链路事实，不盲目切换破坏性策略
2. **控制确权**
   - 在执行任何写入前，先 `halt`
   - 验证目标确已进入 `target halted` 状态
   - 记录 PC、SP、target state 作为当前基线
3. **执行动作**
   - 读取类动作：限定地址范围、长度和目标空间
   - 写入类动作：说明影响范围，优先提供 Dry Run 或验证计划
   - 刷写类动作：必须附带镜像路径、地址、校验方案与失败回退说明
4. **结果校验**
   - 对读取结果，说明值、地址、解释和局限
   - 对写入/刷写结果，执行回读或 `verify` 类验证
5. **优雅退出**
   - 清理断点、观察点、临时配置
   - 按用户意图执行 `resume`、`halt 保持` 或 `shutdown`

### 3.3 失败处理

若任一步骤失败，应优先报告以下事实：
- 失败阶段
- OpenOCD 原始输出
- 当前目标是否仍可控
- 下一步最小恢复动作

禁止在以下情况下继续写入：
- 未确认目标已 `halt`
- 未确认目标地址空间是否有效
- 未获得用户对高风险动作的明确授权

## 4. 输出要求

输出至少应包含：
- `summary`：本次控制动作完成了什么
- `current_state`：连接状态、target state、PC、SP、是否 halted
- `evidence`：地址、寄存器值、回读结果、OpenOCD 关键响应
- `next_action`：建议继续读取、交接分析，或安全收尾
- `risk_or_limit`：例如“当前未授权写入”或“目标已离线”
- `raw_output`：底层原始响应

若动作失败，还应补充：
- `success: false`
- `error_code`
- `message`

对于刷写、解锁、恢复类动作，必须额外给出：
- 影响区域
- 校验结果
- 是否已执行回退

## 5. 路由规则

- 若当前目标已经异常停机，需要先保全现场，路由到 `openocd-triage-and-capture`
- 若已经拿到现场证据，需要解释根因，路由到 `openocd-deep-debug-analysis`
- 若当前动作要被重复执行、统计通过率或进行长程验证，路由到 `openocd-automation-validation`

交接时至少提供：
- 当前连接状态
- 已执行的控制动作
- 已确认的地址、寄存器或错误输出
- 尚未获批的高风险动作

## 6. 共享规范

本 skill 严格继承并遵守：
- [COMMON-NORMS.md](../shared/COMMON-NORMS.md)
- [SKILL-STYLE-GUIDE.md](../shared/SKILL-STYLE-GUIDE.md)

特别强调三条：
1. 未授权不写。
2. 未 `halt` 不写。
3. 未校验不宣告成功。

## 7. 参考资料
- [安全门禁与连接规则 (safety-rules.md)](references/safety-rules.md)
- [会话检查清单 (session-checklist.md)](references/session-checklist.md)
- [标准操作流模板 (workflows.md)](references/workflows.md)
