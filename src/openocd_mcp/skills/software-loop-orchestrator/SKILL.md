---
name: software-loop-orchestrator
description: Skill for software-side closed-loop orchestration across project generation, build, artifact consistency, debug preparation, triage routing, deep analysis handoff, and regression closure.
---

# 软件侧闭环全局编排指南 (Software Loop Orchestrator)

## 1. 角色定位

该 skill 是整个软件侧闭环的**方向编排层**，用于指导 LLM 在 `keil2cmake` 与并入后的 `openocd-mcp` 之间做全局目标判断、方向收束、Agent 工作项生成与技能交接。

主责包括：
- 明确当前总目标、成功条件与约束
- 识别最新信号属于哪个关注域：`engineering`、`build`、`artifact`、`debug`、`runtime`、`validation`
- 决定下一步方向，而不是写死具体补丁脚本
- 在失败时生成 Agent 工作项，让 Agent 自主搜集信息、修改工作区并立即验证
- 保持闭环稳定：目标 -> 信号 -> 决策 -> 执行 -> 验证 -> 反思 -> 交接

它不负责：
- 直接替代 `openocd-core-operations` 执行底层调试命令
- 直接替代 `openocd-deep-debug-analysis` 做根因解释
- 直接替代 `openocd-automation-validation` 执行脚本回归
- 在缺乏证据时盲目大范围改代码
- 绕过高风险确认去执行刷写、批量删除或不可逆修改

## 2. 何时使用

当任务满足以下任一条件时，应优先使用本 skill：

- 需要从“代码改动”进入“构建 -> 调试 -> 验证”的完整链路
- 需要判断当前是 build 问题、flash/debug 问题，还是运行期问题
- 需要保证 ELF / SVD / openocd.cfg / 构建目录来自同一轮工件
- 需要在多个技能之间做稳定交接，而不是单点处理
- 需要让 LLM 以“流程脑”而不是“单工具脑”方式工作

## 3. 标准流程

### 3.1 相位识别

优先使用方向编排相位，而不是旧式固定阶段：

- `understand`
- `collect`
- `decide`
- `execute`
- `verify`
- `reflect`
- `handoff`
- `done`

再结合当前关注域：

- `engineering`
- `build`
- `artifact`
- `debug`
- `runtime`
- `validation`
- `unknown`

### 3.2 全局闭环顺序

推荐顺序为：
1. 明确目标与成功条件
2. 获取当前信号
3. 收束方向并生成下一步工作项
4. 执行一轮最小动作
5. 立即验证
6. 根据结果进入反思、交接或完成

### 3.3 工件一致性检查

在进入调试与分析前，至少确认：
- 当前源码版本与构建产物匹配
- ELF 与目标下载镜像匹配
- SVD 与芯片型号匹配
- `openocd.cfg` 与接口/目标脚本匹配

工件一致性字段应使用以下固定值：
- `unknown`
- `pending`
- `consistent`
- `inconsistent`

### 3.4 失败回路

若某动作失败，应按以下方式回流：
- configure / build 失败 -> 进入 `decide` 并生成 `AgentWorkItem`
- 调试准备失败 -> 进入 `reflect` 并重新审视工件一致性
- 运行期异常 -> 进入 `collect` 并交给 `openocd-triage-and-capture`
- 分析后仍不确定 -> 补抓证据后重进 `decide`
- 回归失败 -> 进入 `reflect`，再决定是否回到分析或修复

### 3.5 Agent 自主修复回路

默认行为不是“等人明确命令后再修”，而是：
- 编排层先给出目标、约束、方向与候选文件
- Agent 自主决定先读什么、改什么、验证什么
- 若证据与风险边界都足够清晰，则直接推进一轮最小修复
- 若问题超出当前把握范围，再显式停下并请求更多信息或确认

当满足以下条件时，可以主动进入自动修复：
- 已有明确失败证据，例如编译错误、链接错误、配置缺失、工件不一致、稳定复现的运行期异常
- 修改范围可控制在小步内，且能明确说明为什么改这几个文件
- 修复后存在可立即执行的验证动作，例如 re-configure、re-build、re-triage、re-regression

推荐修复顺序：
1. 总结当前失败信号
2. 选择一个最小方向动作
3. 只修改最相关的一组文件
4. 立即重跑验证动作
5. 对比新旧信号，决定继续、反思还是交接

自动修复时的默认工作纪律：
- 单轮优先只解决一个主因骨，不混合多个大问题
- 单轮优先只做一组相关补丁，避免“大爆炸”式改动
- 如果连续 2~3 轮仍无实质进展，应主动停止自动修复并回到分诊/分析
- 若需要高风险操作，再停下并请求明确确认

换句话说：
- 对常见 build / engineering / artifact 问题，Agent 应表现出主动性
- 对高风险、证据不足、影响范围不清的问题，Agent 应表现出克制

适合自动修复的典型问题：
- `engineering`：路径错误、模板缺失、preset/生成参数不一致
- `build`：编译选项、宏、include、链接输入缺失
- `artifact`：ELF / SVD / `openocd.cfg` 不一致或引用错误

不适合直接自动修复的典型问题：
- 证据不足的复杂运行期异常
- 涉及大规模重构的架构问题
- 需要真实硬件确认但当前现场信息缺失的问题

## 4. 输出要求

输出至少应包含：
- `summary`：当前闭环进行到哪一步
- `current_state`：当前阶段、阶段状态、鱼骨主因骨、已准备工件、已完成动作
- `evidence`：当前阶段所依赖的关键事实，例如构建结果、工件路径、失败样本
- `hypothesis`：为什么下一步应该进入某个 skill
- `next_action`：明确的下一阶段动作
- `risk_or_limit`：当前缺失的工件、配置、权限或现场信息

若进入自动修复，还应额外输出：
- `repair_goal`：本轮要修什么
- `repair_scope`：准备修改哪些文件、哪些文件明确不动
- `repair_guardrails`：停止条件、回滚条件、验证条件
- `repair_result`：修复后结果是成功、部分成功还是失败

若 Agent 已根据错误自行发起修复，还建议补充：
- `repair_reasoning`：为什么当前错误足以支撑这轮修复
- `repair_confidence`：当前修复判断的把握程度（如 `high` / `medium` / `low`）

与代码层完全对齐时，应优先使用以下状态字段：
- `workflow_phase`
- `phase_status`
- `focus_domain`
- `artifact_consistency`
- `current_goal`
- `current_signal`
- `active_work_item`
- `planned_actions`
- `completed_actions`
- `pending_action`
- `handoff_skill`
- `last_error`

若自动修复已发生，建议在 `summary` 与 `evidence` 中显式补充：
- 修复前失败证据
- 本轮补丁范围
- 修复后验证结果
- 是否进入下一轮自动修复或停止

对于跨技能交接，必须说明：
- 本轮工件基线是什么
- 哪些动作已经执行
- 哪些动作尚未执行
- 为什么要交给下一个 skill

## 5. 路由规则

- 工程生成、最小构建准备完成后，交给构建与配置执行路径
- 调试准备动作交给 `openocd-core-operations`
- 运行期异常现场交给 `openocd-triage-and-capture`
- 根因解释交给 `openocd-deep-debug-analysis`
- 批量验证交给 `openocd-automation-validation`
- 当 `engineering` / `build` / `artifact` 问题证据充分时，Agent 默认可先执行一轮最小自动修复，再重新进入对应阶段
- 当 `runtime` / `validation` 问题证据不足时，优先补证据，不要直接改代码

鱼骨分因的推荐落点：
- `engineering` -> 工程生成、模板、路径、uvprojx 解析
- `build` -> configure、toolchain、preset、编译与链接
- `artifact` -> ELF / BIN / SVD / openocd.cfg 一致性
- `debug` -> OpenOCD 连接、halt、reset、flash、断点资源
- `runtime` -> HardFault、死锁、外设异常、运行期卡死
- `validation` -> 回归失败、样本波动、flaky case

特别强调：

- 本 skill 负责“决定去哪”，不是“替别人把所有事都做完”
- 若某个 skill 越界，应回收到本 skill 重新做阶段判断
- 自动修复属于**编排授权下的受限动作**，不是无限制自动改写
- “受限”不等于“被动等待”；对已识别的常见问题，Agent 应主动推进修复和验证

## 6. 共享规范

本 skill 严格继承并遵守：
- [COMMON-NORMS.md](../shared/COMMON-NORMS.md)
- [SKILL-STYLE-GUIDE.md](../shared/SKILL-STYLE-GUIDE.md)

特别强调三条：
1. 工件不一致时，不进入深度分析。
2. 阶段不清楚时，先做阶段判定，不直接执行高风险动作。
3. 路由必须带交接信息，而不是只给一句“交给某 skill”。
4. 自动修复必须以证据为前提，以最小补丁为原则，以验证结果为结束条件。

实践上应理解为：
- 默认允许 Agent 在低风险、证据充分场景下自主推进
- 默认要求 Agent 在高风险、证据不足场景下收敛动作
- 严格约束首先作用于流程，而不是对正常主动性的简单限制

## 7. 参考资料
- [闭环阶段矩阵 (loop-phase-matrix.md)](references/loop-phase-matrix.md)
- [工件一致性清单 (artifact-consistency-checklist.md)](references/artifact-consistency-checklist.md)
- [路由决策模板 (routing-decision-template.md)](references/routing-decision-template.md)
- [鱼骨分因路由图 (fishbone-routing-map.md)](references/fishbone-routing-map.md)
- [自动修复回路模板 (auto-repair-loop-template.md)](references/auto-repair-loop-template.md)