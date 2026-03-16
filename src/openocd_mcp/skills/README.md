# OpenOCD Skills

本目录用于描述 `openocd-mcp` 的技能体系。整体原则是：**按责任分层，而不是按零碎场景堆叠**。

当前结构已从原先 10 个分散 skill 收敛为 4 个 OpenOCD 主 skill，并新增 1 个全局软件闭环编排 skill。

## 当前结构

- `openocd-core-operations`：连接、控制、安全门禁、刷写、恢复
- `openocd-triage-and-capture`：故障初筛、双通道取证、现场保全
- `openocd-deep-debug-analysis`：SVD/ELF、外设、RTOS、多核根因分析
- `openocd-automation-validation`：Python 沙箱、回归验证、自动化取证
- `software-loop-orchestrator`：软件侧闭环阶段识别、全局路由、工件一致性、受限自动修复与流程指导
- `shared`：跨 skill 公共规范与写作风格

## 设计目标

该目录的目标不是“覆盖所有名词”，而是让 Agent 在面对问题时能：
1. 快速选对入口 skill。
2. 用统一流程执行。
3. 输出结构稳定、证据可审计。
4. 在需要时顺畅交接给下一个 skill。

## 技能边界

### 1. `openocd-core-operations`
负责与目标建立最小安全控制闭环：连接、暂停、读写、刷写、恢复。

### 2. `openocd-triage-and-capture`
负责第一现场保全：先冻结，再取证，再形成初步分诊结论。

### 3. `openocd-deep-debug-analysis`
负责深度解释：把寄存器、堆栈、符号、外设、RTOS、多核信息拼成根因链。

### 4. `openocd-automation-validation`
负责批量执行与回归：把一次性诊断动作固化成脚本和重复验证流程。

### 5. `software-loop-orchestrator`
负责全局编排：把工程生成、构建、调试准备、现场取证、根因分析、受限自动修复和回归验证串成稳定流程。

## 合并映射

- `openocd-mcp-usage` + `openocd-session-safety` + `openocd-flashing-provisioning`
  -> `openocd-core-operations`
- `openocd-fault-triage` + `openocd-dual-channel-coordination`
  -> `openocd-triage-and-capture`
- `openocd-svd-elf-analysis` + `openocd-hardware-peripheral-debug` + `openocd-rtos-multicore-debug`
  -> `openocd-deep-debug-analysis`
- `openocd-python-sandbox-development` + `openocd-regression-validation`
  -> `openocd-automation-validation`

## 统一风格

5 个主 skill 统一采用以下章节顺序：

1. 角色定位
2. 何时使用
3. 标准流程
4. 输出要求
5. 路由规则
6. 共享规范
7. 参考资料

详细风格约束见：
- `shared/SKILL-STYLE-GUIDE.md`

## 共享规范

所有 skill 共用以下执行原则：

- Safety First
- Evidence First
- Structured Output
- Deterministic Flow
- Routing Discipline
- Cleanup Discipline

详细规范见：
- `shared/COMMON-NORMS.md`

## 维护建议

1. 新增 skill 前，优先判断是否能归入现有 4 类。
2. 修改 skill 时，优先复用 `shared` 中的规范，不重复复制通用条款。
3. 若发现两个 skill 长期共享同一触发条件，应考虑继续收敛。
4. 若发现单个 skill 经常跨边界工作，应先修正文档，再修正提示词。

## 推荐维护顺序

当需要强化技能体系时，优先修改顺序为：
1. `shared/COMMON-NORMS.md`
2. `shared/SKILL-STYLE-GUIDE.md`
3. 主 `SKILL.md`
4. `agents/openai.yaml`
5. `references/*`

这样可以确保先收紧规则，再更新行为，再补模板。