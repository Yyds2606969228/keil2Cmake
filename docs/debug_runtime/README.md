# Debug Runtime Integration

本目录承载从 `openocd-mcp` 并入 `keil2cmake` 的调试运行时文档。

## 组件定位

该组件负责：
- OpenOCD 连接、控制与刷写保护门禁
- 串口缓冲、触发与双通道取证
- SVD / ELF 解析与结构化输出
- Python 任务运行时与自动化验证
- 面向 Agent 的 MCP 调试接口

## 与主仓分工

- `keil2cmake` 主体负责工程生成、构建配置、工具链与产物管理
- `openocd-mcp` 组件负责运行期调试、现场抓取、根因分析支撑与自动化执行
- `keil2cmake.orchestrator` 负责全局方向编排、状态推进与 handoff 决策

## 当前并入内容

- 源码：`src/openocd_mcp/`
- 测试：`tests/openocd_mcp/`
- 技能：`skills/`
- 协议文档：
  - `api_contract.md`
  - `state_model.md`
  - `error_codes.md`

## 分发模型

当前推荐采用 **双入口 EXE 分发**：

- `Keil2Cmake.exe`：主入口，承载工程工具链能力与全局编排
- `openocd-mcp.exe`：MCP 服务入口，供上层客户端以 `stdio` 拉起

该模型下：

- 全局编排不会失效，仍由 `keil2cmake.orchestrator` 负责
- MCP 运行时只是被 handoff 的能力层，而不是新的“主脑”
- 默认不依赖固定 TCP 端口，因此不会引入典型端口冲突问题

## 当前演进状态

当前主仓已经具备：

- 方向编排层
- 调试运行时
- 工件一致性管理
- Agent 工作项与验证回环

后续演进重点应放在：

- 更稳定的双 EXE 打包脚本
- MCP 客户端接入示例
- 主入口与 MCP 服务入口的发布流程固化