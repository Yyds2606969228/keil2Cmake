# 证据组装模板

## 1. 目标 (Objective)
统一异常现场捕获阶段的证据报告格式，确保后续分析能够直接消费，不需要二次猜测上下文。

## 2. 输入 (Input)
- 冻结时刻的 PC、SP、LR
- CPU 寄存器快照 `reg`
- 故障寄存器值
- 串口、RTT 或其他辅助日志片段

## 3. 步骤 (Steps)
1. 记录冻结时刻与目标状态。
2. 提取通用寄存器、异常寄存器与最小必要内存片段。
3. 按“原始值 + 解释 + 局限”格式整理每条证据。
4. 补充串口或 RTT 关键片段，并注明时间关系。

## 4. 输出 (Output)
- Markdown 或 JSON 风格的结构化证据包
- 可直接交给深度分析 skill 的事实清单

## 5. 风险 (Risks)
- 未及时 `halt` 会导致现场被 WDT 或业务逻辑覆盖。
- 日志与寄存器时间线错位会造成伪相关。
- 只保留解释不保留原始值，会导致后续无法复核。

## 6. 示例 (Example)
```text
[freeze_time]  : 2026-03-06T10:20:00
[target_state] : halted
[pc]           : 0x08001234
[fault_regs]   : CFSR=0x00008200, BFAR=0x2000FFF8
[serial_tail]  : "malloc failed\r\nrebooting..."
[note]         : 现场已冻结，未执行 reset
```
