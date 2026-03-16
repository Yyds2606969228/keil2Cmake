# RTOS 级任务分诊清单

## 1. 目标 (Objective)
专门处理 FreeRTOS、RT-Thread 等系统的心跳挂死、任务饥饿、调度停滞问题。

## 2. 输入 (Input)
- RTOS awareness 已开启的 target
- 任务列表、优先级、栈使用率（若可得）
- SysTick 或调度相关寄存器信息

## 3. 步骤 (Steps)
1. 获取全部任务列表与状态。
2. 对比 `Running`、`Ready`、`Blocked`、`Suspended` 分布。
3. 检查 Idle 任务和定时器任务是否被长期饿死。
4. 若支持，检查每个任务的栈余量、等待对象和最后活动点。

## 4. 输出 (Output)
- 任务状态地图
- 可疑任务列表
- 调度层面的初步解释

## 5. 风险 (Risks)
- TCB 或链表头损坏会导致任务视图不可信。
- 只观察当前运行任务会忽略被阻塞链条。
- 调度器停摆时，状态字段可能滞后于真实问题发生点。

## 6. 示例 (Example)
```text
Idle: never scheduled in last observation window
Task_A: Running, priority=7
Task_A stack watermark: critically low
初判：高优先级任务忙循环并伴随潜在栈风险
```
