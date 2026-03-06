"""任务脚本骨架 (Task Script Skeleton)

目标:
- 为受限 Sandbox 中的自动化脚本提供统一入口
- 强制要求显式超时、显式断言、显式清理

输入:
- `context`: 由运行时注入的能力对象
- `max_iterations`: 最大执行轮次

输出:
- 统一结构化结果，便于上层聚合与交接

风险:
- 无限循环、无清理、吞异常都会降低可维护性
"""

from __future__ import annotations

from time import monotonic


def _result(status: str, summary: str, **extra):
    payload = {
        "status": status,
        "summary": summary,
    }
    payload.update(extra)
    return payload


def run_task(context, max_iterations: int = 10, timeout_s: float = 30.0):
    if max_iterations <= 0:
        return _result("Failed", "`max_iterations` 必须大于 0", error_code="INVALID_ARGUMENT")

    if timeout_s <= 0:
        return _result("Failed", "`timeout_s` 必须大于 0", error_code="INVALID_ARGUMENT")

    start = monotonic()
    completed = 0
    failed_samples = []
    halted = False

    try:
        for i in range(max_iterations):
            if monotonic() - start > timeout_s:
                return _result(
                    "Failed",
                    "任务超时",
                    error_code="TIMEOUT",
                    completed_iterations=completed,
                    failed_samples=failed_samples,
                )

            context.execute_tcl("halt")
            halted = True

            value = context.read_memory(0x20000000)
            if value == 0x00000000:
                failed_samples.append({
                    "iteration": i,
                    "address": "0x20000000",
                    "value": hex(value),
                })
                raise AssertionError(f"iteration {i}: unexpected zero value at 0x20000000")

            context.execute_tcl("resume")
            halted = False
            completed += 1

        duration_ms = int((monotonic() - start) * 1000)
        return _result(
            "Passed",
            "任务执行完成",
            completed_iterations=completed,
            duration_ms=duration_ms,
            failed_samples=failed_samples,
        )

    except Exception as exc:
        duration_ms = int((monotonic() - start) * 1000)
        return _result(
            "Failed",
            str(exc),
            error_code="TASK_EXECUTION_FAILED",
            completed_iterations=completed,
            duration_ms=duration_ms,
            failed_samples=failed_samples,
        )

    finally:
        if halted:
            try:
                context.execute_tcl("resume")
            except Exception:
                pass
