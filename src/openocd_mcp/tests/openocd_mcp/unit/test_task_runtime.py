from openocd_mcp.runtime.context import DebugContext
from openocd_mcp.runtime.task_runtime import TaskRuntime
from openocd_mcp.core.errors import MCPServiceError

from threading import Event

import pytest


def _ctx_factory(cancel_event: Event) -> DebugContext:
    mem: dict[int, int] = {}
    return DebugContext(
        read32=lambda addr: mem.get(addr, 0),
        write32=lambda addr, value: mem.__setitem__(addr, value),
        halt=lambda: None,
        resume=lambda: None,
        serial_write=lambda data: None,
        cancel_event=cancel_event,
    )


def test_task_runtime_completes_and_records() -> None:
    runtime = TaskRuntime()
    code = "ctx.record({'k': 1})\nctx.log('done')\nctx.log_data({'x': 2})"
    task_id = runtime.submit("t1", code, _ctx_factory, timeout_ms=1000)
    result = runtime.wait(task_id, timeout_s=1.0)
    assert result["status"] == "completed"
    assert result["result"]["records"] == [{"k": 1}, {"x": 2}]
    assert result["result"]["logs"] == ["done"]


def test_task_runtime_timeout() -> None:
    runtime = TaskRuntime()
    code = "while True:\n    x = 1"
    task_id = runtime.submit("timeout-task", code, _ctx_factory, timeout_ms=50)
    result = runtime.wait(task_id, timeout_s=1.0)
    assert result["status"] == "timeout"


def test_task_runtime_event_callback() -> None:
    runtime = TaskRuntime()
    code = (
        "def on_halt(ctx, payload):\n"
        "    ctx.record({'pc': payload.get('pc', 0)})\n"
        "on('TARGET_HALTED', on_halt)\n"
    )
    task_id = runtime.submit("evt", code, _ctx_factory, timeout_ms=1000)
    result = runtime.wait(task_id, timeout_s=1.0)
    assert result["status"] == "completed"
    runtime.emit("TARGET_HALTED", {"pc": 0x1234})
    result2 = runtime.get(task_id)
    assert result2["result"]["records"] == [{"pc": 0x1234}]


def test_task_runtime_blocks_import_and_open() -> None:
    runtime = TaskRuntime()
    with pytest.raises(MCPServiceError) as import_err:
        runtime.submit("bad-import", "import os", _ctx_factory, timeout_ms=1000)
    assert import_err.value.error_code == "SCR_SECURITY_VIOLATION"
    with pytest.raises(MCPServiceError) as open_err:
        runtime.submit("bad-open", "open('x')", _ctx_factory, timeout_ms=1000)
    assert open_err.value.error_code == "SCR_SECURITY_VIOLATION"


def test_task_runtime_force_cancel() -> None:
    runtime = TaskRuntime()
    task_id = runtime.submit("force-cancel", "while True:\n    x = 1", _ctx_factory, timeout_ms=10_000)
    assert runtime.cancel(task_id, force=True, wait_timeout_s=0.5) is True
    result = runtime.wait(task_id, timeout_s=0.5)
    assert result["status"] == "cancelled"


def test_callback_timeout_disables_callback() -> None:
    runtime = TaskRuntime(callback_timeout_ms=1)
    code = (
        "def on_halt(ctx, payload):\n"
        "    x = 0\n"
        "    while x < 10_000_000:\n"
        "        x += 1\n"
        "on('TARGET_HALTED', on_halt)\n"
    )
    task_id = runtime.submit("slow-cb", code, _ctx_factory, timeout_ms=1000)
    runtime.wait(task_id, timeout_s=1.0)
    runtime.emit("TARGET_HALTED", {"pc": 0x10})
    result = runtime.get(task_id)
    assert result["disabled_callbacks"]
    assert result["callback_failures"]
