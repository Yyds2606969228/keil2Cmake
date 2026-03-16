"""Task runtime for executing short Python scripts with sandbox restrictions."""

from __future__ import annotations

import ast
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
from threading import Event, RLock, Thread
import time
from types import FrameType
from typing import Any, Callable
import sys

from ..core.errors import MCPServiceError
from .context import DebugContext
from .events import EventBus

ContextFactory = Callable[[Event], DebugContext]


class TaskCancelled(RuntimeError):
    """Raised inside task thread to force cancellation."""


class CallbackTimeout(RuntimeError):
    """Raised inside callback thread to force timeout exit."""


@dataclass(slots=True)
class TaskInfo:
    name: str
    status: str = "pending"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None
    thread: Thread | None = None
    cancel_event: Event = field(default_factory=Event)
    context: DebugContext | None = None
    callbacks: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    callback_failures: list[str] = field(default_factory=list)
    disabled_callbacks: set[str] = field(default_factory=set)
    force_stop_failed: bool = False


class TaskRuntime:
    def __init__(self, event_bus: EventBus | None = None, callback_timeout_ms: int = 200) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = RLock()
        self._event_bus = event_bus or EventBus()
        self._callback_timeout_ms = callback_timeout_ms

    @staticmethod
    def _validate_script(code: str) -> None:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise MCPServiceError("SCR_SYNTAX_ERROR", str(exc)) from exc
        disallowed_nodes = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith)
        for node in ast.walk(tree):
            if isinstance(node, disallowed_nodes):
                raise MCPServiceError("SCR_SECURITY_VIOLATION", f"Disallowed statement: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in {"open", "__import__", "eval", "exec", "compile", "input"}:
                raise MCPServiceError("SCR_SECURITY_VIOLATION", f"Disallowed symbol: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise MCPServiceError("SCR_SECURITY_VIOLATION", "Dunder attribute access is blocked.")

    def submit(self, name: str, code: str, context_factory: ContextFactory, timeout_ms: int = 30_000) -> str:
        if not name.strip():
            raise MCPServiceError("SCR_INVALID_NAME", "Task name cannot be empty.")
        self._validate_script(code)
        with self._lock:
            existing = self._tasks.get(name)
            if existing and existing.status in {"pending", "running"}:
                raise MCPServiceError("SCR_TASK_EXISTS", f"Task already running: {name}")
            task = TaskInfo(name=name, status="running")
            self._tasks[name] = task
        thread = Thread(
            target=self._run_task,
            args=(task, code, context_factory, timeout_ms),
            name=f"task-{name}",
            daemon=True,
        )
        task.thread = thread
        thread.start()
        return name

    def _run_task(self, task: TaskInfo, code: str, context_factory: ContextFactory, timeout_ms: int) -> None:
        start = time.monotonic()
        deadline = start + timeout_ms / 1000.0
        task.context = context_factory(task.cancel_event)
        callbacks: dict[str, list[Callable[..., Any]]] = {}

        def on(event_name: str, fn: Callable[..., Any]) -> None:
            if not callable(fn):
                raise TypeError("Callback must be callable.")
            callbacks.setdefault(event_name, []).append(fn)

        def trace(frame: FrameType, event: str, arg: object) -> Callable[[FrameType, str, object], Any]:
            _ = (frame, event, arg)
            if task.cancel_event.is_set():
                raise TaskCancelled("task_cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("task_timeout")
            return trace

        allowed_builtins: dict[str, Any] = {
            "Exception": Exception,
            "TypeError": TypeError,
            "ValueError": ValueError,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
        globals_dict = {"__builtins__": allowed_builtins}
        locals_dict: dict[str, Any] = {
            "ctx": task.context,
            "debugger": task.context.debugger,
            "serial": task.context.serial,
            "on": on,
        }
        try:
            sys.settrace(trace)
            exec(code, globals_dict, locals_dict)
            task.status = "completed"
        except TimeoutError:
            task.status = "timeout"
            task.error = "task_timeout"
        except TaskCancelled:
            task.status = "cancelled"
            task.error = "task_cancelled"
        except RuntimeError as exc:
            task.status = "failed"
            task.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            task.error = f"{exc.__class__.__name__}: {exc}"
        finally:
            sys.settrace(None)
            task.finished_at = datetime.now(timezone.utc)

        if task.status in {"completed", "cancelled", "timeout"}:
            task.callbacks = callbacks
            self._bind_callbacks(task)

    def _bind_callbacks(self, task: TaskInfo) -> None:
        for event_name, fns in task.callbacks.items():
            for fn in fns:
                key = f"{event_name}:{getattr(fn, '__name__', 'lambda')}"
                self._event_bus.on(event_name, self._make_callback(task, fn, key))

    def _make_callback(
        self,
        task: TaskInfo,
        fn: Callable[..., Any],
        callback_key: str,
    ) -> Callable[[dict[str, Any]], None]:
        def wrapper(payload: dict[str, Any]) -> None:
            if callback_key in task.disabled_callbacks:
                return
            ctx = task.context
            if ctx is None:
                return

            error_holder: list[str] = []

            def run_callback() -> None:
                try:
                    param_count = len(inspect.signature(fn).parameters)
                except (TypeError, ValueError):
                    param_count = 0
                try:
                    if param_count >= 2:
                        fn(ctx, payload)
                    elif param_count == 1:
                        fn(payload)
                    else:
                        fn()
                except Exception as exc:  # noqa: BLE001
                    error_holder.append(f"{callback_key}: {exc.__class__.__name__}: {exc}")

            thread = Thread(target=run_callback, name=f"cb-{callback_key}", daemon=True)
            thread.start()
            thread.join(timeout=self._callback_timeout_ms / 1000.0)
            if thread.is_alive():
                self._async_raise(thread, CallbackTimeout)
                thread.join(timeout=0.05)
                task.disabled_callbacks.add(callback_key)
                if thread.is_alive():
                    task.callback_failures.append(f"{callback_key}: timeout(force_stop_failed)")
                else:
                    task.callback_failures.append(f"{callback_key}: timeout")
                return
            if error_holder:
                task.disabled_callbacks.add(callback_key)
                task.callback_failures.extend(error_holder)

        return wrapper

    def emit(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        self._event_bus.emit(event_name, payload or {})

    def get(self, name: str) -> dict[str, Any]:
        task = self._tasks.get(name)
        if task is None:
            raise MCPServiceError("SCR_TASK_NOT_FOUND", f"Task does not exist: {name}")
        data: dict[str, Any] = {
            "id": task.name,
            "name": task.name,
            "status": task.status,
            "error": task.error,
            "started_at": task.started_at.isoformat(),
            "finished_at": task.finished_at.isoformat() if task.finished_at else None,
            "disabled_callbacks": sorted(task.disabled_callbacks),
            "callback_failures": list(task.callback_failures),
            "force_stop_failed": task.force_stop_failed,
        }
        if task.context is not None:
            data["result"] = task.context.snapshot()
        return data

    @staticmethod
    def _async_raise(thread: Thread, exc_type: type[BaseException]) -> bool:
        ident = thread.ident
        if ident is None:
            return False
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(  # type: ignore[attr-defined]
            ctypes.c_ulong(ident),
            ctypes.py_object(exc_type),
        )
        if result == 0:
            return False
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(  # type: ignore[attr-defined]
                ctypes.c_ulong(ident),
                None,
            )
            return False
        return True

    def _force_cancel_task(self, task: TaskInfo, wait_timeout_s: float) -> bool:
        thread = task.thread
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=min(wait_timeout_s, 0.05))
        if not thread.is_alive():
            return True
        deadline = time.monotonic() + min(wait_timeout_s, 0.05)
        while thread.ident is None and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        raised = self._async_raise(thread, TaskCancelled)
        if not raised:
            task.force_stop_failed = True
            return False
        thread.join(timeout=wait_timeout_s)
        if thread.is_alive():
            task.force_stop_failed = True
            return False
        return True

    def cancel(self, name: str, *, force: bool = False, wait_timeout_s: float = 0.2) -> bool:
        task = self._tasks.get(name)
        if task is None:
            raise MCPServiceError("SCR_TASK_NOT_FOUND", f"Task does not exist: {name}")
        task.cancel_event.set()
        if force:
            return self._force_cancel_task(task, wait_timeout_s)
        return True

    def cancel_all(self, *, force: bool = False, wait_timeout_s: float = 0.2) -> dict[str, list[str]]:
        with self._lock:
            tasks = list(self._tasks.values())
        report = {"signaled": [], "stopped": [], "alive": []}
        for task in tasks:
            task.cancel_event.set()
            report["signaled"].append(task.name)
            if force:
                if self._force_cancel_task(task, wait_timeout_s):
                    report["stopped"].append(task.name)
                else:
                    report["alive"].append(task.name)
            elif task.thread is not None and task.thread.is_alive():
                report["alive"].append(task.name)
        return report

    def wait(self, name: str, timeout_s: float = 2.0) -> dict[str, Any]:
        task = self._tasks.get(name)
        if task is None:
            raise MCPServiceError("SCR_TASK_NOT_FOUND", f"Task does not exist: {name}")
        if task.thread and task.thread.is_alive():
            task.thread.join(timeout=timeout_s)
        return self.get(name)
