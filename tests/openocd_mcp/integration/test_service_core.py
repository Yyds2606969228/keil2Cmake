from pathlib import Path
from time import sleep

from openocd_mcp.core.errors import MCPServiceError
from openocd_mcp.tools.service import OpenOCDMCPService
from openocd_mcp.parsers.svd_resolver import SVDResolver
from openocd_mcp.transport.serial.manager import SerialManager


class FakeOpenOCDClient:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.connected = False
        self.log_listener = None
        self.tcl_port = 6666
        self.telnet_port = 4444
        self.log_monitor_started = False
        self.last_start_config = None
        self.close_calls = 0

    def add_log_listener(self, callback) -> None:
        self.log_listener = callback

    def start_log_monitor(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        _ = (host, port)
        self.log_monitor_started = True

    def start(self, _config) -> None:
        self.last_start_config = _config
        self.connected = True

    def connect(self, host: str = "127.0.0.1", port: int = 6666) -> None:
        _ = (host, port)
        self.connected = True

    def execute(self, command: str, timeout_s: float = 2.0) -> str:
        _ = timeout_s
        self.commands.append(command)
        if command.startswith("mdw"):
            return "0x20000000: 12345678"
        if command == "version":
            return "Open On-Chip Debugger 0.12.0"
        if command == "reg pc":
            return "pc (/32): 0x08001234"
        return "ok"

    def control_target(self, action: str) -> str:
        self.commands.append(action)
        return "ok"

    def get_pc(self) -> str | None:
        return "0x08001234"

    def manage_breakpoint(
        self,
        *,
        point_type: str,
        address: int,
        action: str,
        length: int = 4,
        access: str = "w",
    ) -> str:
        self.commands.append(f"{point_type}:{action}:{address:x}:{length}:{access}")
        return "ok"

    def program(
        self,
        file_path: str,
        *,
        address: str | int | None = None,
        verify: bool = True,
        reset: bool = True,
        timeout_s: float = 60.0,
    ) -> str:
        self.commands.append(f"program:{file_path}:{address}:{verify}:{reset}:{timeout_s}")
        return "program ok"

    def close(self) -> None:
        self.close_calls += 1
        self.connected = False

    def clear_all_breakpoints(self, tracked_points=None):
        _ = tracked_points
        return {"cleared": {"bp": 0, "wp": 0}, "remaining": {"bp": [], "wp": []}, "errors": [], "verified": True}


class FakeSerialManager:
    def __init__(self) -> None:
        self.connected_cfg = None
        self.writes: list[tuple[str, str]] = []
        self.buffer = ["line-1", "line-2"]

    def connect(self, cfg) -> None:
        self.connected_cfg = cfg

    def write(self, data: str, mode: str = "ascii") -> None:
        self.writes.append((data, mode))

    def read_buffer(self, lines: int = 50, keyword: str | None = None):
        out = self.buffer[-lines:]
        if keyword:
            return [x for x in out if keyword in x]
        return out

    def set_trigger(self, regex: str, action: str, callback, **kwargs) -> None:
        _ = kwargs
        self._trigger = (regex, action, callback)

    def close(self) -> None:
        pass


def test_service_connect_debugger_returns_version() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.connect_debugger({"adapter": "stlink", "target": "stm32f4x", "auto_start": True})
    assert out["success"] is True
    assert "Open On-Chip Debugger" in out["data"]["version"]


def test_service_connect_debugger_uses_keil2cmake_openocd_path(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "path.cfg"
    cfg.write_text(
        "[PATHS]\nopenocd_path = C:/tool/openocd/bin/openocd.exe\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENOCD_MCP_KEIL2CMAKE_CFG", str(cfg))
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.connect_debugger({"adapter": "stlink", "target": "stm32f4x", "auto_start": True})
    assert out["success"] is True
    assert fake.last_start_config is not None
    assert fake.last_start_config.openocd_bin == "C:/tool/openocd/bin/openocd.exe"


def test_service_connect_serial_accepts_config_and_write_mode() -> None:
    fake = FakeOpenOCDClient()
    serial_mgr = FakeSerialManager()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=serial_mgr)
    out = service.connect_serial("COM9", 115200, {"parity": "E", "stopbits": 1, "bytesize": 8})
    assert out["success"] is True
    assert serial_mgr.connected_cfg is not None
    wr = service.serial_write("AA55", mode="hex")
    assert wr["success"] is True
    assert serial_mgr.writes[-1] == ("AA55", "hex")


def test_service_control_target_returns_pc() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.control_target("halt")
    assert out["success"] is True
    assert out["data"]["state"] == "halted"
    assert out["data"]["pc"] == "0x08001234"


def test_service_read_memory_parses_data() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.read_memory("0x20000000", 1, 32)
    assert out["success"] is True
    assert out["data"]["data"] == [0x12345678]


def test_service_write_memory_honors_width() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.write_memory("0x20000000", 0x1234, width=16)
    assert out["success"] is True
    assert any(cmd.startswith("mwh 0x20000000 0x1234") for cmd in fake.commands)


def test_service_write_memory_critical_requires_confirm() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    blocked = service.write_memory("0x40010000", 0x1, width=32)
    assert blocked["success"] is False
    assert blocked["error_code"] == "SAFE_CONFIRM_REQUIRED"
    dry = service.write_memory("0x40010000", 0x1, width=32, dry_run=True)
    assert dry["success"] is True
    assert dry["data"]["dry_run"] is True
    ok_write = service.write_memory("0x40010000", 0x1, width=32, confirm=True)
    assert ok_write["success"] is True


def test_service_serial_trigger_calls_halt_and_records_timestamp() -> None:
    fake = FakeOpenOCDClient()
    serial_mgr = SerialManager()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=serial_mgr)
    res = service.serial_set_trigger("HardFault", "halt")
    assert res["success"] is True
    serial_mgr.feed_line_for_test("HardFault happened")
    assert "halt" in fake.commands
    assert service._trigger_history  # noqa: SLF001
    assert "triggered_at" in service._trigger_history[-1]  # noqa: SLF001


def test_flash_program_requires_flash_mode(tmp_path: Path) -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    image = tmp_path / "firmware.bin"
    image.write_bytes(b"\x00\x01")
    out = service.flash_program(str(image), dry_run=True)
    assert out["success"] is False
    assert out["error_code"] == "SAFE_FLASH_MODE_REQUIRED"


def test_flash_program_dry_run_and_execute(tmp_path: Path) -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    image = tmp_path / "firmware.bin"
    image.write_bytes(b"\x10\x20")
    enable = service.set_flash_mode(True)
    assert enable["success"] is True
    dry = service.flash_program(str(image), address="0x08000000", dry_run=True)
    assert dry["success"] is True
    assert dry["data"]["dry_run"] is True
    run = service.flash_program(str(image), address="0x08000000", dry_run=False)
    assert run["success"] is True
    assert run["data"]["programmed"] is True
    assert any(cmd.startswith("program:") for cmd in fake.commands)


def test_flash_program_verify_mismatch_returns_structured_error(tmp_path: Path) -> None:
    class MismatchOpenOCDClient(FakeOpenOCDClient):
        def program(
            self,
            file_path: str,
            *,
            address: str | int | None = None,
            verify: bool = True,
            reset: bool = True,
            timeout_s: float = 60.0,
        ) -> str:
            _ = (file_path, address, verify, reset, timeout_s)
            return "verify failed at 0x08000010 expected 0x12345678 got 0x9abcdef0"

    fake = MismatchOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    image = tmp_path / "firmware.bin"
    image.write_bytes(b"\x10\x20")
    service.set_flash_mode(True)
    out = service.flash_program(str(image), address="0x08000000", dry_run=False, verify=True)
    assert out["success"] is False
    assert out["error_code"] == "DBG_FLASH_VERIFY_MISMATCH"
    mismatch = out["data"]["mismatch"]
    assert mismatch["detected"] is True
    assert mismatch["address"] == "0x08000010"
    assert mismatch["expected"] == "0x12345678"
    assert mismatch["actual"] == "0x9abcdef0"


def test_manage_breakpoint_add_del() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    added = service.manage_breakpoint("bp", "0x08000000", "add")
    assert added["success"] is True
    bp_id = added["data"]["id"]
    assert isinstance(bp_id, int)
    removed = service.manage_breakpoint("bp", "0x08000000", "del")
    assert removed["success"] is True
    assert removed["data"]["deleted"] is True


def test_manage_breakpoint_resource_exhausted_error() -> None:
    class ExhaustedOpenOCDClient(FakeOpenOCDClient):
        def manage_breakpoint(
            self,
            *,
            point_type: str,
            address: int,
            action: str,
            length: int = 4,
            access: str = "w",
        ) -> str:
            _ = (point_type, address, action, length, access)
            return "Error: cannot add hardware breakpoint - no free breakpoint comparator"

    fake = ExhaustedOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.manage_breakpoint("bp", "0x08000000", "add")
    assert out["success"] is False
    assert out["error_code"] == "DBG_BREAKPOINT_RESOURCE_EXHAUSTED"


def test_submit_task_uses_id_interface() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    submit = service.submit_task("ctx.record({'ok': 1})")
    assert submit["success"] is True
    task_id = submit["data"]["id"]
    sleep(0.05)
    result = service.get_task_result(task_id)
    assert result["success"] is True
    assert result["data"]["result"]["records"] == [{"ok": 1}]


def test_task_interface_rejects_name_argument() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    try:
        service.submit_task("ctx.record({'ok': 1})", name="legacy")  # type: ignore[call-arg]
        raised = False
    except TypeError:
        raised = True
    assert raised is True


def test_openocd_log_event_emits_watchpoint() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    task = service.submit_task(
        "def cb(ctx, payload):\n"
        "    ctx.record({'wp': payload.get('address')})\n"
        "on('WATCHPOINT_HIT', cb)\n"
    )
    task_id = task["data"]["id"]
    service.tasks.wait(task_id, timeout_s=1.0)
    assert fake.log_listener is not None
    fake.log_listener("watchpoint hit at 0x20000000", {"event": "WATCHPOINT_HIT", "address": "0x20000000"})
    result = service.get_task_result(task_id)
    assert result["data"]["result"]["records"] == [{"wp": "0x20000000"}]


def test_read_memory_accepts_svd_register_name(tmp_path: Path) -> None:
    svd_path = tmp_path / "test.svd"
    svd_path.write_text(
        """
<device>
  <peripherals>
    <peripheral>
      <name>TIM1</name>
      <baseAddress>0x40010000</baseAddress>
      <registers>
        <register>
          <name>CR1</name>
          <addressOffset>0x00</addressOffset>
        </register>
      </registers>
    </peripheral>
  </peripherals>
</device>
""".strip(),
        encoding="utf-8",
    )
    fake = FakeOpenOCDClient()
    svd = SVDResolver()
    svd.load(str(svd_path))
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager(), svd=svd)
    out = service.read_memory("TIM1->CR1", 1, 32)
    assert out["success"] is True
    assert out["data"]["base"] == "0x40010000"


def test_connect_debugger_existing_session_starts_log_monitor() -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    out = service.connect_debugger({"auto_start": False, "host": "127.0.0.1", "tcl_port": 6666, "telnet_port": 4444})
    assert out["success"] is True
    assert fake.log_monitor_started is True


def test_read_peripheral_fallback_returns_partial_data() -> None:
    class FallbackResolver:
        def resolve(self, name: str):
            _ = name
            raise MCPServiceError("DBG_REGISTER_NOT_FOUND", "parse failed")

        def resolve_best_effort(self, name: str):
            _ = name
            return {"address": 0x40010000, "raw_xml_snippet": "<register><name>CR1</name></register>", "error": "x"}

        def raw_xml_snippet(self, name: str):
            _ = name
            return "<register><name>CR1</name></register>"

    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager(), svd=FallbackResolver())  # type: ignore[arg-type]
    out = service.read_peripheral("TIM1->CR1")
    assert out["status"] == "partial_success"
    assert out["data"]["raw_xml_snippet"] is not None


def test_service_connect_debugger_closes_openocd_if_svd_load_fails(tmp_path: Path) -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    missing = tmp_path / "missing.svd"

    out = service.connect_debugger(
        {"adapter": "stlink", "target": "stm32f4x", "auto_start": True, "svd_path": str(missing)}
    )

    assert out["success"] is False
    assert fake.close_calls == 1


def test_service_connect_debugger_closes_existing_connection_if_elf_load_fails(tmp_path: Path) -> None:
    fake = FakeOpenOCDClient()
    service = OpenOCDMCPService(openocd=fake, serial_mgr=SerialManager())
    missing = tmp_path / "missing.elf"

    out = service.connect_debugger(
        {
            "auto_start": False,
            "host": "127.0.0.1",
            "tcl_port": 6666,
            "telnet_port": 4444,
            "elf_path": str(missing),
        }
    )

    assert out["success"] is False
    assert fake.close_calls == 1
