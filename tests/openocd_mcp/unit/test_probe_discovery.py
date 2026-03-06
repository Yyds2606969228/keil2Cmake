from openocd_mcp.transport.probe_discovery import (
    _discover_from_windows_pnp,
    _extract_vid_pid,
    discover_debug_probes,
)


def test_extract_vid_pid() -> None:
    parsed = _extract_vid_pid(r"USB\\VID_0483&PID_3748\\123456")
    assert parsed == (0x0483, 0x3748)


def test_discover_from_windows_pnp_json() -> None:
    raw = (
        '[{"FriendlyName":"ST-Link Debug","InstanceId":"USB\\\\VID_0483&PID_3748\\\\A",'
        '"Class":"USB"},{"FriendlyName":"Other","InstanceId":"USB\\\\VID_1234&PID_ABCD\\\\B","Class":"USB"}]'
    )
    probes = _discover_from_windows_pnp(raw)
    assert len(probes) == 1
    assert probes[0]["family"] == "stlink"
    assert probes[0]["vid"] == "0483"
    assert probes[0]["pid"] == "3748".upper()


def test_discover_debug_probes_from_serial_hwid() -> None:
    serial_ports = [{"port": "COM7", "description": "STLink VCP", "hwid": "USB VID:PID=0483:374B"}]
    probes = discover_debug_probes(serial_ports)
    assert any(p["family"] == "stlink" for p in probes)
