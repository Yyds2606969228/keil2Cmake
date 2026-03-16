"""Best-effort debug probe discovery."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

KNOWN_PROBES: dict[tuple[int, int], dict[str, str]] = {
    (0x0483, 0x3748): {"vendor": "STMicroelectronics", "model": "ST-Link/V2", "family": "stlink"},
    (0x0483, 0x374B): {"vendor": "STMicroelectronics", "model": "ST-Link/V2-1", "family": "stlink"},
    (0x0483, 0x374F): {"vendor": "STMicroelectronics", "model": "ST-Link/V3", "family": "stlink"},
    (0x1366, 0x0101): {"vendor": "SEGGER", "model": "J-Link", "family": "jlink"},
    (0x0D28, 0x0204): {"vendor": "Arm", "model": "CMSIS-DAP", "family": "cmsis-dap"},
}

VID_PID_RE = re.compile(r"VID[_: ]([0-9A-Fa-f]{4}).*PID[_: ]([0-9A-Fa-f]{4})")
VID_PID_EQ_RE = re.compile(r"VID:PID=([0-9A-Fa-f]{4}):([0-9A-Fa-f]{4})")


def _probe_from_vid_pid(vid: int, pid: int, *, source: str, instance_id: str = "") -> dict[str, Any] | None:
    meta = KNOWN_PROBES.get((vid, pid))
    if meta is None:
        return None
    return {
        "vid": f"{vid:04X}",
        "pid": f"{pid:04X}",
        "vendor": meta["vendor"],
        "model": meta["model"],
        "family": meta["family"],
        "instance_id": instance_id,
        "source": source,
    }


def _extract_vid_pid(text: str) -> tuple[int, int] | None:
    match = VID_PID_RE.search(text)
    if not match:
        match = VID_PID_EQ_RE.search(text)
    if not match:
        return None
    return int(match.group(1), 16), int(match.group(2), 16)


def _discover_from_serial_hwid(serial_ports: list[dict[str, str]]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for item in serial_ports:
        hwid = item.get("hwid", "")
        parsed = _extract_vid_pid(hwid)
        if parsed is None:
            continue
        probe = _probe_from_vid_pid(parsed[0], parsed[1], source="serial_hwid", instance_id=item.get("port", ""))
        if probe:
            probes.append(probe)
    return probes


def _run_windows_pnp_query() -> str:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-PnpDevice -PresentOnly | Select-Object FriendlyName,InstanceId,Class | ConvertTo-Json -Compress",
    ]
    completed = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        check=False,
    )
    return completed.stdout.strip()


def _discover_from_windows_pnp(raw_json: str) -> list[dict[str, Any]]:
    if not raw_json:
        return []
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    rows: list[dict[str, Any]]
    if isinstance(parsed, list):
        rows = [x for x in parsed if isinstance(x, dict)]
    elif isinstance(parsed, dict):
        rows = [parsed]
    else:
        return []

    probes: list[dict[str, Any]] = []
    for row in rows:
        instance_id = str(row.get("InstanceId", ""))
        vid_pid = _extract_vid_pid(instance_id)
        if vid_pid is None:
            continue
        probe = _probe_from_vid_pid(vid_pid[0], vid_pid[1], source="windows_pnp", instance_id=instance_id)
        if probe is None:
            continue
        friendly = row.get("FriendlyName")
        if isinstance(friendly, str) and friendly.strip():
            probe["friendly_name"] = friendly.strip()
        probes.append(probe)
    return probes


def _discover_from_pyusb() -> list[dict[str, Any]]:
    try:
        import usb.core  # type: ignore[import-untyped]
    except ImportError:
        return []
    probes: list[dict[str, Any]] = []
    try:
        devices = usb.core.find(find_all=True)  # type: ignore[attr-defined]
    except Exception:
        # pyusb is installed but backend driver (libusb/winusb) is unavailable.
        return []
    for dev in devices:
        probe = _probe_from_vid_pid(
            int(dev.idVendor),
            int(dev.idProduct),
            source="pyusb",
            instance_id=f"bus:{getattr(dev, 'bus', '')}/addr:{getattr(dev, 'address', '')}",
        )
        if probe:
            probes.append(probe)
    return probes


def discover_debug_probes(serial_ports: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    if serial_ports:
        probes.extend(_discover_from_serial_hwid(serial_ports))
    if sys.platform.startswith("win"):
        try:
            raw = _run_windows_pnp_query()
            probes.extend(_discover_from_windows_pnp(raw))
        except Exception:
            pass
    probes.extend(_discover_from_pyusb())

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for probe in probes:
        key = (
            str(probe.get("vid", "")),
            str(probe.get("pid", "")),
            str(probe.get("instance_id", "")),
        )
        dedup[key] = probe
    return list(dedup.values())
