# -*- coding: utf-8 -*-

import os
from pathlib import Path

from ..common import ensure_dir
from ..keil.config import get_armgcc_path, get_openocd_path
from ..template_engine import write_template, write_at_template


def infer_openocd_target(device: str) -> str:
    if not device:
        return ""
    dev = str(device).strip().lower()
    patterns = [
        ("stm32f0", "target/stm32f0x.cfg"),
        ("stm32f1", "target/stm32f1x.cfg"),
        ("stm32f2", "target/stm32f2x.cfg"),
        ("stm32f3", "target/stm32f3x.cfg"),
        ("stm32f4", "target/stm32f4x.cfg"),
        ("stm32f7", "target/stm32f7x.cfg"),
        ("stm32g0", "target/stm32g0x.cfg"),
        ("stm32g4", "target/stm32g4x.cfg"),
        ("stm32h7", "target/stm32h7x.cfg"),
        ("stm32l0", "target/stm32l0.cfg"),
        ("stm32l1", "target/stm32l1.cfg"),
        ("stm32l4", "target/stm32l4x.cfg"),
        ("stm32wb", "target/stm32wbx.cfg"),
        ("stm32wl", "target/stm32wlx.cfg"),
        ("nrf51", "target/nrf51.cfg"),
        ("nrf52", "target/nrf52.cfg"),
        ("lpc17", "target/lpc17xx.cfg"),
        ("lpc54", "target/lpc54xxx.cfg"),
    ]
    for prefix, target in patterns:
        if dev.startswith(prefix):
            return target
    return ""




def infer_openocd_interface(debugger: str) -> str:
    if not debugger:
        return ""
    dbg = str(debugger).strip().lower()
    mapping = {
        "stlink": "interface/stlink.cfg",
        "jlink": "interface/jlink.cfg",
        "daplink": "interface/cmsis-dap.cfg",
    }
    return mapping.get(dbg, "")


def infer_openocd_transport(debugger: str) -> str:
    if not debugger:
        return ""
    dbg = str(debugger).strip().lower()
    mapping = {
        # ST-Link uses HLA transport naming in OpenOCD scripts.
        "stlink": "hla_swd",
        # For Cortex-M, SWD is the common default for J-Link/CMSIS-DAP.
        "jlink": "swd",
        "daplink": "swd",
    }
    return mapping.get(dbg, "")


def _normalize_json_path(path: str) -> str:
    return str(path).replace("\\", "/") if path else ""


def _infer_armgcc_bin_dir(armgcc_path: str) -> str:
    if not armgcc_path:
        return ""
    p = Path(armgcc_path)
    if p.is_file():
        return str(p.parent)
    return str(p)


def _infer_gdb_path(armgcc_path: str) -> str:
    bin_dir = _infer_armgcc_bin_dir(armgcc_path)
    if not bin_dir:
        return "arm-none-eabi-gdb"
    exe = "arm-none-eabi-gdb.exe" if os.name == "nt" else "arm-none-eabi-gdb"
    return os.path.join(bin_dir, exe)


def generate_openocd_files(
    project_root: str,
    mcu: str = "",
    debugger: str = "",
    overwrite: bool = False,
) -> dict[str, str]:
    """Generate cortex-debug launch/tasks and OpenOCD config for an existing CMake project."""
    root = os.path.abspath(project_root)
    vscode_dir = os.path.join(root, ".vscode")
    user_dir = os.path.join(root, "cmake", "user")
    ensure_dir(vscode_dir)
    ensure_dir(user_dir)

    openocd_path = get_openocd_path() or "openocd"
    gdb_path = _infer_gdb_path(get_armgcc_path())
    debug_executable = "${workspaceFolder}/build/${workspaceFolderBasename}.elf"

    openocd_target = infer_openocd_target(mcu)
    openocd_interface = infer_openocd_interface(debugger)
    openocd_transport = infer_openocd_transport(debugger)

    context = {
        "K2C_OPENOCD_PATH": _normalize_json_path(openocd_path),
        "K2C_GDB_PATH": _normalize_json_path(gdb_path),
        "K2C_DEBUG_EXECUTABLE": _normalize_json_path(debug_executable),
        "K2C_OPENOCD_SEARCHDIR_JSON": "",
        "K2C_OPENOCD_SCRIPTS_ARGS": "",
        "K2C_OPENOCD_INTERFACE_LINE": (
            f"source [find {openocd_interface}]" if openocd_interface else ""
        ),
        "K2C_OPENOCD_TRANSPORT_LINE": (
            f"transport select {openocd_transport}" if openocd_transport else ""
        ),
        "K2C_OPENOCD_TARGET_LINE": (
            f"source [find {openocd_target}]" if openocd_target else ""
        ),
    }

    openocd_cfg = os.path.join(user_dir, "openocd.cfg")
    if overwrite or not os.path.exists(openocd_cfg):
        write_at_template("openocd.cfg.in.j2", context, openocd_cfg, encoding="utf-8")

    launch_json = os.path.join(vscode_dir, "launch.json")
    if overwrite or not os.path.exists(launch_json):
        write_at_template("launch.json.in.j2", context, launch_json, encoding="utf-8")

    tasks_json = os.path.join(vscode_dir, "tasks.json")
    if overwrite or not os.path.exists(tasks_json):
        write_at_template("tasks.json.in.j2", context, tasks_json, encoding="utf-8")

    return {
        "openocd_cfg": openocd_cfg,
        "launch_json": launch_json,
        "tasks_json": tasks_json,
        "openocd_path": openocd_path,
        "gdb_path": gdb_path,
        "debug_executable": debug_executable,
        "openocd_target": openocd_target,
        "openocd_interface": openocd_interface,
        "openocd_transport": openocd_transport,
    }


def generate_debug_templates(project_root: str) -> None:
    """Generate debug template files used by CMake configure-time sync."""
    internal_dir = os.path.join(project_root, "cmake", "internal")
    templates_dir = os.path.join(internal_dir, "templates")
    ensure_dir(templates_dir)

    write_template(
        "openocd.cfg.in.j2",
        {},
        os.path.join(templates_dir, "openocd.cfg.in"),
        encoding="utf-8",
    )
    write_template(
        "launch.json.in.j2",
        {},
        os.path.join(templates_dir, "launch.json.in"),
        encoding="utf-8",
    )
    write_template(
        "tasks.json.in.j2",
        {},
        os.path.join(templates_dir, "tasks.json.in"),
        encoding="utf-8",
    )
