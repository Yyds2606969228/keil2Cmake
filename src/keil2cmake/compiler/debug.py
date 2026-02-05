# -*- coding: utf-8 -*-

import os

from ..common import ensure_dir
from ..template_engine import write_template


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




def generate_debug_templates(project_root: str) -> None:
    """Generate CMake-time debug templates (files generated on configure)."""
    internal_dir = os.path.join(project_root, "cmake", "internal")
    templates_dir = os.path.join(internal_dir, "templates")
    ensure_dir(templates_dir)

    write_template(
        "k2c_debug.cmake.j2",
        {},
        os.path.join(internal_dir, "k2c_debug.cmake"),
        encoding="utf-8",
    )
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
