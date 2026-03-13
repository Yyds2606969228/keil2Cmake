"""Resolve OpenOCD binary path from local configuration."""

from __future__ import annotations

from configparser import ConfigParser
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "keil2cmake" / "path.cfg"
ENV_CONFIG_PATH = "OPENOCD_MCP_KEIL2CMAKE_CFG"
ENV_OPENOCD_PATH = "OPENOCD_PATH"


def resolve_openocd_binary() -> str | None:
    """Resolve OpenOCD binary path from environment or keil2cmake path.cfg."""
    env_path = os.getenv(ENV_OPENOCD_PATH)
    if env_path:
        return env_path

    cfg_path = os.getenv(ENV_CONFIG_PATH, str(DEFAULT_CONFIG_PATH))
    file_path = Path(cfg_path)
    if not file_path.exists():
        return None

    parser = ConfigParser()
    parser.read(file_path, encoding="utf-8")
    candidate = parser.get("PATHS", "openocd_path", fallback="").strip()
    if not candidate:
        return None
    return candidate

