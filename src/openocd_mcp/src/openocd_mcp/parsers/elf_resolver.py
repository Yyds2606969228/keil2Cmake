"""ELF symbol resolver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import MCPServiceError

try:
    from elftools.elf.elffile import ELFFile  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependency optional in tests
    ELFFile = None


class ELFResolver:
    def __init__(self) -> None:
        self._symbols: dict[str, int] = {}
        self.path: str | None = None

    def load(self, path: str) -> None:
        elf_file = Path(path)
        if not elf_file.exists():
            raise MCPServiceError("DBG_ELF_NOT_FOUND", f"ELF file not found: {path}")
        if ELFFile is None:
            raise MCPServiceError("DBG_ELF_DEPENDENCY_MISSING", "pyelftools is not installed.")
        symbols: dict[str, int] = {}
        with elf_file.open("rb") as f:
            parsed: Any = ELFFile(f)
            for section in parsed.iter_sections():
                if section.header.sh_type != "SHT_SYMTAB":
                    continue
                for sym in section.iter_symbols():
                    if not sym.name:
                        continue
                    if sym.entry.st_value == 0:
                        continue
                    symbols[sym.name] = int(sym.entry.st_value)
        self._symbols = symbols
        self.path = str(elf_file)

    def resolve(self, token: str | int) -> int:
        if isinstance(token, int):
            return token
        value = token.strip()
        if value.startswith("0x"):
            return int(value, 16)
        if value.isdigit():
            return int(value, 10)
        if value in self._symbols:
            return self._symbols[value]
        raise MCPServiceError("DBG_SYMBOL_NOT_FOUND", f"Unknown symbol or address: {token}")
