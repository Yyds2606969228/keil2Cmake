"""SVD register resolver with XML fallback parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from ..core.errors import MCPServiceError

try:
    from cmsis_svd.parser import SVDParser  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependency optional in tests
    SVDParser = None


@dataclass(slots=True)
class RegisterInfo:
    peripheral: str
    register: str
    address: int
    fields: dict[str, dict[str, int | str]]


class SVDResolver:
    def __init__(self) -> None:
        self.path: str | None = None
        self._registers: dict[str, RegisterInfo] = {}

    def load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise MCPServiceError("DBG_SVD_NOT_FOUND", f"SVD file not found: {path}")
        self.path = str(file_path)
        self._registers = {}
        loaded = False
        if SVDParser is not None:
            try:
                self._load_with_cmsis(file_path)
                loaded = True
            except Exception:
                loaded = False
        if not loaded:
            self._load_with_xml(file_path)

    def _load_with_cmsis(self, path: Path) -> None:
        parser = SVDParser.for_xml_file(str(path))
        device = parser.get_device()
        output: dict[str, RegisterInfo] = {}
        for peripheral in device.peripherals:
            base = int(peripheral.base_address)
            for register in peripheral.registers or []:
                address = base + int(register.address_offset)
                fields: dict[str, dict[str, int | str]] = {}
                for field in register.fields or []:
                    fields[field.name] = {
                        "bit_offset": int(field.bit_offset),
                        "bit_width": int(field.bit_width),
                        "description": field.description or "",
                    }
                key = self._make_key(peripheral.name, register.name)
                output[key] = RegisterInfo(
                    peripheral=peripheral.name,
                    register=register.name,
                    address=address,
                    fields=fields,
                )
        self._registers = output

    def _load_with_xml(self, path: Path) -> None:
        tree = ET.parse(path)
        root = tree.getroot()
        output: dict[str, RegisterInfo] = {}
        for p in root.findall(".//peripheral"):
            periph_name = (p.findtext("name") or "").strip()
            base_text = (p.findtext("baseAddress") or "").strip()
            if not periph_name or not base_text:
                continue
            base = int(base_text, 0)
            for r in p.findall(".//register"):
                reg_name = (r.findtext("name") or "").strip()
                offset_text = (r.findtext("addressOffset") or "").strip()
                if not reg_name or not offset_text:
                    continue
                address = base + int(offset_text, 0)
                fields: dict[str, dict[str, int | str]] = {}
                for f in r.findall(".//field"):
                    field_name = (f.findtext("name") or "").strip()
                    bit_offset = int((f.findtext("bitOffset") or "0").strip(), 0)
                    bit_width = int((f.findtext("bitWidth") or "1").strip(), 0)
                    fields[field_name] = {
                        "bit_offset": bit_offset,
                        "bit_width": bit_width,
                        "description": (f.findtext("description") or "").strip(),
                    }
                key = self._make_key(periph_name, reg_name)
                output[key] = RegisterInfo(
                    peripheral=periph_name,
                    register=reg_name,
                    address=address,
                    fields=fields,
                )
        self._registers = output

    @staticmethod
    def _make_key(peripheral: str, register: str) -> str:
        return f"{peripheral.upper()}->{register.upper()}"

    @staticmethod
    def _normalize_name(name: str) -> str:
        token = name.replace(".", "->").replace("/", "->")
        if "->" not in token:
            raise MCPServiceError("DBG_PERIPHERAL_FORMAT_ERROR", "Peripheral name must be PERIPH->REG.")
        return token

    def resolve(self, name: str) -> RegisterInfo:
        key = self._normalize_name(name).upper()
        info = self._registers.get(key)
        if info is None:
            raise MCPServiceError("DBG_REGISTER_NOT_FOUND", f"Register not found in SVD: {name}")
        return info

    def decode_fields(self, info: RegisterInfo, value: int) -> dict[str, dict[str, int | str]]:
        output: dict[str, dict[str, int | str]] = {}
        for field_name, spec in info.fields.items():
            bit_offset = int(spec["bit_offset"])
            bit_width = int(spec["bit_width"])
            mask = (1 << bit_width) - 1
            output[field_name] = {
                "val": (value >> bit_offset) & mask,
                "description": str(spec.get("description", "")),
            }
        return output

    def raw_xml_snippet(self, name: str, max_chars: int = 500) -> str | None:
        if not self.path:
            return None
        token = self._normalize_name(name)
        peripheral, register = token.split("->", 1)
        tree = ET.parse(self.path)
        root = tree.getroot()
        for p in root.findall(".//peripheral"):
            if (p.findtext("name") or "").strip().upper() != peripheral.upper():
                continue
            for r in p.findall(".//register"):
                if (r.findtext("name") or "").strip().upper() != register.upper():
                    continue
                text = ET.tostring(r, encoding="unicode")
                return text[:max_chars]
        return None

    def resolve_best_effort(self, name: str, max_chars: int = 500) -> dict[str, str | int | None]:
        try:
            token = self._normalize_name(name)
        except MCPServiceError as exc:
            return {"address": None, "raw_xml_snippet": None, "error": exc.message}
        if not self.path:
            return {"address": None, "raw_xml_snippet": None, "error": "SVD is not loaded."}
        peripheral, register = token.split("->", 1)
        tree = ET.parse(self.path)
        root = tree.getroot()
        for p in root.findall(".//peripheral"):
            if (p.findtext("name") or "").strip().upper() != peripheral.upper():
                continue
            base_text = (p.findtext("baseAddress") or "").strip()
            base = int(base_text, 0) if base_text else None
            for r in p.findall(".//register"):
                if (r.findtext("name") or "").strip().upper() != register.upper():
                    continue
                offset_text = (r.findtext("addressOffset") or "").strip()
                offset = int(offset_text, 0) if offset_text else None
                address = (base + offset) if (base is not None and offset is not None) else None
                return {
                    "address": address,
                    "raw_xml_snippet": ET.tostring(r, encoding="unicode")[:max_chars],
                    "error": None,
                }
            return {
                "address": None,
                "raw_xml_snippet": ET.tostring(p, encoding="unicode")[:max_chars],
                "error": f"Register not found in peripheral: {name}",
            }
        return {"address": None, "raw_xml_snippet": None, "error": f"Peripheral not found: {name}"}
