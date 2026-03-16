# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .codegen import generate_c_code, generate_manifest
from .converter import load_onnx_model
from .runtime import validate_model_consistency


def _run_command(cmd: list[str], error_msg: str) -> None:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"{error_msg}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if detail:
            detail = detail.splitlines()[0]
            raise RuntimeError(f"{error_msg}: {detail}")
        raise RuntimeError(error_msg)


def _infer_tool(toolchain_path: str, name: str) -> str:
    if not toolchain_path:
        return name
    p = Path(toolchain_path)
    if p.is_file():
        return str(p.parent / name)
    return str(Path(toolchain_path) / name)


def generate_tinyml_project(
    model_path: str,
    output_root: str,
    weights: str,
    emit: str,
    toolchain_bin: str = "",
    strict_validation: bool = True,
) -> dict[str, object]:
    backend = "c"
    model = load_onnx_model(model_path)
    model_name = Path(model_path).stem

    root = Path(output_root)
    project_dir = root / model_name
    project_dir.mkdir(parents=True, exist_ok=True)

    codegen_result = generate_c_code(model, str(project_dir), model_name, weights)
    manifest_path = generate_manifest(
        model,
        str(project_dir),
        backend,
        weights,
        int(codegen_result["arena_bytes"]),
        codegen_result.get("op_backends", []),
        codegen_result.get("backend_stats", {}),
        codegen_result.get("fallback_stats", {}),
    )
    validation = validate_model_consistency(
        model,
        model_path,
        source_path=str(codegen_result["source"]),
        header_path=str(codegen_result["header"]),
    )
    strict_mode = bool(strict_validation)
    if validation.status == "failed":
        detail = validation.reason or "consistency check failed"
        raise ValueError(f"Consistency check failed: {detail}")
    if strict_mode and validation.status == "skipped":
        detail = validation.reason or "consistency check skipped"
        raise ValueError(f"Consistency check skipped in strict mode: {detail}")

    lib_path = ""
    if emit == "lib":
        gcc = _infer_tool(toolchain_bin, "arm-none-eabi-gcc.exe" if os.name == "nt" else "arm-none-eabi-gcc")
        ar = _infer_tool(toolchain_bin, "arm-none-eabi-ar.exe" if os.name == "nt" else "arm-none-eabi-ar")
        src = codegen_result["source"]
        obj = str(project_dir / f"{model_name}.o")
        lib_path = str(project_dir / f"lib{model_name}.a")
        cmd_compile = [gcc, "-c", str(src), "-o", obj, "-std=c99", "-O2"]
        cmd_ar = [ar, "rcs", lib_path, obj]
        _run_command(cmd_compile, "Failed to compile model source with arm-none-eabi-gcc")
        _run_command(cmd_ar, "Failed to archive model library with arm-none-eabi-ar")

    return {
        "project_dir": str(project_dir),
        "model_name": model_name,
        "header": codegen_result["header"],
        "source": codegen_result["source"],
        "manifest": manifest_path,
        "library": lib_path,
        "backend": backend,
        "weights": weights,
        "validation": validation,
        "strict_validation": strict_mode,
    }
