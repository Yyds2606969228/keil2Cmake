# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

from ..keil.config import get_armgcc_path
from .codegen import generate_c_code, generate_manifest
from .converter import load_onnx_model
from .runtime import validate_model_consistency


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
    backend: str,
    quant: str,
    weights: str,
    emit: str,
) -> dict[str, object]:
    if backend != "c":
        if backend != "cmsis-nn":
            raise ValueError("Only backend 'c' or 'cmsis-nn' is supported in this version.")
    model = load_onnx_model(model_path)
    quant = quant.lower()
    if quant not in ("fp32", "int8", "int16"):
        raise ValueError("Unsupported quant type.")
    if quant != "fp32":
        has_q = any(t.dtype in ("uint8", "int8", "int16") for t in model.tensors.values())
        if not has_q:
            raise ValueError("Quantized mode requires Q/DQ with uint8/int8/int16 tensors.")
        if quant == "int8" and any(t.dtype == "int16" for t in model.tensors.values()):
            raise ValueError("Quant mode int8 does not allow int16 tensors.")
        if quant == "int16" and any(t.dtype in ("uint8", "int8") for t in model.tensors.values()):
            raise ValueError("Quant mode int16 does not allow uint8/int8 tensors.")
    model_name = Path(model_path).stem

    root = Path(output_root)
    project_dir = root / model_name
    project_dir.mkdir(parents=True, exist_ok=True)

    codegen_result = generate_c_code(model, str(project_dir), model_name, weights, quant, backend)
    manifest_path = generate_manifest(
        model,
        str(project_dir),
        backend,
        quant,
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
    if validation.status == "failed":
        detail = validation.reason or "consistency check failed"
        raise ValueError(f"Consistency check failed: {detail}")

    lib_path = ""
    if emit == "lib":
        armgcc_path = get_armgcc_path()
        gcc = _infer_tool(armgcc_path, "arm-none-eabi-gcc.exe" if os.name == "nt" else "arm-none-eabi-gcc")
        ar = _infer_tool(armgcc_path, "arm-none-eabi-ar.exe" if os.name == "nt" else "arm-none-eabi-ar")
        src = codegen_result["source"]
        obj = str(project_dir / f"{model_name}.o")
        lib_path = str(project_dir / f"lib{model_name}.a")
        cmd_compile = f"\"{gcc}\" -c \"{src}\" -o \"{obj}\" -std=c99 -O2"
        cmd_ar = f"\"{ar}\" rcs \"{lib_path}\" \"{obj}\""
        ret_compile = os.system(cmd_compile)
        if ret_compile != 0:
            raise RuntimeError("Failed to compile model source with arm-none-eabi-gcc.")
        ret_ar = os.system(cmd_ar)
        if ret_ar != 0:
            raise RuntimeError("Failed to archive model library with arm-none-eabi-ar.")

    return {
        "project_dir": str(project_dir),
        "model_name": model_name,
        "header": codegen_result["header"],
        "source": codegen_result["source"],
        "manifest": manifest_path,
        "library": lib_path,
        "backend": backend,
        "quant": quant,
        "weights": weights,
        "validation": validation,
    }
