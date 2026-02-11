# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from ..converter.ir import ModelIR


@dataclass
class CRunResult:
    ok: bool
    reason: str = ""
    output: np.ndarray | None = None


def _dtype_to_numpy(dtype: str):
    if dtype == "float32":
        return np.float32
    if dtype == "uint8":
        return np.uint8
    if dtype == "int8":
        return np.int8
    if dtype == "int16":
        return np.int16
    return None


def _dtype_to_c(dtype: str) -> str | None:
    if dtype == "float32":
        return "float"
    if dtype == "uint8":
        return "uint8_t"
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    return None


def _find_host_compiler() -> str | None:
    candidates: list[str] = []
    cc_env = os.environ.get("CC", "").strip()
    if cc_env:
        candidates.append(cc_env.split()[0])
    candidates.extend(["gcc", "clang", "cc", "gcc.exe", "clang.exe"])
    for cand in candidates:
        path = shutil.which(cand)
        if path:
            return path
    return None


def _runner_source(header_basename: str, input_ctype: str, output_ctype: str) -> str:
    return (
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"#include \"{header_basename}\"\n\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc != 3) {\n"
        "    return 2;\n"
        "  }\n"
        "  const char* in_path = argv[1];\n"
        "  const char* out_path = argv[2];\n"
        "  FILE* fi = fopen(in_path, \"rb\");\n"
        "  if (!fi) {\n"
        "    return 3;\n"
        "  }\n"
        f"  {input_ctype} input[K2C_INPUT_SIZE];\n"
        f"  {output_ctype} output[K2C_OUTPUT_SIZE];\n"
        "  size_t n_in = fread(input, sizeof(input[0]), K2C_INPUT_SIZE, fi);\n"
        "  fclose(fi);\n"
        "  if (n_in != K2C_INPUT_SIZE) {\n"
        "    return 4;\n"
        "  }\n"
        "  k2c_forward(input, output);\n"
        "  FILE* fo = fopen(out_path, \"wb\");\n"
        "  if (!fo) {\n"
        "    return 5;\n"
        "  }\n"
        "  size_t n_out = fwrite(output, sizeof(output[0]), K2C_OUTPUT_SIZE, fo);\n"
        "  fclose(fo);\n"
        "  if (n_out != K2C_OUTPUT_SIZE) {\n"
        "    return 6;\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    )


def run_generated_c_model(
    model: ModelIR,
    source_path: str,
    header_path: str,
    input_data: np.ndarray,
    *,
    timeout_sec: float = 30.0,
) -> CRunResult:
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        return CRunResult(ok=False, reason="only single input/output is supported")

    source = Path(source_path)
    header = Path(header_path)
    if not source.exists() or not header.exists():
        return CRunResult(ok=False, reason="generated source/header not found")

    input_tensor = model.inputs[0]
    output_tensor = model.outputs[0]
    input_np = _dtype_to_numpy(input_tensor.dtype)
    output_np = _dtype_to_numpy(output_tensor.dtype)
    input_ctype = _dtype_to_c(input_tensor.dtype)
    output_ctype = _dtype_to_c(output_tensor.dtype)
    if input_np is None or output_np is None or input_ctype is None or output_ctype is None:
        return CRunResult(ok=False, reason="unsupported validation dtype")

    compiler = _find_host_compiler()
    if not compiler:
        return CRunResult(ok=False, reason="host compiler not found")

    in_shape = list(input_tensor.shape)
    out_shape = list(output_tensor.shape)
    in_data = np.asarray(input_data, dtype=input_np).reshape(in_shape)
    out_count = int(np.prod(out_shape)) if out_shape else 1

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_bin = td_path / "k2c_in.bin"
        output_bin = td_path / "k2c_out.bin"
        runner_c = td_path / "k2c_runner.c"
        exe = td_path / ("k2c_runner.exe" if os.name == "nt" else "k2c_runner")

        runner_c.write_text(
            _runner_source(header.name, input_ctype, output_ctype),
            encoding="utf-8",
        )
        in_data.reshape(-1).tofile(str(input_bin))

        cmd = [
            compiler,
            "-O2",
            "-std=c99",
            "-I",
            str(header.parent),
            str(runner_c),
            str(source),
            "-o",
            str(exe),
        ]
        if os.name != "nt":
            cmd.append("-lm")

        try:
            comp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except Exception as exc:
            return CRunResult(ok=False, reason=f"compile error: {exc}")
        if comp.returncode != 0:
            detail = (comp.stderr or comp.stdout or "").strip().splitlines()
            short = detail[0] if detail else "unknown compiler error"
            return CRunResult(ok=False, reason=f"compile failed: {short}")

        try:
            run = subprocess.run(
                [str(exe), str(input_bin), str(output_bin)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except Exception as exc:
            return CRunResult(ok=False, reason=f"run error: {exc}")
        if run.returncode != 0:
            detail = (run.stderr or run.stdout or "").strip().splitlines()
            short = detail[0] if detail else f"exit code {run.returncode}"
            return CRunResult(ok=False, reason=f"runner failed: {short}")
        if not output_bin.exists():
            return CRunResult(ok=False, reason="runner output file missing")

        out = np.fromfile(str(output_bin), dtype=output_np)
        if out.size != out_count:
            return CRunResult(ok=False, reason="runner output size mismatch")
        out = out.reshape(out_shape if out_shape else ())
        return CRunResult(ok=True, output=out)
