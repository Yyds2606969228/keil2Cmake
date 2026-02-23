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
    outputs: dict[str, np.ndarray] | None = None


def _dtype_to_numpy(dtype: str):
    if dtype == "float32":
        return np.float32
    if dtype == "uint8":
        return np.uint8
    if dtype == "int8":
        return np.int8
    if dtype == "int16":
        return np.int16
    if dtype == "int32":
        return np.int32
    if dtype == "int64":
        return np.int64
    if dtype == "bool":
        return np.bool_
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
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    if dtype == "bool":
        return "uint8_t"
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


def _runner_source(header_basename: str) -> str:
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
        "  size_t n_in = 0;\n"
        "  size_t n_out = 0;\n"
        "  const k2c_io_desc_t* in_desc = k2c_get_input_desc(&n_in);\n"
        "  const k2c_io_desc_t* out_desc = k2c_get_output_desc(&n_out);\n"
        "  if (!in_desc || !out_desc || n_in == 0 || n_out == 0) {\n"
        "    fclose(fi);\n"
        "    return 4;\n"
        "  }\n"
        "  const void** input_ptrs = (const void**)calloc(n_in, sizeof(void*));\n"
        "  void** output_ptrs = (void**)calloc(n_out, sizeof(void*));\n"
        "  void** input_bufs = (void**)calloc(n_in, sizeof(void*));\n"
        "  void** output_bufs = (void**)calloc(n_out, sizeof(void*));\n"
        "  if (!input_ptrs || !output_ptrs || !input_bufs || !output_bufs) {\n"
        "    fclose(fi);\n"
        "    free((void*)input_ptrs);\n"
        "    free(output_ptrs);\n"
        "    free(input_bufs);\n"
        "    free(output_bufs);\n"
        "    return 5;\n"
        "  }\n"
        "  for (size_t i = 0; i < n_in; ++i) {\n"
        "    size_t bytes = in_desc[i].elem_size * in_desc[i].size;\n"
        "    if (bytes == 0) {\n"
        "      bytes = 1;\n"
        "    }\n"
        "    input_bufs[i] = malloc(bytes);\n"
        "    if (!input_bufs[i]) {\n"
        "      fclose(fi);\n"
        "      for (size_t j = 0; j < i; ++j) free(input_bufs[j]);\n"
        "      free((void*)input_ptrs);\n"
        "      free(output_ptrs);\n"
        "      free(input_bufs);\n"
        "      free(output_bufs);\n"
        "      return 6;\n"
        "    }\n"
        "    size_t n_read = fread(input_bufs[i], 1, bytes, fi);\n"
        "    if (n_read != bytes) {\n"
        "      fclose(fi);\n"
        "      for (size_t j = 0; j <= i; ++j) free(input_bufs[j]);\n"
        "      free((void*)input_ptrs);\n"
        "      free(output_ptrs);\n"
        "      free(input_bufs);\n"
        "      free(output_bufs);\n"
        "      return 7;\n"
        "    }\n"
        "    input_ptrs[i] = input_bufs[i];\n"
        "  }\n"
        "  fclose(fi);\n"
        "  for (size_t i = 0; i < n_out; ++i) {\n"
        "    size_t bytes = out_desc[i].elem_size * out_desc[i].size;\n"
        "    if (bytes == 0) {\n"
        "      bytes = 1;\n"
        "    }\n"
        "    output_bufs[i] = malloc(bytes);\n"
        "    if (!output_bufs[i]) {\n"
        "      for (size_t j = 0; j < n_in; ++j) free(input_bufs[j]);\n"
        "      for (size_t j = 0; j < i; ++j) free(output_bufs[j]);\n"
        "      free((void*)input_ptrs);\n"
        "      free(output_ptrs);\n"
        "      free(input_bufs);\n"
        "      free(output_bufs);\n"
        "      return 8;\n"
        "    }\n"
        "    output_ptrs[i] = output_bufs[i];\n"
        "  }\n"
        "  k2c_forward(input_ptrs, output_ptrs);\n"
        "  FILE* fo = fopen(out_path, \"wb\");\n"
        "  if (!fo) {\n"
        "    for (size_t i = 0; i < n_in; ++i) free(input_bufs[i]);\n"
        "    for (size_t i = 0; i < n_out; ++i) free(output_bufs[i]);\n"
        "    free((void*)input_ptrs);\n"
        "    free(output_ptrs);\n"
        "    free(input_bufs);\n"
        "    free(output_bufs);\n"
        "    return 9;\n"
        "  }\n"
        "  for (size_t i = 0; i < n_out; ++i) {\n"
        "    size_t bytes = out_desc[i].elem_size * out_desc[i].size;\n"
        "    if (bytes == 0) {\n"
        "      bytes = 1;\n"
        "    }\n"
        "    size_t n_written = fwrite(output_bufs[i], 1, bytes, fo);\n"
        "    if (n_written != bytes) {\n"
        "      fclose(fo);\n"
        "      for (size_t j = 0; j < n_in; ++j) free(input_bufs[j]);\n"
        "      for (size_t j = 0; j < n_out; ++j) free(output_bufs[j]);\n"
        "      free((void*)input_ptrs);\n"
        "      free(output_ptrs);\n"
        "      free(input_bufs);\n"
        "      free(output_bufs);\n"
        "      return 10;\n"
        "    }\n"
        "  }\n"
        "  fclose(fo);\n"
        "  for (size_t i = 0; i < n_in; ++i) free(input_bufs[i]);\n"
        "  for (size_t i = 0; i < n_out; ++i) free(output_bufs[i]);\n"
        "  free((void*)input_ptrs);\n"
        "  free(output_ptrs);\n"
        "  free(input_bufs);\n"
        "  free(output_bufs);\n"
        "  return 0;\n"
        "}\n"
    )


def run_generated_c_model(
    model: ModelIR,
    source_path: str,
    header_path: str,
    input_data: np.ndarray | dict[str, np.ndarray],
    *,
    timeout_sec: float = 30.0,
) -> CRunResult:
    source = Path(source_path)
    header = Path(header_path)
    if not source.exists() or not header.exists():
        return CRunResult(ok=False, reason="generated source/header not found")

    if isinstance(input_data, dict):
        feed_map = dict(input_data)
    else:
        if len(model.inputs) != 1:
            return CRunResult(ok=False, reason="multi-input model requires dict input")
        feed_map = {model.inputs[0].name: input_data}

    for tensor in model.inputs + model.outputs:
        if _dtype_to_numpy(tensor.dtype) is None or _dtype_to_c(tensor.dtype) is None:
            return CRunResult(ok=False, reason=f"unsupported validation dtype: {tensor.dtype}")

    compiler = _find_host_compiler()
    if not compiler:
        return CRunResult(ok=False, reason="host compiler not found")

    prepared_inputs: dict[str, np.ndarray] = {}
    for tensor in model.inputs:
        if tensor.name not in feed_map:
            return CRunResult(ok=False, reason=f"missing input data: {tensor.name}")
        np_dtype = _dtype_to_numpy(tensor.dtype)
        assert np_dtype is not None
        shape = list(tensor.shape)
        try:
            arr = np.asarray(feed_map[tensor.name], dtype=np_dtype)
            arr = arr.reshape(shape if shape else ())
        except Exception as exc:
            return CRunResult(ok=False, reason=f"input reshape failed for {tensor.name}: {exc}")
        prepared_inputs[tensor.name] = arr

    output_specs: list[tuple[str, object, list[int], int]] = []
    for tensor in model.outputs:
        np_dtype = _dtype_to_numpy(tensor.dtype)
        assert np_dtype is not None
        shape = list(tensor.shape)
        count = int(np.prod(shape)) if shape else 1
        output_specs.append((tensor.name, np_dtype, shape, count))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        input_bin = td_path / "k2c_in.bin"
        output_bin = td_path / "k2c_out.bin"
        runner_c = td_path / "k2c_runner.c"
        exe = td_path / ("k2c_runner.exe" if os.name == "nt" else "k2c_runner")

        runner_c.write_text(
            _runner_source(header.name),
            encoding="utf-8",
        )
        with input_bin.open("wb") as fi:
            for tensor in model.inputs:
                prepared_inputs[tensor.name].reshape(-1).tofile(fi)

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

        outputs: dict[str, np.ndarray] = {}
        with output_bin.open("rb") as fo:
            for name, np_dtype, shape, count in output_specs:
                out = np.fromfile(fo, dtype=np_dtype, count=count)
                if out.size != count:
                    return CRunResult(ok=False, reason=f"runner output size mismatch: {name}")
                outputs[name] = out.reshape(shape if shape else ())
            extra = fo.read(1)
            if extra:
                return CRunResult(ok=False, reason="runner output has trailing bytes")

        first_output = outputs[model.outputs[0].name] if model.outputs else None
        return CRunResult(ok=True, output=first_output, outputs=outputs)
