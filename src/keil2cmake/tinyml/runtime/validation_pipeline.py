# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable

import numpy as np

from ..converter.ir import ModelIR
from .c_runner import run_generated_c_model


def build_validation_inputs(
    model: ModelIR,
    *,
    seed: int,
    max_input_elems: int,
) -> tuple[dict[str, np.ndarray] | None, str]:
    if not model.inputs or not model.outputs:
        return None, "model has no inputs or outputs"

    rng = np.random.default_rng(seed)
    input_data: dict[str, np.ndarray] = {}
    for input_tensor in model.inputs:
        input_shape = list(input_tensor.shape)
        if any(dim <= 0 for dim in input_shape):
            return None, f"input shape unknown: {input_tensor.name}"
        input_elems = int(np.prod(input_shape)) if input_shape else 1
        if input_elems > max_input_elems:
            return None, f"input too large for validation: {input_tensor.name}"

        if input_tensor.dtype == "float32":
            data = rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        elif input_tensor.dtype == "uint8":
            data = rng.integers(0, 256, size=input_shape, dtype=np.uint8)
        elif input_tensor.dtype == "int8":
            data = rng.integers(-128, 128, size=input_shape, dtype=np.int8)
        elif input_tensor.dtype == "int16":
            data = rng.integers(-32768, 32768, size=input_shape, dtype=np.int16)
        elif input_tensor.dtype == "int32":
            data = rng.integers(-32768, 32768, size=input_shape, dtype=np.int32)
        elif input_tensor.dtype == "int64":
            data = rng.integers(-32768, 32768, size=input_shape, dtype=np.int64)
        elif input_tensor.dtype == "bool":
            data = rng.integers(0, 2, size=input_shape, dtype=np.uint8).astype(np.bool_)
        else:
            return None, f"input dtype unsupported: {input_tensor.dtype}"

        if input_shape:
            input_data[input_tensor.name] = data.reshape(input_shape)
        else:
            input_data[input_tensor.name] = np.asarray(data).reshape(())
    return input_data, ""


def run_prediction_outputs(
    model: ModelIR,
    input_data: dict[str, np.ndarray],
    output_names: list[str],
    *,
    source_path: str,
    header_path: str,
    eval_model: Callable[[ModelIR, dict[str, np.ndarray]], dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray] | None, str, str]:
    if source_path and header_path:
        c_run = run_generated_c_model(model, source_path, header_path, input_data)
        if not c_run.ok or c_run.outputs is None:
            return None, "generated-c", f"generated-c run skipped: {c_run.reason}"
        return c_run.outputs, "generated-c", ""

    try:
        pred_tensors = eval_model(model, input_data)
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
        return None, "python-eval", f"predict eval error: {exc}"

    pred_outputs: dict[str, np.ndarray] = {}
    for out_name in output_names:
        if out_name not in pred_tensors:
            return None, "python-eval", f"predict output missing: {out_name}"
        pred_outputs[out_name] = pred_tensors[out_name]
    return pred_outputs, "python-eval", ""


def compare_outputs(
    model: ModelIR,
    *,
    ref_outputs: dict[str, np.ndarray],
    pred_outputs: dict[str, np.ndarray],
    rtol: float,
    atol: float,
    int8_atol: float,
    int16_atol: float,
) -> tuple[bool, str, float, float]:
    max_abs = 0.0
    max_rel = 0.0

    for out_tensor in model.outputs:
        out_name = out_tensor.name
        if out_name not in ref_outputs:
            return False, f"reference output missing: {out_name}", max_abs, max_rel
        if out_name not in pred_outputs:
            return False, f"predict output missing: {out_name}", max_abs, max_rel

        pred = np.asarray(pred_outputs[out_name])
        ref = np.asarray(ref_outputs[out_name])
        if pred.shape != ref.shape:
            return False, f"output shape mismatch: {out_name}", max_abs, max_rel

        if out_tensor.dtype in ("uint8", "int8", "int16", "int32", "int64", "bool"):
            if out_tensor.dtype == "bool":
                pred_i64 = pred.astype(np.uint8).astype(np.int64)
                ref_i64 = ref.astype(np.uint8).astype(np.int64)
            else:
                pred_i64 = pred.astype(np.int64)
                ref_i64 = ref.astype(np.int64)
            diff = np.abs(pred_i64 - ref_i64)
            max_diff = float(np.max(diff)) if diff.size else 0.0
            if out_tensor.dtype in ("uint8", "int8"):
                tol = float(int8_atol)
            elif out_tensor.dtype == "int16":
                tol = float(int16_atol)
            else:
                tol = 0.0
            max_abs = max(max_abs, max_diff)
            if max_diff > tol:
                return False, f"output mismatch: {out_name}", max_diff, max_rel
            continue

        pred_f = pred.astype(np.float32)
        ref_f = np.asarray(ref, dtype=np.float32)
        if not np.allclose(pred_f, ref_f, rtol=rtol, atol=atol):
            abs_err = np.max(np.abs(pred_f - ref_f))
            rel_err = np.max(np.abs(pred_f - ref_f) / (np.abs(ref_f) + 1e-8))
            return False, f"output mismatch: {out_name}", float(abs_err), float(rel_err)

        abs_err = float(np.max(np.abs(pred_f - ref_f))) if pred_f.size else 0.0
        rel_err = (
            float(np.max(np.abs(pred_f - ref_f) / (np.abs(ref_f) + 1e-8)))
            if pred_f.size
            else 0.0
        )
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)

    return True, "", max_abs, max_rel
