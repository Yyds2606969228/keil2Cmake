# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..converter.ir import ModelIR
from .local_evaluator import _eval_model
from .reference_engine import ReferenceEvaluator, has_reference_backend, ort, run_reference_output
from .validation_pipeline import build_validation_inputs, compare_outputs, run_prediction_outputs


@dataclass
class ValidationResult:
    status: str
    reason: str = ""
    max_abs: float = 0.0
    max_rel: float = 0.0
    engine: str = ""

def _run_reference_output(
    model_path: str,
    input_data: dict[str, np.ndarray],
    output_names: list[str],
    *,
    allow_reference_fallback: bool = True,
) -> tuple[dict[str, np.ndarray] | None, str, str]:
    return run_reference_output(
        model_path,
        input_data,
        output_names,
        allow_reference_fallback=allow_reference_fallback,
    )


def validate_model_consistency(
    model: ModelIR,
    model_path: str,
    *,
    source_path: str = "",
    header_path: str = "",
    seed: int = 0,
    max_input_elems: int = 200000,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    int8_atol: float = 1.0,
    int16_atol: float = 1.0,
    allow_reference_fallback: bool = True,
) -> ValidationResult:
    if not has_reference_backend():
        return ValidationResult(status="skipped", reason="onnxruntime/reference evaluator unavailable")
    input_data, input_reason = build_validation_inputs(
        model,
        seed=seed,
        max_input_elems=max_input_elems,
    )
    if input_data is None:
        return ValidationResult(status="skipped", reason=input_reason)

    output_names = [t.name for t in model.outputs]
    ref_outputs, ref_engine, ref_reason = _run_reference_output(
        model_path,
        input_data,
        output_names,
        allow_reference_fallback=allow_reference_fallback,
    )
    if ref_outputs is None:
        return ValidationResult(status="skipped", reason=ref_reason)

    pred_outputs, pred_engine, pred_reason = run_prediction_outputs(
        model,
        input_data,
        output_names,
        source_path=source_path,
        header_path=header_path,
        eval_model=_eval_model,
    )
    if pred_outputs is None:
        return ValidationResult(status="skipped", reason=pred_reason)

    engine = f"{pred_engine} vs {ref_engine}"
    ok, compare_reason, max_abs, max_rel = compare_outputs(
        model,
        ref_outputs=ref_outputs,
        pred_outputs=pred_outputs,
        rtol=rtol,
        atol=atol,
        int8_atol=int8_atol,
        int16_atol=int16_atol,
    )
    if not ok:
        return ValidationResult(
            status="failed",
            reason=compare_reason,
            max_abs=max_abs,
            max_rel=max_rel,
            engine=engine,
        )
    return ValidationResult(status="passed", max_abs=max_abs, max_rel=max_rel, engine=engine)
