# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import onnx

_OPTIONAL_IMPORT_ERRORS = (ImportError, OSError)
_BASE_REFERENCE_EXEC_ERRORS = (OSError, RuntimeError, ValueError, TypeError, NotImplementedError)
_ORT_EXEC_ERRORS: tuple[type[BaseException], ...] = ()

try:
    import onnxruntime as ort
except _OPTIONAL_IMPORT_ERRORS:  # pragma: no cover - optional dependency
    ort = None
else:
    try:
        from onnxruntime.capi import onnxruntime_pybind11_state as ort_state
    except _OPTIONAL_IMPORT_ERRORS:
        ort_state = None
    if ort_state is not None:
        _ORT_EXEC_ERRORS = tuple(
            cls for cls in (
                getattr(ort_state, "EPFail", None),
                getattr(ort_state, "EngineError", None),
                getattr(ort_state, "Fail", None),
                getattr(ort_state, "InvalidArgument", None),
                getattr(ort_state, "InvalidGraph", None),
                getattr(ort_state, "InvalidProtobuf", None),
                getattr(ort_state, "ModelLoaded", None),
                getattr(ort_state, "NoModel", None),
                getattr(ort_state, "NoSuchFile", None),
                getattr(ort_state, "NotImplemented", None),
                getattr(ort_state, "RuntimeException", None),
            )
            if isinstance(cls, type) and issubclass(cls, Exception)
        )

try:
    from onnx.reference import ReferenceEvaluator
except _OPTIONAL_IMPORT_ERRORS:  # pragma: no cover - optional fallback
    ReferenceEvaluator = None

_REFERENCE_EXEC_ERRORS = _BASE_REFERENCE_EXEC_ERRORS + _ORT_EXEC_ERRORS


def has_reference_backend() -> bool:
    return ort is not None or ReferenceEvaluator is not None


def run_reference_output(
    model_path: str,
    input_data: dict[str, np.ndarray],
    output_names: list[str],
    *,
    allow_reference_fallback: bool = True,
) -> tuple[dict[str, np.ndarray] | None, str, str]:
    def _map_outputs(
        outputs: list[np.ndarray] | tuple[np.ndarray, ...],
        names: list[str],
    ) -> tuple[dict[str, np.ndarray] | None, str]:
        if not outputs:
            return None, "reference output missing"
        if len(outputs) != len(names):
            return None, "reference output count mismatch"
        return {name: np.array(val) for name, val in zip(names, outputs)}, ""

    def _run_ort_with_ir_compat() -> tuple[dict[str, np.ndarray] | None, str]:
        # Some environments ship older onnxruntime builds that reject newer IR.
        try:
            compat_model = onnx.load(model_path)
            ir_version = int(getattr(compat_model, "ir_version", 0) or 0)
            if ir_version <= 9:
                return None, "ir-compat retry skipped"
            compat_model.ir_version = 9
            sess = ort.InferenceSession(  # type: ignore[union-attr]
                compat_model.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
            outputs = sess.run(output_names or None, input_data)
            names = output_names or [o.name for o in sess.get_outputs()]
            mapped, reason = _map_outputs(outputs, names)
            if mapped is None:
                return None, reason
            return mapped, ""
        except _REFERENCE_EXEC_ERRORS as compat_exc:
            return None, f"ir-compat retry failed: {compat_exc}"

    ort_reason = ""
    if ort is not None:
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            outputs = sess.run(output_names or None, input_data)
            names = output_names or [o.name for o in sess.get_outputs()]
            mapped, reason = _map_outputs(outputs, names)
            if mapped is None:
                return None, "onnxruntime", reason
            return mapped, "onnxruntime", ""
        except _REFERENCE_EXEC_ERRORS as exc:
            ort_reason = f"onnxruntime error: {exc}"
            msg = str(exc).lower()
            if "unsupported model ir version" in msg or "max supported ir version" in msg:
                compat_outputs, compat_reason = _run_ort_with_ir_compat()
                if compat_outputs is not None:
                    return compat_outputs, "onnxruntime(ir-compat)", ""
                if compat_reason:
                    ort_reason = f"{ort_reason}; {compat_reason}"
    else:
        ort_reason = "onnxruntime unavailable"

    if not allow_reference_fallback:
        return None, "onnxruntime", ort_reason or "onnxruntime output missing"

    if ReferenceEvaluator is not None:
        try:
            onnx_model = onnx.load(model_path)
            ref_eval = ReferenceEvaluator(onnx_model)
            outputs = ref_eval.run(output_names or None, input_data)
            if outputs:
                names = output_names
                if not names:
                    names = [v.name for v in onnx_model.graph.output]
                if len(outputs) != len(names):
                    return None, "onnx.reference", "reference output count mismatch"
                mapped = {name: np.array(val) for name, val in zip(names, outputs)}
                return mapped, "onnx.reference", ""
            return None, "onnx.reference", "reference output missing"
        except _REFERENCE_EXEC_ERRORS as exc:
            if ort_reason:
                return None, "onnx.reference", f"{ort_reason}; reference eval error: {exc}"
            return None, "onnx.reference", f"reference eval error: {exc}"
    return None, "", ort_reason
