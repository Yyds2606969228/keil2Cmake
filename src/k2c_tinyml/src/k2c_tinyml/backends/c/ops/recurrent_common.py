# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from ....operators.context import EmitContext


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    alpha: float
    beta: float


_SUPPORTED_ACTS = {
    "relu",
    "tanh",
    "sigmoid",
    "affine",
    "leakyrelu",
    "thresholdedrelu",
    "scaledtanh",
    "hardsigmoid",
    "elu",
    "softsign",
    "softplus",
}


def decode_attr_str(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").lower()
    return str(value).lower()


def direction_and_count(op_name: str, direction_attr: object) -> tuple[str, int]:
    direction = decode_attr_str(direction_attr, "forward")
    if direction == "bidirectional":
        return direction, 2
    if direction in ("forward", "reverse"):
        return direction, 1
    raise ValueError(f"{op_name} direction must be forward/reverse/bidirectional.")


def _activation_list(raw: object) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="ignore").lower())
        else:
            out.append(str(item).lower())
    return out


def _float_list(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    return [float(raw)]


def _default_alpha(act: str) -> float:
    if act == "leakyrelu":
        return 0.01
    if act == "thresholdedrelu":
        return 1.0
    if act == "scaledtanh":
        return 1.0
    if act == "hardsigmoid":
        return 0.2
    if act == "elu":
        return 1.0
    if act == "affine":
        return 1.0
    return 0.0


def _default_beta(act: str) -> float:
    if act == "scaledtanh":
        return 1.0
    if act == "hardsigmoid":
        return 0.5
    if act == "affine":
        return 0.0
    return 0.0


def build_activation_specs(
    op_name: str,
    *,
    raw_acts: object,
    raw_alpha: object,
    raw_beta: object,
    expected_count: int,
    default_cycle: list[str],
) -> list[ActivationSpec]:
    if expected_count <= 0:
        raise ValueError(f"{op_name} expected activation count must be > 0.")
    acts = _activation_list(raw_acts)
    if not acts:
        reps = expected_count // len(default_cycle)
        if reps * len(default_cycle) != expected_count:
            raise ValueError(f"{op_name} invalid default activation cycle.")
        acts = default_cycle * reps
    if len(acts) != expected_count:
        raise ValueError(f"{op_name} activations count mismatch.")
    for act in acts:
        if act not in _SUPPORTED_ACTS:
            raise ValueError(f"{op_name} activation '{act}' is unsupported.")
    alphas = _float_list(raw_alpha)
    betas = _float_list(raw_beta)
    specs: list[ActivationSpec] = []
    for idx, act in enumerate(acts):
        alpha = alphas[idx] if idx < len(alphas) else _default_alpha(act)
        beta = betas[idx] if idx < len(betas) else _default_beta(act)
        specs.append(ActivationSpec(name=act, alpha=float(alpha), beta=float(beta)))
    return specs


def activation_c_expr(pre_expr: str, spec: ActivationSpec) -> str:
    x = f"({pre_expr})"
    alpha = f"{spec.alpha:.9g}f"
    beta = f"{spec.beta:.9g}f"
    if spec.name == "relu":
        return f"({x} > 0.0f ? {x} : 0.0f)"
    if spec.name == "tanh":
        return f"tanhf({x})"
    if spec.name == "sigmoid":
        return f"(1.0f / (1.0f + expf(-{x})))"
    if spec.name == "affine":
        return f"({alpha} * {x} + {beta})"
    if spec.name == "leakyrelu":
        return f"({x} >= 0.0f ? {x} : ({alpha} * {x}))"
    if spec.name == "thresholdedrelu":
        return f"({x} > {alpha} ? {x} : 0.0f)"
    if spec.name == "scaledtanh":
        return f"({alpha} * tanhf({beta} * {x}))"
    if spec.name == "hardsigmoid":
        return f"fminf(fmaxf({alpha} * {x} + {beta}, 0.0f), 1.0f)"
    if spec.name == "elu":
        return f"({x} >= 0.0f ? {x} : ({alpha} * (expf({x}) - 1.0f)))"
    if spec.name == "softsign":
        return f"({x} / (1.0f + fabsf({x})))"
    if spec.name == "softplus":
        return f"logf(1.0f + expf({x}))"
    raise ValueError(f"Unsupported activation '{spec.name}'.")


def emit_activation_assign(
    ctx: EmitContext,
    *,
    indent: str,
    out_var: str,
    pre_var: str,
    specs_by_dir: list[ActivationSpec],
    dir_var: str,
) -> None:
    if len(specs_by_dir) == 1:
        ctx.lines.append(f"{indent}{out_var} = {activation_c_expr(pre_var, specs_by_dir[0])};")
        return
    if len(specs_by_dir) != 2:
        raise ValueError("Activation direction count must be 1 or 2.")
    ctx.lines.append(f"{indent}if ({dir_var} == 0) {{")
    ctx.lines.append(f"{indent}  {out_var} = {activation_c_expr(pre_var, specs_by_dir[0])};")
    ctx.lines.append(f"{indent}}} else {{")
    ctx.lines.append(f"{indent}  {out_var} = {activation_c_expr(pre_var, specs_by_dir[1])};")
    ctx.lines.append(f"{indent}}}")


def emit_clip(
    ctx: EmitContext,
    *,
    indent: str,
    value_var: str,
    clip_var: str | None,
) -> None:
    if clip_var is None:
        return
    ctx.lines.append(f"{indent}if ({value_var} > {clip_var}) {value_var} = {clip_var};")
    ctx.lines.append(f"{indent}if ({value_var} < -{clip_var}) {value_var} = -{clip_var};")


def assert_rec_dtype(ctx: EmitContext, op_name: str, tensor_name: str, role: str) -> None:
    dtype = ctx.dtype(tensor_name)
    if dtype not in ("float32", "int8", "int16"):
        raise ValueError(f"{op_name} {role} dtype must be float32/int8/int16.")
    if dtype in ("int8", "int16"):
        _ = ctx.qparams(tensor_name)


def read_real_expr(ctx: EmitContext, tensor_name: str, idx_expr: str) -> str:
    dtype = ctx.dtype(tensor_name)
    ptr = ctx.map_ptr(tensor_name)
    if dtype == "float32":
        return f"{ptr}[{idx_expr}]"
    if dtype in ("int8", "int16"):
        scale, zero = ctx.qparams(tensor_name)
        return f"(((float){ptr}[{idx_expr}] - {zero}) * {scale:.9g}f)"
    raise ValueError(f"Unsupported recurrent tensor dtype: {dtype}")


def emit_store_real(
    ctx: EmitContext,
    *,
    indent: str,
    tensor_name: str,
    idx_expr: str,
    value_expr: str,
) -> None:
    dtype = ctx.dtype(tensor_name)
    ptr = ctx.map_ptr(tensor_name)
    if dtype == "float32":
        ctx.lines.append(f"{indent}{ptr}[{idx_expr}] = {value_expr};")
        return
    if dtype == "int8":
        so, zo = ctx.qparams(tensor_name)
        qv = ctx.next_symbol("k2c_q")
        ctx.lines.append(f"{indent}int {qv} = (int)roundf(({value_expr}) / {so:.9g}f) + {zo};")
        ctx.lines.append(f"{indent}if ({qv} < -128) {qv} = -128;")
        ctx.lines.append(f"{indent}if ({qv} > 127) {qv} = 127;")
        ctx.lines.append(f"{indent}{ptr}[{idx_expr}] = (int8_t){qv};")
        return
    if dtype == "int16":
        so, zo = ctx.qparams(tensor_name)
        qv = ctx.next_symbol("k2c_q")
        ctx.lines.append(f"{indent}int {qv} = (int)roundf(({value_expr}) / {so:.9g}f) + {zo};")
        ctx.lines.append(f"{indent}if ({qv} < -32768) {qv} = -32768;")
        ctx.lines.append(f"{indent}if ({qv} > 32767) {qv} = 32767;")
        ctx.lines.append(f"{indent}{ptr}[{idx_expr}] = (int16_t){qv};")
        return
    raise ValueError(f"Unsupported recurrent output dtype: {dtype}")


def emit_fill_real_zero(
    ctx: EmitContext,
    *,
    indent: str,
    tensor_name: str,
    count_expr: str,
) -> None:
    ptr = ctx.map_ptr(tensor_name)
    dtype = ctx.dtype(tensor_name)
    if dtype == "float32":
        ctx.lines.append(f"{indent}for (size_t i = 0; i < {count_expr}; ++i) {ptr}[i] = 0.0f;")
        return
    if dtype == "int8":
        _, zo = ctx.qparams(tensor_name)
        qz = max(-128, min(127, int(zo)))
        ctx.lines.append(f"{indent}for (size_t i = 0; i < {count_expr}; ++i) {ptr}[i] = (int8_t){qz};")
        return
    if dtype == "int16":
        _, zo = ctx.qparams(tensor_name)
        qz = max(-32768, min(32767, int(zo)))
        ctx.lines.append(f"{indent}for (size_t i = 0; i < {count_expr}; ++i) {ptr}[i] = (int16_t){qz};")
        return
    raise ValueError("Unsupported recurrent fill dtype.")
