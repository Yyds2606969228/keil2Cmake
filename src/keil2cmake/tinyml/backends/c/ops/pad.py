# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_pad, get_const_ints, get_const_scalar


@register_op("Pad")
def emit_pad(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Pad expects at least 1 input.")
    mode = str(node.attrs.get("mode", "constant"))
    if mode != "constant":
        raise ValueError("Pad supports constant mode only.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    in_name = node.inputs[0]
    inp = ctx.map_ptr(in_name)
    in_shape = ctx.shape(in_name)
    out_shape = ctx.shape(out_tensor)
    pads = node.attrs.get("pads")
    if pads is None and len(node.inputs) >= 2:
        pads = get_const_ints(ctx.model, node.inputs[1])
    if pads is None:
        raise ValueError("Pad pads not provided.")
    value = node.attrs.get("value", None)
    if value is None and len(node.inputs) >= 3:
        value = get_const_scalar(ctx.model, node.inputs[2])
    value = float(value) if value is not None else 0.0
    fill_value: float | str = value
    out_dtype = ctx.dtype(out_tensor)
    if out_dtype in ("int8", "int16"):
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        q = int(round(value / so) + zo)
        if q < qmin:
            q = qmin
        if q > qmax:
            q = qmax
        fill_value = str(q)
    emit_op_pad(ctx.lines, out, inp, in_shape, out_shape, [int(v) for v in pads], fill_value)

