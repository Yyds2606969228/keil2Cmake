# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ...c.ops.global_average_pool import emit_global_avg_pool as emit_global_avg_pool_c
from .registry import register_op


@register_op("GlobalAveragePool")
def emit_global_avg_pool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("GlobalAveragePool expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    x_name = node.inputs[0]
    if out_dtype not in ("int8", "int16") or ctx.dtype(x_name) != out_dtype:
        ctx.backend_used = "c"
        emit_global_avg_pool_c(ctx, node)
        return

    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_tensor)
    if len(x_shape) != 4 or len(out_shape) != 4:
        ctx.backend_used = "c"
        emit_global_avg_pool_c(ctx, node)
        return
    n, c, h, w_in = x_shape
    n_out, c_out, out_h, out_w = out_shape
    if n != 1 or n_out != 1 or c != c_out or out_h != 1 or out_w != 1:
        ctx.backend_used = "c"
        emit_global_avg_pool_c(ctx, node)
        return

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_tensor)
    in_nhwc = ctx.next_symbol("k2c_pool_in")
    out_nhwc = ctx.next_symbol("k2c_pool_out")
    status_var = ctx.next_symbol("k2c_cmsis_status")
    done_label = ctx.next_symbol("k2c_cmsis_done")
    if out_dtype == "int8":
        ctype = "int8_t"
        func = "arm_avgpool_s8"
        act_min = "-128"
        act_max = "127"
    else:
        ctype = "int16_t"
        func = "arm_avgpool_s16"
        act_min = "-32768"
        act_max = "32767"

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"  static {ctype} {in_nhwc}[{c} * {h} * {w_in}];")
    ctx.lines.append(f"  static {ctype} {out_nhwc}[{c_out} * {out_h} * {out_w}];")
    ctx.lines.append(f"  for (size_t ih = 0; ih < {h}; ++ih) {{")
    ctx.lines.append(f"    for (size_t iw = 0; iw < {w_in}; ++iw) {{")
    ctx.lines.append(f"      for (size_t ic = 0; ic < {c}; ++ic) {{")
    ctx.lines.append(
        f"        {in_nhwc}[(ih * {w_in} + iw) * {c} + ic] = "
        f"{x}[ic * {h} * {w_in} + ih * {w_in} + iw];"
    )
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
    ctx.lines.append("  cmsis_nn_context nn_ctx = { .buf = NULL, .size = 0 };")
    ctx.lines.append("  cmsis_nn_pool_params pool_params = {")
    ctx.lines.append("    .stride = { .h = 1, .w = 1 },")
    ctx.lines.append("    .padding = { .h = 0, .w = 0 },")
    ctx.lines.append(f"    .activation = {{ .min = {act_min}, .max = {act_max} }},")
    ctx.lines.append("    .count_include_pad = 0,")
    ctx.lines.append("  };")
    ctx.lines.append(f"  cmsis_nn_dims input_dims = {{ .n = 1, .h = {h}, .w = {w_in}, .c = {c} }};")
    ctx.lines.append(f"  cmsis_nn_dims filter_dims = {{ .n = 1, .h = {h}, .w = {w_in}, .c = 1 }};")
    ctx.lines.append(
        f"  cmsis_nn_dims output_dims = {{ .n = 1, .h = {out_h}, .w = {out_w}, .c = {c_out} }};"
    )
    ctx.lines.append(
        f"  arm_cmsis_nn_status {status_var} = {func}("
        f"&nn_ctx, &pool_params, &input_dims, {in_nhwc}, &filter_dims, &output_dims, {out_nhwc});"
    )
    ctx.lines.append(f"  if ({status_var} == ARM_CMSIS_NN_SUCCESS) {{")
    ctx.lines.append(f"    for (size_t oc = 0; oc < {c_out}; ++oc) {{")
    ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    ctx.lines.append(
        f"          {out}[oc * {out_h} * {out_w} + oh * {out_w} + ow] = "
        f"{out_nhwc}[(oh * {out_w} + ow) * {c_out} + oc];"
    )
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append(f"    goto {done_label};")
    ctx.lines.append("  }")
    ctx.lines.append("#endif")

    emit_global_avg_pool_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
