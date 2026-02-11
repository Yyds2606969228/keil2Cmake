# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import quantize_multiplier, tensor_size
from ...c.ops.conv import emit_conv as emit_conv_c
from .registry import register_op


@register_op("Conv")
def emit_conv(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Conv expects at least 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    x_name = node.inputs[0]
    w_name = node.inputs[1]
    b_name = node.inputs[2] if len(node.inputs) > 2 else None

    if out_dtype != "int8":
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return
    if ctx.dtype(x_name) != "int8" or ctx.dtype(w_name) != "int8":
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return

    x_shape = ctx.shape(x_name)
    w_shape = ctx.shape(w_name)
    out_shape = ctx.shape(out_tensor)
    if len(x_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return

    n, c_in, h, w_in = x_shape
    m, c_per_g, k_h, k_w = w_shape
    n_out, c_out, out_h, out_w = out_shape
    if n != 1 or n_out != 1:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return

    strides = list(node.attrs.get("strides", [1, 1]))
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    dilations = list(node.attrs.get("dilations", [1, 1]))
    groups = int(node.attrs.get("group", 1))
    depthwise = groups == c_in == c_out and c_per_g == 1 and groups > 1
    regular = groups == 1
    if not regular and not depthwise:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return
    if len(strides) != 2 or len(dilations) != 2:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return

    stride_h, stride_w = strides
    pad_h0, pad_w0, pad_h1, pad_w1 = pads
    dil_h, dil_w = dilations
    if pad_h0 != pad_h1 or pad_w0 != pad_w1:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return

    sx, zx = ctx.qparams(x_name)
    sw, zw = ctx.qparams(w_name)
    so, zo = ctx.qparams(out_tensor)
    if zw != 0:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return
    bias_scale = sx * sw

    bias_tensor = ctx.model.tensors.get(b_name) if b_name else None
    bias_vals: list[int] | None
    if b_name is None:
        bias_vals = [0] * c_out
    else:
        bias_vals = None
        if bias_tensor is not None and bias_tensor.data is not None:
            b_size = tensor_size(ctx.shape(b_name))
            if b_size == c_out and bias_scale != 0.0:
                if bias_tensor.dtype == "float32":
                    bias_vals = [
                        int(round(float(v) / bias_scale)) for v in bias_tensor.data[:c_out]
                    ]
                elif bias_tensor.dtype in ("int32", "int64") and bias_tensor.qscale is None:
                    bias_vals = [int(v) for v in bias_tensor.data[:c_out]]
                elif bias_tensor.dtype in ("int8", "int16"):
                    if bias_tensor.qscale is not None and bias_tensor.qzero is not None:
                        sc = float(bias_tensor.qscale)
                        zc = int(bias_tensor.qzero)
                        bias_vals = [
                            int(round((float(v) - zc) * sc / bias_scale))
                            for v in bias_tensor.data[:c_out]
                        ]
                elif bias_tensor.dtype in ("int32", "int64") and bias_tensor.qscale is not None:
                    zc = int(bias_tensor.qzero or 0)
                    sc = float(bias_tensor.qscale)
                    bias_vals = [
                        int(round((float(v) - zc) * sc / bias_scale))
                        for v in bias_tensor.data[:c_out]
                    ]
    if bias_vals is None:
        ctx.backend_used = "c"
        emit_conv_c(ctx, node)
        return

    out = ctx.map_ptr(out_tensor)
    x = ctx.map_ptr(x_name)
    w = ctx.map_ptr(w_name)

    real_multiplier = (sx * sw) / so if so != 0.0 else 0.0
    multiplier, shift = quantize_multiplier(real_multiplier)
    mult_name = ctx.next_symbol("k2c_conv_mult")
    shift_name = ctx.next_symbol("k2c_conv_shift")
    bias_name = ctx.next_symbol("k2c_conv_bias")
    in_nhwc = ctx.next_symbol("k2c_conv_in")
    out_nhwc = ctx.next_symbol("k2c_conv_out")
    w_nhwc = ctx.next_symbol("k2c_conv_w")
    status_var = ctx.next_symbol("k2c_cmsis_status")
    done_label = ctx.next_symbol("k2c_cmsis_done")

    mult_vals = ", ".join(str(multiplier) for _ in range(c_out))
    shift_vals = ", ".join(str(shift) for _ in range(c_out))
    bias_vals_str = ", ".join(str(v) for v in bias_vals)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"  static const int32_t {mult_name}[{c_out}] = {{ {mult_vals} }};")
    ctx.lines.append(f"  static const int32_t {shift_name}[{c_out}] = {{ {shift_vals} }};")
    ctx.lines.append(f"  static const int32_t {bias_name}[{c_out}] = {{ {bias_vals_str} }};")
    ctx.lines.append(f"  static int8_t {in_nhwc}[{c_in} * {h} * {w_in}];")
    ctx.lines.append(f"  static int8_t {out_nhwc}[{c_out} * {out_h} * {out_w}];")
    if depthwise:
        ctx.lines.append(f"  static int8_t {w_nhwc}[{k_h} * {k_w} * {c_out}];")
    else:
        ctx.lines.append(f"  static int8_t {w_nhwc}[{c_out} * {k_h} * {k_w} * {c_in}];")

    ctx.lines.append(f"  for (size_t ih = 0; ih < {h}; ++ih) {{")
    ctx.lines.append(f"    for (size_t iw = 0; iw < {w_in}; ++iw) {{")
    ctx.lines.append(f"      for (size_t ic = 0; ic < {c_in}; ++ic) {{")
    ctx.lines.append(
        f"        {in_nhwc}[(ih * {w_in} + iw) * {c_in} + ic] = "
        f"{x}[ic * {h} * {w_in} + ih * {w_in} + iw];"
    )
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

    if depthwise:
        ctx.lines.append(f"  for (size_t oc = 0; oc < {c_out}; ++oc) {{")
        ctx.lines.append(f"    for (size_t kh = 0; kh < {k_h}; ++kh) {{")
        ctx.lines.append(f"      for (size_t kw = 0; kw < {k_w}; ++kw) {{")
        ctx.lines.append(
            f"        {w_nhwc}[(kh * {k_w} + kw) * {c_out} + oc] = "
            f"{w}[oc * {k_h} * {k_w} + kh * {k_w} + kw];"
        )
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    else:
        ctx.lines.append(f"  for (size_t oc = 0; oc < {c_out}; ++oc) {{")
        ctx.lines.append(f"    for (size_t kh = 0; kh < {k_h}; ++kh) {{")
        ctx.lines.append(f"      for (size_t kw = 0; kw < {k_w}; ++kw) {{")
        ctx.lines.append(f"        for (size_t ic = 0; ic < {c_in}; ++ic) {{")
        ctx.lines.append(
            f"          {w_nhwc}[((oc * {k_h} + kh) * {k_w} + kw) * {c_in} + ic] = "
            f"{w}[oc * {c_in} * {k_h} * {k_w} + ic * {k_h} * {k_w} + kh * {k_w} + kw];"
        )
        ctx.lines.append("        }")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")

    if depthwise:
        ctx.lines.append("  cmsis_nn_dw_conv_params conv_params = {")
    else:
        ctx.lines.append("  cmsis_nn_conv_params conv_params = {")
    ctx.lines.append(f"    .input_offset = {-zx},")
    ctx.lines.append(f"    .output_offset = {-zo},")
    ctx.lines.append("    .activation = { .min = -128, .max = 127 },")
    ctx.lines.append(f"    .stride = {{ .h = {stride_h}, .w = {stride_w} }},")
    ctx.lines.append(f"    .padding = {{ .h = {pad_h0}, .w = {pad_w0} }},")
    ctx.lines.append(f"    .dilation = {{ .h = {dil_h}, .w = {dil_w} }},")
    if depthwise:
        ctx.lines.append("    .ch_mult = 1,")
    ctx.lines.append("  };")
    ctx.lines.append(
        f"  cmsis_nn_per_channel_quant_params quant_params = {{ .multiplier = {mult_name}, .shift = {shift_name} }};"
    )
    ctx.lines.append(f"  cmsis_nn_dims input_dims = {{ .n = 1, .h = {h}, .w = {w_in}, .c = {c_in} }};")
    if depthwise:
        ctx.lines.append(
            f"  cmsis_nn_dims filter_dims = {{ .n = 1, .h = {k_h}, .w = {k_w}, .c = {c_out} }};"
        )
    else:
        ctx.lines.append(
            f"  cmsis_nn_dims filter_dims = {{ .n = {c_out}, .h = {k_h}, .w = {k_w}, .c = {c_in} }};"
        )
    ctx.lines.append(f"  cmsis_nn_dims bias_dims = {{ .n = 1, .h = 1, .w = 1, .c = {c_out} }};")
    ctx.lines.append(
        f"  cmsis_nn_dims output_dims = {{ .n = 1, .h = {out_h}, .w = {out_w}, .c = {c_out} }};"
    )
    ctx.lines.append(f"  arm_cmsis_nn_status {status_var} = ARM_CMSIS_NN_SUCCESS;")
    if depthwise:
        ctx.lines.append(
            "  int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(&input_dims, &filter_dims);"
        )
        ctx.lines.append("  if (buf_size > 0) {")
        ctx.lines.append("    int8_t scratch[buf_size];")
        ctx.lines.append("    cmsis_nn_context nn_ctx = { .buf = scratch, .size = (uint32_t)buf_size };")
        ctx.lines.append(
            f"    {status_var} = arm_depthwise_conv_wrapper_s8("
            "&nn_ctx, &conv_params, &quant_params, &input_dims, "
            f"{in_nhwc}, &filter_dims, {w_nhwc}, &bias_dims, {bias_name}, "
            f"&output_dims, {out_nhwc});"
        )
        ctx.lines.append("  } else {")
        ctx.lines.append("    cmsis_nn_context nn_ctx = { .buf = NULL, .size = 0 };")
        ctx.lines.append(
            f"    {status_var} = arm_depthwise_conv_wrapper_s8("
            "&nn_ctx, &conv_params, &quant_params, &input_dims, "
            f"{in_nhwc}, &filter_dims, {w_nhwc}, &bias_dims, {bias_name}, "
            f"&output_dims, {out_nhwc});"
        )
        ctx.lines.append("  }")
    else:
        ctx.lines.append("  int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&input_dims, &filter_dims);")
        ctx.lines.append("  if (buf_size > 0) {")
        ctx.lines.append("    int8_t scratch[buf_size];")
        ctx.lines.append("    cmsis_nn_context nn_ctx = { .buf = scratch, .size = (uint32_t)buf_size };")
        ctx.lines.append(
            f"    {status_var} = arm_convolve_wrapper_s8("
            "&nn_ctx, &conv_params, &quant_params, &input_dims, "
            f"{in_nhwc}, &filter_dims, {w_nhwc}, &bias_dims, {bias_name}, "
            f"&output_dims, {out_nhwc});"
        )
        ctx.lines.append("  } else {")
        ctx.lines.append("    cmsis_nn_context nn_ctx = { .buf = NULL, .size = 0 };")
        ctx.lines.append(
            f"    {status_var} = arm_convolve_wrapper_s8("
            "&nn_ctx, &conv_params, &quant_params, &input_dims, "
            f"{in_nhwc}, &filter_dims, {w_nhwc}, &bias_dims, {bias_name}, "
            f"&output_dims, {out_nhwc});"
        )
        ctx.lines.append("  }")

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

    emit_conv_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
