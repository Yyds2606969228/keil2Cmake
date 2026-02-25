#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

OUTPUT_ZH = ROOT / "docs" / "onnx_opset12_coverage_matrix.md"
OUTPUT_EN = ROOT / "docs" / "onnx_opset12_coverage_matrix_EN.md"
CODEGEN = SRC / "keil2cmake" / "tinyml" / "codegen.py"
C_OPS_DIR = SRC / "keil2cmake" / "tinyml" / "backends" / "c" / "ops"


@dataclass(frozen=True)
class Row:
    op: str
    c_native: bool
    quantized: bool
    level: str  # full | constrained | not_implemented


C_CONSTRAINTS_EN: dict[str, str] = {
    "Gather": "Current subset: constant 1D indices.",
    "GatherND": "Current subset: batch_dims=0.",
    "ArgMax": "Supports axis/keepdims/select_last_index.",
    "ArgMin": "Supports axis/keepdims/select_last_index.",
    "Expand": "Shape input must be constant; broadcast rules applied.",
    "Where": "Broadcast semantics.",
    "Pad": "Subset: mode=constant; static rank>=1.",
    "Slice": "Subset: steps=1; static rank>=1.",
    "ReduceMean": "Supports axes/keepdims (fp32).",
    "ReduceSum": "Supports axes/keepdims (fp32).",
    "ReduceMax": "Supports axes/keepdims (fp32).",
    "ReduceMin": "Supports axes/keepdims (fp32).",
    "Transpose": "Supports static rank>=1; perm must be valid.",
    "CumSum": "Axis input must be constant scalar; supports exclusive/reverse={0,1}.",
    "EyeLike": "Current subset: rank=2.",
    "Det": "Current subset: 2D square matrix input (fp32).",
    "NonMaxSuppression": "Current subset: static 3D boxes/scores; output fixed [N,3], invalid rows=-1.",
    "Einsum": "Current subset: ij,jk->ik / bij,bjk->bik / bij,jk->bik / ij,bjk->bik.",
    "RoiAlign": "Current subset: static 4D NCHW + static rois/batch_indices shape; mode=avg/max (fp32).",
    "ReverseSequence": "Current subset: static shape; sequence_lens constant.",
    "Compress": "Current subset: condition must be constant 1D; supports axis/no-axis.",
    "BitShift": "Integer dtype; broadcast semantics; direction=LEFT/RIGHT.",
    "MatMulInteger": "Current subset: 2D; optional zero_point must be constant scalar.",
    "QLinearMatMul": "Current subset: 2D; scale/zero_point must be constant scalar.",
    "OneHot": "Depth must be constant scalar; values must be constant length-2 tensor.",
    "Scatter": "Compat subset mapped to ScatterElements semantics; reduction=none only.",
    "ScatterElements": "Supports reduction=none/add/mul/max/min.",
    "ScatterND": "Supports reduction=none/add/mul/max/min.",
    "RandomUniform": "Current subset: float32/int8/int16 output.",
    "RandomUniformLike": "Current subset: float32/int8/int16 output.",
    "RandomNormal": "Current subset: float32/int8/int16 output.",
    "RandomNormalLike": "Current subset: float32/int8/int16 output.",
    "Multinomial": "Current subset: float32/int8/int16 2D input -> int32/int64 output.",
    "Unique": "Current subset: axis=None (flatten); fixed-capacity outputs.",
    "MaxRoiPool": "Current subset: float32/int8/int16 NCHW + float32 rois[num_rois,5].",
    "If": "Registered; requires control-flow lowering before C codegen.",
    "Loop": "Registered; requires control-flow lowering before C codegen.",
    "Scan": "Registered; requires control-flow lowering before C codegen.",
    "SequenceConstruct": "Registered; requires sequence lowering before C codegen.",
    "SequenceEmpty": "Registered; requires sequence lowering before C codegen.",
    "SequenceAt": "Registered; requires sequence lowering before C codegen.",
    "SequenceInsert": "Registered; requires sequence lowering before C codegen.",
    "SequenceErase": "Registered; requires sequence lowering before C codegen.",
    "SequenceLength": "Registered; requires sequence lowering before C codegen.",
    "SplitToSequence": "Registered; requires sequence lowering before C codegen.",
    "ConcatFromSequence": "Registered; requires sequence lowering before C codegen.",
    "StringNormalizer": "Current subset: pre-tokenized numeric tensors only.",
    "TfIdfVectorizer": "Current subset: int32/int64 unigram TF/TFIDF.",
}

C_CONSTRAINTS_ZH: dict[str, str] = {
    "Gather": "当前子集：常量 1D indices。",
    "GatherND": "当前子集：batch_dims=0。",
    "ArgMax": "支持 axis/keepdims/select_last_index。",
    "ArgMin": "支持 axis/keepdims/select_last_index。",
    "Expand": "shape 输入必须为常量；按广播规则计算。",
    "Where": "支持广播语义。",
    "Pad": "子集：mode=constant；静态 rank>=1。",
    "Slice": "子集：steps=1；静态 rank>=1。",
    "ReduceMean": "支持 axes/keepdims（fp32）。",
    "ReduceSum": "支持 axes/keepdims（fp32）。",
    "ReduceMax": "支持 axes/keepdims（fp32）。",
    "ReduceMin": "支持 axes/keepdims（fp32）。",
    "Transpose": "支持静态 rank>=1；perm 必须合法。",
    "CumSum": "axis 输入必须为常量标量；支持 exclusive/reverse={0,1}。",
    "EyeLike": "当前子集：rank=2。",
    "Det": "当前子集：2D 方阵输入（fp32）。",
    "NonMaxSuppression": "当前子集：静态 3D boxes/scores；输出固定 [N,3]，无效行为 -1。",
    "Einsum": "当前子集：ij,jk->ik / bij,bjk->bik / bij,jk->bik / ij,bjk->bik。",
    "RoiAlign": "当前子集：静态 4D NCHW + 静态 rois/batch_indices；mode=avg/max（fp32）。",
    "ReverseSequence": "当前子集：静态 shape；sequence_lens 为常量。",
    "Compress": "当前子集：condition 必须为常量 1D；支持 axis/无 axis。",
    "BitShift": "整数 dtype；支持广播语义；direction=LEFT/RIGHT。",
    "MatMulInteger": "当前子集：2D；可选 zero_point 必须为常量标量。",
    "QLinearMatMul": "当前子集：2D；scale/zero_point 必须为常量标量。",
    "OneHot": "depth 必须为常量标量；values 必须为长度 2 的常量张量。",
    "Scatter": "兼容子集：按 ScatterElements 语义映射；仅支持 reduction=none。",
    "ScatterElements": "支持 reduction=none/add/mul/max/min。",
    "ScatterND": "支持 reduction=none/add/mul/max/min。",
    "RandomUniform": "当前子集：float32/int8/int16 输出。",
    "RandomUniformLike": "当前子集：float32/int8/int16 输出。",
    "RandomNormal": "当前子集：float32/int8/int16 输出。",
    "RandomNormalLike": "当前子集：float32/int8/int16 输出。",
    "Multinomial": "当前子集：float32/int8/int16 的 2D 输入 -> int32/int64 输出。",
    "Unique": "当前子集：axis=None（展平）；固定容量输出。",
    "MaxRoiPool": "当前子集：float32/int8/int16 NCHW + float32 rois[num_rois,5]。",
    "If": "已注册；C 代码生成前需先完成控制流 lowering。",
    "Loop": "已注册；C 代码生成前需先完成控制流 lowering。",
    "Scan": "已注册；C 代码生成前需先完成控制流 lowering。",
    "SequenceConstruct": "已注册；C 代码生成前需先完成序列 lowering。",
    "SequenceEmpty": "已注册；C 代码生成前需先完成序列 lowering。",
    "SequenceAt": "已注册；C 代码生成前需先完成序列 lowering。",
    "SequenceInsert": "已注册；C 代码生成前需先完成序列 lowering。",
    "SequenceErase": "已注册；C 代码生成前需先完成序列 lowering。",
    "SequenceLength": "已注册；C 代码生成前需先完成序列 lowering。",
    "SplitToSequence": "已注册；C 代码生成前需先完成序列 lowering。",
    "ConcatFromSequence": "已注册；C 代码生成前需先完成序列 lowering。",
    "StringNormalizer": "当前子集：仅支持预分词后的数值张量。",
    "TfIdfVectorizer": "当前子集：int32/int64 unigram TF/TFIDF。",
}

FAMILY_ORDER = [
    "nn_core",
    "math",
    "reduction",
    "tensor_shape",
    "logic",
    "quant_integer",
    "recurrent",
    "vision",
    "random",
    "sequence",
    "control_flow",
    "text",
    "misc",
]


FAMILY_LABELS = {
    "nn_core": {"zh": "神经网络核心", "en": "Neural Network Core"},
    "math": {"zh": "逐元素与数学", "en": "Elementwise & Math"},
    "reduction": {"zh": "归约与索引", "en": "Reduction & Index"},
    "tensor_shape": {"zh": "张量形状与布局", "en": "Tensor Shape & Layout"},
    "logic": {"zh": "逻辑与比较", "en": "Logic & Comparison"},
    "quant_integer": {"zh": "量化与整型", "en": "Quantization & Integer"},
    "recurrent": {"zh": "循环神经网络", "en": "Recurrent Neural Network"},
    "vision": {"zh": "视觉与检测", "en": "Vision & Detection"},
    "random": {"zh": "随机与采样", "en": "Random & Sampling"},
    "sequence": {"zh": "序列", "en": "Sequence"},
    "control_flow": {"zh": "控制流", "en": "Control Flow"},
    "text": {"zh": "文本", "en": "Text"},
    "misc": {"zh": "其他", "en": "Misc"},
}


NN_CORE_OPS = {
    "AveragePool",
    "BatchNormalization",
    "Celu",
    "Conv",
    "ConvTranspose",
    "DepthToSpace",
    "Dropout",
    "Elu",
    "Gemm",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "HardSigmoid",
    "Hardmax",
    "InstanceNormalization",
    "LRN",
    "LeakyRelu",
    "LogSoftmax",
    "LpNormalization",
    "LpPool",
    "MaxPool",
    "MaxUnpool",
    "MeanVarianceNormalization",
    "NegativeLogLikelihoodLoss",
    "PRelu",
    "Relu",
    "Selu",
    "Shrink",
    "Sigmoid",
    "Softmax",
    "SoftmaxCrossEntropyLoss",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "ThresholdedRelu",
    "Tanh",
}


MATH_OPS = {
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "Clip",
    "Cos",
    "Cosh",
    "Div",
    "Erf",
    "Exp",
    "Floor",
    "IsInf",
    "IsNaN",
    "Log",
    "MatMul",
    "Max",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Neg",
    "Pow",
    "Reciprocal",
    "Round",
    "Sign",
    "Sin",
    "Sinh",
    "Sqrt",
    "Sub",
    "Sum",
    "Tan",
}


REDUCTION_OPS = {
    "ArgMax",
    "ArgMin",
    "CumSum",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "TopK",
}


TENSOR_SHAPE_OPS = {
    "Cast",
    "Compress",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Det",
    "Einsum",
    "Expand",
    "EyeLike",
    "Flatten",
    "Gather",
    "GatherElements",
    "GatherND",
    "Identity",
    "NonZero",
    "OneHot",
    "Pad",
    "Range",
    "Reshape",
    "Resize",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "Shape",
    "Size",
    "Slice",
    "Split",
    "Squeeze",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "Unique",
    "Upsample",
    "Where",
}


LOGIC_OPS = {
    "And",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Not",
    "Or",
    "Xor",
}


QUANT_INTEGER_OPS = {
    "BitShift",
    "ConvInteger",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "MatMulInteger",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
}


RNN_OPS = {
    "GRU",
    "LSTM",
    "RNN",
}


VISION_OPS = {
    "MaxRoiPool",
    "NonMaxSuppression",
    "RoiAlign",
}


RANDOM_OPS = {
    "Bernoulli",
    "Multinomial",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
}


SEQUENCE_OPS = {
    "ConcatFromSequence",
    "ReverseSequence",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SplitToSequence",
}


CONTROL_FLOW_OPS = {
    "If",
    "Loop",
    "Scan",
}


TEXT_OPS = {
    "StringNormalizer",
    "TfIdfVectorizer",
}


I18N = {
    "zh": {
        "title": "# ONNX Opset12 覆盖矩阵（C 后端）",
        "counterpart": "- 英文版: `onnx_opset12_coverage_matrix_EN.md`",
        "generated": "生成时间（UTC）",
        "scope": "统计范围",
        "scope_val": "`domain in ('', 'ai.onnx')` 且 `since_version <= 12`",
        "backend_rule": "后端规则",
        "backend_rule_val": "仅统计 C 后端",
        "default_cli": "默认 CLI",
        "quant_rule": "量化判定",
        "quant_rule_val": "基于模型中的 Q/DQ 与张量 dtype 推断",
        "weights_rule": "权重存储",
        "weights_rule_val": "保持 ONNX 原始 dtype（`float/int8/int16/int32/int64`）",
        "quick": "## 快速结论",
        "summary": "## 覆盖总览",
        "legend": "## 级别说明",
        "constrained": "## 受约束算子列表",
        "families": "## 算子家族视图",
        "families_detail": "### 家族详情（按支持级别）",
        "details": "## 明细矩阵",
        "table_metric": "指标",
        "table_count": "数量",
        "table_ratio": "占比",
        "table_family": "家族",
        "table_total": "总数",
        "table_full": "完整支持",
        "table_cons": "受约束",
        "table_missing": "未实现",
        "metric_total": "Opset12 算子总数",
        "metric_c": "C 后端覆盖",
        "metric_quant": "量化覆盖率（`quant_ops ∩ C`）",
        "metric_full": "完整支持（无额外约束）",
        "metric_cons": "受约束支持",
        "metric_missing": "未实现",
        "quick_line_1": "- C 后端覆盖率：`{c_supported}/{total}`（{c_rate:.1f}%）",
        "quick_line_2": "- 量化覆盖率（按 `quant_ops`）：`{quant_cov}/{c_supported}`（{quant_rate:.1f}%）",
        "quick_line_3": "- 受约束算子：`{cons_count}` 个，建议优先评审该列表",
        "legend_full": "- `完整支持`：已实现，且无额外子集约束",
        "legend_cons": "- `受约束`：已实现，但存在形状/类型/属性限制",
        "legend_missing": "- `未实现`：当前尚无 C 后端实现",
        "none": "_无_",
        "section_a": "### A. 受约束算子",
        "section_b": "### B. 完整支持算子",
        "section_c": "### C. 未实现算子",
        "table_op": "算子",
        "table_c": "C",
        "table_q": "量化(int8/int16)",
        "table_level": "级别",
        "table_note": "说明",
        "level_full": "完整支持",
        "level_cons": "受约束",
        "level_missing": "未实现",
        "basic_note": "基础支持。",
        "missing_note": "未实现。",
        "family_line_full": "完整支持",
        "family_line_cons": "受约束",
        "family_line_missing": "未实现",
        "quant_set": "## 量化算子集合（`codegen.py::quant_ops`）",
    },
    "en": {
        "title": "# ONNX Opset12 Coverage Matrix (C Backend)",
        "counterpart": "- Chinese version: `onnx_opset12_coverage_matrix.md`",
        "generated": "Generated at (UTC)",
        "scope": "Scope",
        "scope_val": "`domain in ('', 'ai.onnx')` and `since_version <= 12`",
        "backend_rule": "Backend rule",
        "backend_rule_val": "C backend only",
        "default_cli": "Default CLI",
        "quant_rule": "Quantization rule",
        "quant_rule_val": "Inferred from model Q/DQ and tensor dtypes",
        "weights_rule": "Weight storage",
        "weights_rule_val": "Keep original ONNX dtype (`float/int8/int16/int32/int64`)",
        "quick": "## Quick Takeaways",
        "summary": "## Coverage Summary",
        "legend": "## Level Legend",
        "constrained": "## Constrained Operators",
        "families": "## Operator Family View",
        "families_detail": "### Family Details (by level)",
        "details": "## Matrix Details",
        "table_metric": "Metric",
        "table_count": "Count",
        "table_ratio": "Ratio",
        "table_family": "Family",
        "table_total": "Total",
        "table_full": "Full",
        "table_cons": "Constrained",
        "table_missing": "Not Implemented",
        "metric_total": "Opset12 operators",
        "metric_c": "Covered by C backend",
        "metric_quant": "Quantized coverage (`quant_ops ∩ C`)",
        "metric_full": "Fully supported (no extra constraints)",
        "metric_cons": "Constrained support",
        "metric_missing": "Not implemented",
        "quick_line_1": "- C backend coverage: `{c_supported}/{total}` ({c_rate:.1f}%)",
        "quick_line_2": "- Quantized coverage (by `quant_ops`): `{quant_cov}/{c_supported}` ({quant_rate:.1f}%)",
        "quick_line_3": "- Constrained operators: `{cons_count}`; review this list first",
        "legend_full": "- `Full`: implemented with no extra subset constraints",
        "legend_cons": "- `Constrained`: implemented, but with shape/type/attribute constraints",
        "legend_missing": "- `Not Implemented`: no C backend implementation yet",
        "none": "_None_",
        "section_a": "### A. Constrained Operators",
        "section_b": "### B. Fully Supported Operators",
        "section_c": "### C. Not Implemented",
        "table_op": "Operator",
        "table_c": "C",
        "table_q": "Quant(int8/int16)",
        "table_level": "Level",
        "table_note": "Notes",
        "level_full": "Full",
        "level_cons": "Constrained",
        "level_missing": "Not Implemented",
        "basic_note": "Basic support.",
        "missing_note": "Not implemented.",
        "family_line_full": "Full",
        "family_line_cons": "Constrained",
        "family_line_missing": "Not Implemented",
        "quant_set": "## Quant Operator Set (`codegen.py::quant_ops`)",
    },
}



def _extract_string_set(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Set):
        return {
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple, ast.Set))
    ):
        out: set[str] = set()
        for elt in node.args[0].elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.add(elt.value)
        return out
    return set()


def _load_quant_ops(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if any(t.id == "quant_ops" for t in targets):
                out = _extract_string_set(node.value)
                if out:
                    return out
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "quant_ops"
        ):
            if node.value is None:
                continue
            out = _extract_string_set(node.value)
            if out:
                return out
    raise RuntimeError("Failed to locate quant_ops in codegen.py")


def _scan_registered_ops(ops_dir: Path) -> set[str]:
    out: set[str] = set()
    for py in sorted(ops_dir.glob("*.py")):
        if py.name in ("__init__.py", "registry.py"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "register_op"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    and isinstance(dec.args[0].value, str)
                ):
                    out.add(dec.args[0].value)
    return out


def _opset12_ops() -> list[str]:
    latest_by_name: dict[str, int] = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.domain not in ("", "ai.onnx"):
            continue
        if schema.since_version > 12:
            continue
        prev = latest_by_name.get(schema.name)
        if prev is None or schema.since_version > prev:
            latest_by_name[schema.name] = schema.since_version
    return sorted(latest_by_name.keys())


def _mark(v: bool) -> str:
    return "Y" if v else "N"


def _level(c_native: bool, op: str) -> str:
    if not c_native:
        return "not_implemented"
    if _family_for(op) in ("reduction", "tensor_shape", "quant_integer", "recurrent"):
        return "full"
    if op in C_CONSTRAINTS_EN:
        return "constrained"
    return "full"


def _family_for(op: str) -> str:
    if op in CONTROL_FLOW_OPS:
        return "control_flow"
    if op in SEQUENCE_OPS:
        return "sequence"
    if op in RNN_OPS:
        return "recurrent"
    if op in VISION_OPS:
        return "vision"
    if op in QUANT_INTEGER_OPS:
        return "quant_integer"
    if op in RANDOM_OPS:
        return "random"
    if op in TEXT_OPS:
        return "text"
    if op in REDUCTION_OPS or op.startswith("Reduce"):
        return "reduction"
    if op in LOGIC_OPS:
        return "logic"
    if op in NN_CORE_OPS:
        return "nn_core"
    if op in MATH_OPS:
        return "math"
    if op in TENSOR_SHAPE_OPS:
        return "tensor_shape"
    return "misc"


def _build_rows(ops: list[str], quant_ops: set[str], c_ops: set[str]) -> list[Row]:
    rows: list[Row] = []
    for op in ops:
        c_native = op in c_ops
        rows.append(
            Row(
                op=op,
                c_native=c_native,
                quantized=op in quant_ops,
                level=_level(c_native, op),
            )
        )
    return rows


def _ratio(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return (float(n) / float(d)) * 100.0


def _note_for_row(row: Row, lang: str) -> str:
    loc = I18N[lang]
    if row.level == "full":
        return loc["basic_note"]
    if row.level == "not_implemented":
        return loc["missing_note"]
    if lang == "zh":
        return C_CONSTRAINTS_ZH.get(row.op, C_CONSTRAINTS_EN.get(row.op, loc["basic_note"]))
    return C_CONSTRAINTS_EN.get(row.op, loc["basic_note"])


def _level_label(level: str, lang: str) -> str:
    loc = I18N[lang]
    if level == "full":
        return loc["level_full"]
    if level == "constrained":
        return loc["level_cons"]
    return loc["level_missing"]


def _render_table(rows: list[Row], lang: str) -> list[str]:
    loc = I18N[lang]
    out: list[str] = []
    out.append(
        f"| {loc['table_op']} | {loc['table_c']} | {loc['table_q']} | {loc['table_level']} | {loc['table_note']} |"
    )
    out.append("|---|---:|---:|---|---|")
    for r in rows:
        out.append(
            f"| {r.op} | {_mark(r.c_native)} | {_mark(r.quantized)} | {_level_label(r.level, lang)} | {_note_for_row(r, lang)} |"
        )
    out.append("")
    return out


def _format_ops(rows: list[Row], level: str, none_text: str) -> str:
    names = sorted(r.op for r in rows if r.level == level)
    if not names:
        return none_text
    return ", ".join(f"`{name}`" for name in names)


def _render_family_view(rows: list[Row], lang: str) -> list[str]:
    loc = I18N[lang]
    out: list[str] = []
    total = len(rows)
    families = {name: [] for name in FAMILY_ORDER}

    for row in rows:
        families[_family_for(row.op)].append(row)

    out.append(loc["families"])
    out.append("")
    out.append(
        f"| {loc['table_family']} | {loc['table_total']} | {loc['table_full']} | {loc['table_cons']} | {loc['table_missing']} | {loc['table_ratio']} |"
    )
    out.append("|---|---:|---:|---:|---:|---:|")

    for family in FAMILY_ORDER:
        family_rows = families[family]
        if not family_rows:
            continue
        family_total = len(family_rows)
        full_count = sum(1 for r in family_rows if r.level == "full")
        cons_count = sum(1 for r in family_rows if r.level == "constrained")
        missing_count = sum(1 for r in family_rows if r.level == "not_implemented")
        ratio = _ratio(family_total, total)
        label = FAMILY_LABELS[family][lang]
        out.append(
            f"| {label} | {family_total} | {full_count} | {cons_count} | {missing_count} | {ratio:.1f}% |"
        )

    out.append("")
    out.append(loc["families_detail"])
    out.append("")

    for family in FAMILY_ORDER:
        family_rows = families[family]
        if not family_rows:
            continue
        label = FAMILY_LABELS[family][lang]
        out.append(f"#### {label} ({len(family_rows)})")
        out.append("")
        out.append(
            f"- {loc['family_line_full']}: {_format_ops(family_rows, 'full', loc['none'])}"
        )
        out.append(
            f"- {loc['family_line_cons']}: {_format_ops(family_rows, 'constrained', loc['none'])}"
        )
        out.append(
            f"- {loc['family_line_missing']}: {_format_ops(family_rows, 'not_implemented', loc['none'])}"
        )
        out.append("")
    return out


def _render(rows: list[Row], quant_ops: set[str], lang: str) -> str:
    loc = I18N[lang]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    total = len(rows)
    c_supported = sum(1 for r in rows if r.c_native)
    quant_cov = sum(1 for r in rows if r.quantized and r.c_native)
    full_count = sum(1 for r in rows if r.level == "full")
    cons_count = sum(1 for r in rows if r.level == "constrained")
    missing_count = sum(1 for r in rows if r.level == "not_implemented")

    c_rate = _ratio(c_supported, total)
    quant_rate = _ratio(quant_cov, c_supported)
    full_rate = _ratio(full_count, total)
    cons_rate = _ratio(cons_count, total)
    missing_rate = _ratio(missing_count, total)

    constrained_rows = sorted((r for r in rows if r.level == "constrained"), key=lambda r: r.op)
    full_rows = sorted((r for r in rows if r.level == "full"), key=lambda r: r.op)
    missing_rows = sorted((r for r in rows if r.level == "not_implemented"), key=lambda r: r.op)

    constrained_ops = ", ".join(f"`{r.op}`" for r in constrained_rows)

    lines: list[str] = []
    lines.append(loc["title"])
    lines.append("")
    lines.append(loc["counterpart"])
    lines.append(f"- {loc['generated']}: `{now}`")
    lines.append(f"- {loc['scope']}: {loc['scope_val']}")
    lines.append(f"- {loc['backend_rule']}: {loc['backend_rule_val']}")
    lines.append(f"- {loc['default_cli']}: `--weights flash --emit c`")
    lines.append(f"- {loc['quant_rule']}: {loc['quant_rule_val']}")
    lines.append(f"- {loc['weights_rule']}: {loc['weights_rule_val']}")
    lines.append("")
    lines.append(loc["quick"])
    lines.append("")
    lines.append(loc["quick_line_1"].format(c_supported=c_supported, total=total, c_rate=c_rate))
    lines.append(
        loc["quick_line_2"].format(quant_cov=quant_cov, c_supported=c_supported, quant_rate=quant_rate)
    )
    lines.append(loc["quick_line_3"].format(cons_count=cons_count))
    lines.append("")
    lines.append(loc["summary"])
    lines.append("")
    lines.append(f"| {loc['table_metric']} | {loc['table_count']} | {loc['table_ratio']} |")
    lines.append("|---|---:|---:|")
    lines.append(f"| {loc['metric_total']} | {total} | 100.0% |")
    lines.append(f"| {loc['metric_c']} | {c_supported} | {c_rate:.1f}% |")
    lines.append(f"| {loc['metric_quant']} | {quant_cov} | {quant_rate:.1f}% |")
    lines.append(f"| {loc['metric_full']} | {full_count} | {full_rate:.1f}% |")
    lines.append(f"| {loc['metric_cons']} | {cons_count} | {cons_rate:.1f}% |")
    lines.append(f"| {loc['metric_missing']} | {missing_count} | {missing_rate:.1f}% |")
    lines.append("")
    lines.append(loc["legend"])
    lines.append("")
    lines.append(loc["legend_full"])
    lines.append(loc["legend_cons"])
    lines.append(loc["legend_missing"])
    lines.append("")
    lines.append(loc["constrained"])
    lines.append("")
    lines.append(constrained_ops if constrained_ops else loc["none"])
    lines.append("")
    lines.extend(_render_family_view(rows, lang))
    lines.append("")
    lines.append(loc["details"])
    lines.append("")
    lines.append(loc["section_a"])
    lines.append("")
    lines.extend(_render_table(constrained_rows, lang))
    lines.append(loc["section_b"])
    lines.append("")
    lines.extend(_render_table(full_rows, lang))
    lines.append(loc["section_c"])
    lines.append("")
    lines.extend(_render_table(missing_rows, lang))
    lines.append(loc["quant_set"])
    lines.append("")
    lines.append("```text")
    lines.append(", ".join(sorted(quant_ops)))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    quant_ops = _load_quant_ops(CODEGEN)
    c_ops = _scan_registered_ops(C_OPS_DIR)
    ops = _opset12_ops()
    rows = _build_rows(ops, quant_ops, c_ops)

    OUTPUT_ZH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ZH.write_text(_render(rows, quant_ops, "zh"), encoding="utf-8")
    OUTPUT_EN.write_text(_render(rows, quant_ops, "en"), encoding="utf-8")

    print(f"[ok] wrote {OUTPUT_ZH}")
    print(f"[ok] wrote {OUTPUT_EN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
