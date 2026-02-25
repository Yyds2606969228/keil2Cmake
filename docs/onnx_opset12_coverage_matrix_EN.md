# ONNX Opset12 Coverage Matrix (C Backend)

- Chinese version: `onnx_opset12_coverage_matrix.md`
- Generated at (UTC): `2026-02-25 03:27:29Z`
- Scope: `domain in ('', 'ai.onnx')` and `since_version <= 12`
- Backend rule: C backend only
- Default CLI: `--weights flash --emit c`
- Quantization rule: Inferred from model Q/DQ and tensor dtypes
- Weight storage: Keep original ONNX dtype (`float/int8/int16/int32/int64`)

## Quick Takeaways

- C backend coverage: `162/162` (100.0%)
- Quantized coverage (by `quant_ops`): `162/162` (100.0%)
- Constrained operators: `22`; review this list first

## Coverage Summary

| Metric | Count | Ratio |
|---|---:|---:|
| Opset12 operators | 162 | 100.0% |
| Covered by C backend | 162 | 100.0% |
| Quantized coverage (`quant_ops ∩ C`) | 162 | 100.0% |
| Fully supported (no extra constraints) | 140 | 86.4% |
| Constrained support | 22 | 13.6% |
| Not implemented | 0 | 0.0% |

## Level Legend

- `Full`: implemented with no extra subset constraints
- `Constrained`: implemented, but with shape/type/attribute constraints
- `Not Implemented`: no C backend implementation yet

## Constrained Operators

`ConcatFromSequence`, `If`, `Loop`, `MaxRoiPool`, `Multinomial`, `NonMaxSuppression`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `ReverseSequence`, `RoiAlign`, `Scan`, `SequenceAt`, `SequenceConstruct`, `SequenceEmpty`, `SequenceErase`, `SequenceInsert`, `SequenceLength`, `SplitToSequence`, `StringNormalizer`, `TfIdfVectorizer`

## Operator Family View

| Family | Total | Full | Constrained | Not Implemented | Ratio |
|---|---:|---:|---:|---:|---:|
| Neural Network Core | 36 | 36 | 0 | 0 | 22.2% |
| Elementwise & Math | 36 | 36 | 0 | 0 | 22.2% |
| Reduction & Index | 14 | 14 | 0 | 0 | 8.6% |
| Tensor Shape & Layout | 34 | 34 | 0 | 0 | 21.0% |
| Logic & Comparison | 9 | 9 | 0 | 0 | 5.6% |
| Quantization & Integer | 8 | 8 | 0 | 0 | 4.9% |
| Recurrent Neural Network | 3 | 3 | 0 | 0 | 1.9% |
| Vision & Detection | 3 | 0 | 3 | 0 | 1.9% |
| Random & Sampling | 5 | 0 | 5 | 0 | 3.1% |
| Sequence | 9 | 0 | 9 | 0 | 5.6% |
| Control Flow | 3 | 0 | 3 | 0 | 1.9% |
| Text | 2 | 0 | 2 | 0 | 1.2% |

### Family Details (by level)

#### Neural Network Core (36)

- Full: `AveragePool`, `BatchNormalization`, `Celu`, `Conv`, `ConvTranspose`, `DepthToSpace`, `Dropout`, `Elu`, `Gemm`, `GlobalAveragePool`, `GlobalLpPool`, `GlobalMaxPool`, `HardSigmoid`, `Hardmax`, `InstanceNormalization`, `LRN`, `LeakyRelu`, `LogSoftmax`, `LpNormalization`, `LpPool`, `MaxPool`, `MaxUnpool`, `MeanVarianceNormalization`, `NegativeLogLikelihoodLoss`, `PRelu`, `Relu`, `Selu`, `Shrink`, `Sigmoid`, `Softmax`, `SoftmaxCrossEntropyLoss`, `Softplus`, `Softsign`, `SpaceToDepth`, `Tanh`, `ThresholdedRelu`
- Constrained: _None_
- Not Implemented: _None_

#### Elementwise & Math (36)

- Full: `Abs`, `Acos`, `Acosh`, `Add`, `Asin`, `Asinh`, `Atan`, `Atanh`, `Ceil`, `Clip`, `Cos`, `Cosh`, `Div`, `Erf`, `Exp`, `Floor`, `IsInf`, `IsNaN`, `Log`, `MatMul`, `Max`, `Mean`, `Min`, `Mod`, `Mul`, `Neg`, `Pow`, `Reciprocal`, `Round`, `Sign`, `Sin`, `Sinh`, `Sqrt`, `Sub`, `Sum`, `Tan`
- Constrained: _None_
- Not Implemented: _None_

#### Reduction & Index (14)

- Full: `ArgMax`, `ArgMin`, `CumSum`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceMax`, `ReduceMean`, `ReduceMin`, `ReduceProd`, `ReduceSum`, `ReduceSumSquare`, `TopK`
- Constrained: _None_
- Not Implemented: _None_

#### Tensor Shape & Layout (34)

- Full: `Cast`, `Compress`, `Concat`, `Constant`, `ConstantOfShape`, `Det`, `Einsum`, `Expand`, `EyeLike`, `Flatten`, `Gather`, `GatherElements`, `GatherND`, `Identity`, `NonZero`, `OneHot`, `Pad`, `Range`, `Reshape`, `Resize`, `Scatter`, `ScatterElements`, `ScatterND`, `Shape`, `Size`, `Slice`, `Split`, `Squeeze`, `Tile`, `Transpose`, `Unique`, `Unsqueeze`, `Upsample`, `Where`
- Constrained: _None_
- Not Implemented: _None_

#### Logic & Comparison (9)

- Full: `And`, `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Not`, `Or`, `Xor`
- Constrained: _None_
- Not Implemented: _None_

#### Quantization & Integer (8)

- Full: `BitShift`, `ConvInteger`, `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger`, `QLinearConv`, `QLinearMatMul`, `QuantizeLinear`
- Constrained: _None_
- Not Implemented: _None_

#### Recurrent Neural Network (3)

- Full: `GRU`, `LSTM`, `RNN`
- Constrained: _None_
- Not Implemented: _None_

#### Vision & Detection (3)

- Full: _None_
- Constrained: `MaxRoiPool`, `NonMaxSuppression`, `RoiAlign`
- Not Implemented: _None_

#### Random & Sampling (5)

- Full: _None_
- Constrained: `Multinomial`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`
- Not Implemented: _None_

#### Sequence (9)

- Full: _None_
- Constrained: `ConcatFromSequence`, `ReverseSequence`, `SequenceAt`, `SequenceConstruct`, `SequenceEmpty`, `SequenceErase`, `SequenceInsert`, `SequenceLength`, `SplitToSequence`
- Not Implemented: _None_

#### Control Flow (3)

- Full: _None_
- Constrained: `If`, `Loop`, `Scan`
- Not Implemented: _None_

#### Text (2)

- Full: _None_
- Constrained: `StringNormalizer`, `TfIdfVectorizer`
- Not Implemented: _None_


## Matrix Details

### A. Constrained Operators

| Operator | C | Quant(int8/int16) | Level | Notes |
|---|---:|---:|---|---|
| ConcatFromSequence | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| If | Y | Y | Constrained | Registered; requires control-flow lowering before C codegen. |
| Loop | Y | Y | Constrained | Registered; requires control-flow lowering before C codegen. |
| MaxRoiPool | Y | Y | Constrained | Current subset: float32/int8/int16 NCHW + float32 rois[num_rois,5]. |
| Multinomial | Y | Y | Constrained | Current subset: float32/int8/int16 2D input -> int32/int64 output. |
| NonMaxSuppression | Y | Y | Constrained | Current subset: static 3D boxes/scores; output fixed [N,3], invalid rows=-1. |
| RandomNormal | Y | Y | Constrained | Current subset: float32/int8/int16 output. |
| RandomNormalLike | Y | Y | Constrained | Current subset: float32/int8/int16 output. |
| RandomUniform | Y | Y | Constrained | Current subset: float32/int8/int16 output. |
| RandomUniformLike | Y | Y | Constrained | Current subset: float32/int8/int16 output. |
| ReverseSequence | Y | Y | Constrained | Current subset: static shape; sequence_lens constant. |
| RoiAlign | Y | Y | Constrained | Current subset: static 4D NCHW + static rois/batch_indices shape; mode=avg/max (fp32). |
| Scan | Y | Y | Constrained | Registered; requires control-flow lowering before C codegen. |
| SequenceAt | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| SequenceConstruct | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| SequenceEmpty | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| SequenceErase | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| SequenceInsert | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| SequenceLength | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| SplitToSequence | Y | Y | Constrained | Registered; requires sequence lowering before C codegen. |
| StringNormalizer | Y | Y | Constrained | Current subset: pre-tokenized numeric tensors only. |
| TfIdfVectorizer | Y | Y | Constrained | Current subset: int32/int64 unigram TF/TFIDF. |

### B. Fully Supported Operators

| Operator | C | Quant(int8/int16) | Level | Notes |
|---|---:|---:|---|---|
| Abs | Y | Y | Full | Basic support. |
| Acos | Y | Y | Full | Basic support. |
| Acosh | Y | Y | Full | Basic support. |
| Add | Y | Y | Full | Basic support. |
| And | Y | Y | Full | Basic support. |
| ArgMax | Y | Y | Full | Basic support. |
| ArgMin | Y | Y | Full | Basic support. |
| Asin | Y | Y | Full | Basic support. |
| Asinh | Y | Y | Full | Basic support. |
| Atan | Y | Y | Full | Basic support. |
| Atanh | Y | Y | Full | Basic support. |
| AveragePool | Y | Y | Full | Basic support. |
| BatchNormalization | Y | Y | Full | Basic support. |
| BitShift | Y | Y | Full | Basic support. |
| Cast | Y | Y | Full | Basic support. |
| Ceil | Y | Y | Full | Basic support. |
| Celu | Y | Y | Full | Basic support. |
| Clip | Y | Y | Full | Basic support. |
| Compress | Y | Y | Full | Basic support. |
| Concat | Y | Y | Full | Basic support. |
| Constant | Y | Y | Full | Basic support. |
| ConstantOfShape | Y | Y | Full | Basic support. |
| Conv | Y | Y | Full | Basic support. |
| ConvInteger | Y | Y | Full | Basic support. |
| ConvTranspose | Y | Y | Full | Basic support. |
| Cos | Y | Y | Full | Basic support. |
| Cosh | Y | Y | Full | Basic support. |
| CumSum | Y | Y | Full | Basic support. |
| DepthToSpace | Y | Y | Full | Basic support. |
| DequantizeLinear | Y | Y | Full | Basic support. |
| Det | Y | Y | Full | Basic support. |
| Div | Y | Y | Full | Basic support. |
| Dropout | Y | Y | Full | Basic support. |
| DynamicQuantizeLinear | Y | Y | Full | Basic support. |
| Einsum | Y | Y | Full | Basic support. |
| Elu | Y | Y | Full | Basic support. |
| Equal | Y | Y | Full | Basic support. |
| Erf | Y | Y | Full | Basic support. |
| Exp | Y | Y | Full | Basic support. |
| Expand | Y | Y | Full | Basic support. |
| EyeLike | Y | Y | Full | Basic support. |
| Flatten | Y | Y | Full | Basic support. |
| Floor | Y | Y | Full | Basic support. |
| GRU | Y | Y | Full | Basic support. |
| Gather | Y | Y | Full | Basic support. |
| GatherElements | Y | Y | Full | Basic support. |
| GatherND | Y | Y | Full | Basic support. |
| Gemm | Y | Y | Full | Basic support. |
| GlobalAveragePool | Y | Y | Full | Basic support. |
| GlobalLpPool | Y | Y | Full | Basic support. |
| GlobalMaxPool | Y | Y | Full | Basic support. |
| Greater | Y | Y | Full | Basic support. |
| GreaterOrEqual | Y | Y | Full | Basic support. |
| HardSigmoid | Y | Y | Full | Basic support. |
| Hardmax | Y | Y | Full | Basic support. |
| Identity | Y | Y | Full | Basic support. |
| InstanceNormalization | Y | Y | Full | Basic support. |
| IsInf | Y | Y | Full | Basic support. |
| IsNaN | Y | Y | Full | Basic support. |
| LRN | Y | Y | Full | Basic support. |
| LSTM | Y | Y | Full | Basic support. |
| LeakyRelu | Y | Y | Full | Basic support. |
| Less | Y | Y | Full | Basic support. |
| LessOrEqual | Y | Y | Full | Basic support. |
| Log | Y | Y | Full | Basic support. |
| LogSoftmax | Y | Y | Full | Basic support. |
| LpNormalization | Y | Y | Full | Basic support. |
| LpPool | Y | Y | Full | Basic support. |
| MatMul | Y | Y | Full | Basic support. |
| MatMulInteger | Y | Y | Full | Basic support. |
| Max | Y | Y | Full | Basic support. |
| MaxPool | Y | Y | Full | Basic support. |
| MaxUnpool | Y | Y | Full | Basic support. |
| Mean | Y | Y | Full | Basic support. |
| MeanVarianceNormalization | Y | Y | Full | Basic support. |
| Min | Y | Y | Full | Basic support. |
| Mod | Y | Y | Full | Basic support. |
| Mul | Y | Y | Full | Basic support. |
| Neg | Y | Y | Full | Basic support. |
| NegativeLogLikelihoodLoss | Y | Y | Full | Basic support. |
| NonZero | Y | Y | Full | Basic support. |
| Not | Y | Y | Full | Basic support. |
| OneHot | Y | Y | Full | Basic support. |
| Or | Y | Y | Full | Basic support. |
| PRelu | Y | Y | Full | Basic support. |
| Pad | Y | Y | Full | Basic support. |
| Pow | Y | Y | Full | Basic support. |
| QLinearConv | Y | Y | Full | Basic support. |
| QLinearMatMul | Y | Y | Full | Basic support. |
| QuantizeLinear | Y | Y | Full | Basic support. |
| RNN | Y | Y | Full | Basic support. |
| Range | Y | Y | Full | Basic support. |
| Reciprocal | Y | Y | Full | Basic support. |
| ReduceL1 | Y | Y | Full | Basic support. |
| ReduceL2 | Y | Y | Full | Basic support. |
| ReduceLogSum | Y | Y | Full | Basic support. |
| ReduceLogSumExp | Y | Y | Full | Basic support. |
| ReduceMax | Y | Y | Full | Basic support. |
| ReduceMean | Y | Y | Full | Basic support. |
| ReduceMin | Y | Y | Full | Basic support. |
| ReduceProd | Y | Y | Full | Basic support. |
| ReduceSum | Y | Y | Full | Basic support. |
| ReduceSumSquare | Y | Y | Full | Basic support. |
| Relu | Y | Y | Full | Basic support. |
| Reshape | Y | Y | Full | Basic support. |
| Resize | Y | Y | Full | Basic support. |
| Round | Y | Y | Full | Basic support. |
| Scatter | Y | Y | Full | Basic support. |
| ScatterElements | Y | Y | Full | Basic support. |
| ScatterND | Y | Y | Full | Basic support. |
| Selu | Y | Y | Full | Basic support. |
| Shape | Y | Y | Full | Basic support. |
| Shrink | Y | Y | Full | Basic support. |
| Sigmoid | Y | Y | Full | Basic support. |
| Sign | Y | Y | Full | Basic support. |
| Sin | Y | Y | Full | Basic support. |
| Sinh | Y | Y | Full | Basic support. |
| Size | Y | Y | Full | Basic support. |
| Slice | Y | Y | Full | Basic support. |
| Softmax | Y | Y | Full | Basic support. |
| SoftmaxCrossEntropyLoss | Y | Y | Full | Basic support. |
| Softplus | Y | Y | Full | Basic support. |
| Softsign | Y | Y | Full | Basic support. |
| SpaceToDepth | Y | Y | Full | Basic support. |
| Split | Y | Y | Full | Basic support. |
| Sqrt | Y | Y | Full | Basic support. |
| Squeeze | Y | Y | Full | Basic support. |
| Sub | Y | Y | Full | Basic support. |
| Sum | Y | Y | Full | Basic support. |
| Tan | Y | Y | Full | Basic support. |
| Tanh | Y | Y | Full | Basic support. |
| ThresholdedRelu | Y | Y | Full | Basic support. |
| Tile | Y | Y | Full | Basic support. |
| TopK | Y | Y | Full | Basic support. |
| Transpose | Y | Y | Full | Basic support. |
| Unique | Y | Y | Full | Basic support. |
| Unsqueeze | Y | Y | Full | Basic support. |
| Upsample | Y | Y | Full | Basic support. |
| Where | Y | Y | Full | Basic support. |
| Xor | Y | Y | Full | Basic support. |

### C. Not Implemented

| Operator | C | Quant(int8/int16) | Level | Notes |
|---|---:|---:|---|---|

## Quant Operator Set (`codegen.py::quant_ops`)

```text
Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, BitShift, Cast, Ceil, Celu, Clip, Compress, Concat, ConcatFromSequence, Constant, ConstantOfShape, Conv, ConvInteger, ConvTranspose, Cos, Cosh, CumSum, DepthToSpace, DequantizeLinear, Det, Div, Dropout, DynamicQuantizeLinear, Einsum, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, GRU, Gather, GatherElements, GatherND, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HardSigmoid, Hardmax, Identity, If, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, Loop, LpNormalization, LpPool, MatMul, MatMulInteger, Max, MaxPool, MaxRoiPool, MaxUnpool, Mean, MeanVarianceNormalization, Min, Mod, Mul, Multinomial, Neg, NegativeLogLikelihoodLoss, NonMaxSuppression, NonZero, Not, OneHot, Or, PRelu, Pad, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, RNN, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, ReverseSequence, RoiAlign, Round, Scan, Scatter, ScatterElements, ScatterND, Selu, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, SoftmaxCrossEntropyLoss, Softplus, Softsign, SpaceToDepth, Split, SplitToSequence, Sqrt, Squeeze, StringNormalizer, Sub, Sum, Tan, Tanh, TfIdfVectorizer, ThresholdedRelu, Tile, TopK, Transpose, Unique, Unsqueeze, Upsample, Where, Xor
```
