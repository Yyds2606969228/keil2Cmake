# ONNX Opset12 Coverage Matrix

- Generated at (UTC): `2026-02-23 16:33:37Z`
- Scope: `domain in ('', 'ai.onnx')` and `since_version <= 12`
- Backend rule: C backend only
- Default CLI: `--weights flash --emit c`
- Quantization: inferred from model graph Q/DQ and tensor dtypes
- Weight storage: keep original ONNX dtype (`float/int8/int16/int32/int64`)

## Summary

| Metric | Count |
|---|---:|
| Opset12 operators | 162 |
| C native support | 162 |
| Quantized coverage on C (`quant_ops` intersect C support) | 162 |

## Matrix

| Operator | C | Quant(int8/int16) | Notes |
|---|---:|---:|---|
| Abs | Y | Y | basic support |
| Acos | Y | Y | basic support |
| Acosh | Y | Y | basic support |
| Add | Y | Y | basic support |
| And | Y | Y | basic support |
| ArgMax | Y | Y | C: supports axis/keepdims/select_last_index |
| ArgMin | Y | Y | C: supports axis/keepdims/select_last_index |
| Asin | Y | Y | basic support |
| Asinh | Y | Y | basic support |
| Atan | Y | Y | basic support |
| Atanh | Y | Y | basic support |
| AveragePool | Y | Y | C: static rank>=3 (N,C,*); supports count_include_pad |
| BatchNormalization | Y | Y | C: static rank>=2 (N,C,*) |
| BitShift | Y | Y | C: integer dtype; broadcast semantics; direction=LEFT/RIGHT |
| Cast | Y | Y | basic support |
| Ceil | Y | Y | basic support |
| Celu | Y | Y | basic support |
| Clip | Y | Y | basic support |
| Compress | Y | Y | C: current subset: condition must be constant 1D; supports axis/no-axis |
| Concat | Y | Y | basic support |
| ConcatFromSequence | Y | Y | C: registered; requires sequence lowering before C codegen |
| Constant | Y | Y | basic support |
| ConstantOfShape | Y | Y | basic support |
| Conv | Y | Y | C: subset: 4D NCHW; N>=1; group>=1 with Cin=group*CperG and Cout%group==0 |
| ConvInteger | Y | Y | basic support |
| ConvTranspose | Y | Y | C: subset: 4D NCHW; currently fp32 and group=1 |
| Cos | Y | Y | basic support |
| Cosh | Y | Y | basic support |
| CumSum | Y | Y | C: axis input must be constant scalar; supports exclusive/reverse={0,1} |
| DepthToSpace | Y | Y | C: subset: 4D NCHW; DCR/CRD; C divisible by blocksize^2 |
| DequantizeLinear | Y | Y | basic support |
| Det | Y | Y | C: current subset: 2D square matrix input (fp32) |
| Div | Y | Y | basic support |
| Dropout | Y | Y | basic support |
| DynamicQuantizeLinear | Y | Y | basic support |
| Einsum | Y | Y | C: current subset: ij,jk->ik / bij,bjk->bik / bij,jk->bik / ij,bjk->bik |
| Elu | Y | Y | basic support |
| Equal | Y | Y | basic support |
| Erf | Y | Y | basic support |
| Exp | Y | Y | basic support |
| Expand | Y | Y | C: shape input must be constant; broadcast rules applied |
| EyeLike | Y | Y | C: current subset: rank=2 |
| Flatten | Y | Y | basic support |
| Floor | Y | Y | basic support |
| GRU | Y | Y | C: current subset: float32, forward direction, num_directions=1, linear_before_reset=0 |
| Gather | Y | Y | C: current subset: constant 1D indices |
| GatherElements | Y | Y | basic support |
| GatherND | Y | Y | C: current subset: batch_dims=0 |
| Gemm | Y | Y | C: subset: transA=0, transB=0, alpha=1, beta=1 |
| GlobalAveragePool | Y | Y | C: static rank>=3 (N,C,*), output [N,C,1,...,1] |
| GlobalLpPool | Y | Y | C: static rank>=3 (N,C,*); p>0 |
| GlobalMaxPool | Y | Y | C: static rank>=3 (N,C,*), output [N,C,1,...,1] |
| Greater | Y | Y | basic support |
| GreaterOrEqual | Y | Y | basic support |
| HardSigmoid | Y | Y | basic support |
| Hardmax | Y | Y | basic support |
| Identity | Y | Y | basic support |
| If | Y | Y | C: registered; requires control-flow lowering before C codegen |
| InstanceNormalization | Y | Y | C: rank>=3; computes on N,C,* (fp32) |
| IsInf | Y | Y | basic support |
| IsNaN | Y | Y | basic support |
| LRN | Y | Y | C: rank>=3; channel window normalization (fp32) |
| LSTM | Y | Y | C: current subset: float32, forward direction, num_directions=1, no peephole |
| LeakyRelu | Y | Y | basic support |
| Less | Y | Y | basic support |
| LessOrEqual | Y | Y | basic support |
| Log | Y | Y | basic support |
| LogSoftmax | Y | Y | basic support |
| Loop | Y | Y | C: registered; requires control-flow lowering before C codegen |
| LpNormalization | Y | Y | C: subset: fp32, static rank>=1 |
| LpPool | Y | Y | C: static rank>=3 (N,C,*); p>0 |
| MatMul | Y | Y | basic support |
| MatMulInteger | Y | Y | C: current subset: 2D; optional zero_point must be constant scalar |
| Max | Y | Y | basic support |
| MaxPool | Y | Y | C: static rank>=3 (N,C,*) |
| MaxRoiPool | Y | Y | C: current subset: float32/int8/int16 NCHW + float32 rois[num_rois,5] |
| MaxUnpool | Y | Y | C: current subset: 4D NCHW unpool with provided indices |
| Mean | Y | Y | basic support |
| MeanVarianceNormalization | Y | Y | C: subset: fp32, 4D NCHW, axes=[0,2,3] |
| Min | Y | Y | basic support |
| Mod | Y | Y | basic support |
| Mul | Y | Y | basic support |
| Multinomial | Y | Y | C: current subset: float32/int8/int16 2D input -> int32/int64 output |
| Neg | Y | Y | basic support |
| NegativeLogLikelihoodLoss | Y | Y | C: current subset: float32/int8/int16 rank-2 input [N,C], rank-1 target [N] |
| NonMaxSuppression | Y | Y | C: current subset: static 3D boxes/scores; output fixed [N,3], invalid rows=-1 |
| NonZero | Y | Y | basic support |
| Not | Y | Y | basic support |
| OneHot | Y | Y | C: depth must be constant scalar; values must be constant length-2 tensor |
| Or | Y | Y | basic support |
| PRelu | Y | Y | basic support |
| Pad | Y | Y | C: subset: mode=constant; static rank>=1 |
| Pow | Y | Y | basic support |
| QLinearConv | Y | Y | basic support |
| QLinearMatMul | Y | Y | C: current subset: 2D; scale/zero_point must be constant scalar |
| QuantizeLinear | Y | Y | basic support |
| RNN | Y | Y | C: current subset: float32, forward direction, num_directions=1 |
| RandomNormal | Y | Y | C: current subset: float32/int8/int16 output |
| RandomNormalLike | Y | Y | C: current subset: float32/int8/int16 output |
| RandomUniform | Y | Y | C: current subset: float32/int8/int16 output |
| RandomUniformLike | Y | Y | C: current subset: float32/int8/int16 output |
| Range | Y | Y | basic support |
| Reciprocal | Y | Y | basic support |
| ReduceL1 | Y | Y | basic support |
| ReduceL2 | Y | Y | basic support |
| ReduceLogSum | Y | Y | basic support |
| ReduceLogSumExp | Y | Y | basic support |
| ReduceMax | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceMean | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceMin | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceProd | Y | Y | basic support |
| ReduceSum | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceSumSquare | Y | Y | basic support |
| Relu | Y | Y | basic support |
| Reshape | Y | Y | basic support |
| Resize | Y | Y | basic support |
| ReverseSequence | Y | Y | C: current subset: static shape; sequence_lens constant |
| RoiAlign | Y | Y | C: current subset: static 4D NCHW + static rois/batch_indices shape; mode=avg/max (fp32) |
| Round | Y | Y | basic support |
| Scan | Y | Y | C: registered; requires control-flow lowering before C codegen |
| Scatter | Y | Y | C: compat subset mapped to ScatterElements semantics; reduction=none only |
| ScatterElements | Y | Y | C: supports reduction=none/add/mul/max/min |
| ScatterND | Y | Y | C: supports reduction=none/add/mul/max/min |
| Selu | Y | Y | basic support |
| SequenceAt | Y | Y | C: registered; requires sequence lowering before C codegen |
| SequenceConstruct | Y | Y | C: registered; requires sequence lowering before C codegen |
| SequenceEmpty | Y | Y | C: registered; requires sequence lowering before C codegen |
| SequenceErase | Y | Y | C: registered; requires sequence lowering before C codegen |
| SequenceInsert | Y | Y | C: registered; requires sequence lowering before C codegen |
| SequenceLength | Y | Y | C: registered; requires sequence lowering before C codegen |
| Shape | Y | Y | basic support |
| Shrink | Y | Y | C: current subset: fp32/int8/int16 |
| Sigmoid | Y | Y | basic support |
| Sign | Y | Y | basic support |
| Sin | Y | Y | basic support |
| Sinh | Y | Y | basic support |
| Size | Y | Y | basic support |
| Slice | Y | Y | C: subset: steps=1; static rank>=1 |
| Softmax | Y | Y | C: supports static rank>=1; axis can be any valid axis |
| SoftmaxCrossEntropyLoss | Y | Y | C: current subset: float32/int8/int16 rank-2 logits [N,C], rank-1 target [N] |
| Softplus | Y | Y | basic support |
| Softsign | Y | Y | basic support |
| SpaceToDepth | Y | Y | C: subset: 4D NCHW; H/W divisible by blocksize |
| Split | Y | Y | basic support |
| SplitToSequence | Y | Y | C: registered; requires sequence lowering before C codegen |
| Sqrt | Y | Y | basic support |
| Squeeze | Y | Y | basic support |
| StringNormalizer | Y | Y | C: current subset: pre-tokenized numeric tensors only |
| Sub | Y | Y | basic support |
| Sum | Y | Y | basic support |
| Tan | Y | Y | basic support |
| Tanh | Y | Y | basic support |
| TfIdfVectorizer | Y | Y | C: current subset: int32/int64 unigram TF/TFIDF |
| ThresholdedRelu | Y | Y | basic support |
| Tile | Y | Y | basic support |
| TopK | Y | Y | basic support |
| Transpose | Y | Y | C: supports static rank>=1; perm must be valid |
| Unique | Y | Y | C: current subset: axis=None (flatten); fixed-capacity outputs |
| Unsqueeze | Y | Y | basic support |
| Upsample | Y | Y | basic support |
| Where | Y | Y | C: broadcast semantics |
| Xor | Y | Y | basic support |

## Quant Operator Set (`codegen.py::quant_ops`)

```text
Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, BitShift, Cast, Ceil, Celu, Clip, Compress, Concat, ConcatFromSequence, Constant, ConstantOfShape, Conv, ConvInteger, ConvTranspose, Cos, Cosh, CumSum, DepthToSpace, DequantizeLinear, Det, Div, Dropout, DynamicQuantizeLinear, Einsum, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, GRU, Gather, GatherElements, GatherND, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HardSigmoid, Hardmax, Identity, If, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, Loop, LpNormalization, LpPool, MatMul, MatMulInteger, Max, MaxPool, MaxRoiPool, MaxUnpool, Mean, MeanVarianceNormalization, Min, Mod, Mul, Multinomial, Neg, NegativeLogLikelihoodLoss, NonMaxSuppression, NonZero, Not, OneHot, Or, PRelu, Pad, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, RNN, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, ReverseSequence, RoiAlign, Round, Scan, Scatter, ScatterElements, ScatterND, Selu, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, SoftmaxCrossEntropyLoss, Softplus, Softsign, SpaceToDepth, Split, SplitToSequence, Sqrt, Squeeze, StringNormalizer, Sub, Sum, Tan, Tanh, TfIdfVectorizer, ThresholdedRelu, Tile, TopK, Transpose, Unique, Unsqueeze, Upsample, Where, Xor
```
