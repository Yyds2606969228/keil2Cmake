# ONNX Opset12 Coverage Matrix

- Generated at (UTC): `2026-02-11 09:54:34Z`
- Scope: `domain in ('', 'ai.onnx')` and `since_version <= 12`
- Backend rule: `backend=cmsis-nn` tries CMSIS-NN first, then falls back to C, otherwise errors
- Backend rule: `backend=c` does not fall back
- Default CLI: `--backend c --quant int8 --weights flash --emit c`
- Quantization: `int8/int16` uses Q/DQ graph and `codegen.py::quant_ops`
- Weight storage: keep original ONNX dtype (`float/int8/int16/int32/int64`)

## Summary

| Metric | Count |
|---|---:|
| Opset12 operators | 162 |
| C native support | 136 |
| CMSIS-NN native support | 12 |
| `backend=cmsis-nn` available (with C fallback) | 136 |
| Quantized coverage on C (`quant_ops` ∩ C support) | 136 |

## Matrix

| Operator | C | CMSIS-NN(native) | cmsis-nn backend | Quant(int8/int16) | Notes |
|---|---:|---:|---:|---:|---|
| Abs | Y | N | Y | Y | basic support |
| Acos | Y | N | Y | Y | basic support |
| Acosh | Y | N | Y | Y | basic support |
| Add | Y | Y | Y | Y | CMSIS: int8/int16; same input/output shape; Add also requires aligned scales |
| And | Y | N | Y | Y | basic support |
| ArgMax | Y | N | Y | Y | C: supports axis/keepdims/select_last_index |
| ArgMin | Y | N | Y | Y | C: supports axis/keepdims/select_last_index |
| Asin | Y | N | Y | Y | basic support |
| Asinh | Y | N | Y | Y | basic support |
| Atan | Y | N | Y | Y | basic support |
| Atanh | Y | N | Y | Y | basic support |
| AveragePool | Y | Y | Y | Y | C: static rank>=3 (N,C,*); supports count_include_pad; CMSIS: int8/int16; 4D NCHW, N=1 |
| BatchNormalization | Y | N | Y | Y | C: static rank>=2 (N,C,*) |
| BitShift | Y | N | Y | Y | C: integer dtype; broadcast semantics; direction=LEFT/RIGHT |
| Cast | Y | N | Y | Y | basic support |
| Ceil | Y | N | Y | Y | basic support |
| Celu | Y | N | Y | Y | basic support |
| Clip | Y | N | Y | Y | basic support |
| Compress | Y | N | Y | Y | C: current subset: condition must be constant 1D; supports axis/no-axis |
| Concat | Y | N | Y | Y | basic support |
| ConcatFromSequence | N | N | N | N | not implemented |
| Constant | Y | N | Y | Y | basic support |
| ConstantOfShape | Y | N | Y | Y | basic support |
| Conv | Y | Y | Y | Y | C: subset: 4D NCHW; N>=1; group>=1 with Cin=group*CperG and Cout%group==0; CMSIS: int8, N=1; regular/depthwise subset; generic grouped conv falls back to C |
| ConvInteger | Y | N | Y | Y | basic support |
| ConvTranspose | Y | N | Y | Y | C: subset: 4D NCHW; currently fp32 and group=1 |
| Cos | Y | N | Y | Y | basic support |
| Cosh | Y | N | Y | Y | basic support |
| CumSum | Y | N | Y | Y | C: axis input must be constant scalar; supports exclusive/reverse={0,1} |
| DepthToSpace | Y | N | Y | Y | C: subset: 4D NCHW; DCR/CRD; C divisible by blocksize^2 |
| DequantizeLinear | Y | N | Y | Y | basic support |
| Det | Y | N | Y | Y | C: current subset: 2D square matrix input (fp32) |
| Div | Y | N | Y | Y | basic support |
| Dropout | Y | N | Y | Y | basic support |
| DynamicQuantizeLinear | Y | N | Y | Y | basic support |
| Einsum | Y | N | Y | Y | C: current subset: ij,jk->ik / bij,bjk->bik / bij,jk->bik / ij,bjk->bik |
| Elu | Y | N | Y | Y | basic support |
| Equal | Y | N | Y | Y | basic support |
| Erf | Y | N | Y | Y | basic support |
| Exp | Y | N | Y | Y | basic support |
| Expand | Y | N | Y | Y | C: shape input must be constant; broadcast rules applied |
| EyeLike | Y | N | Y | Y | C: current subset: rank=2 |
| Flatten | Y | N | Y | Y | basic support |
| Floor | Y | N | Y | Y | basic support |
| GRU | N | N | N | N | not implemented |
| Gather | Y | N | Y | Y | C: current subset: constant 1D indices |
| GatherElements | Y | N | Y | Y | basic support |
| GatherND | Y | N | Y | Y | C: current subset: batch_dims=0 |
| Gemm | Y | Y | Y | Y | C: subset: transA=0, transB=0, alpha=1, beta=1; CMSIS: int8; 2D subset equivalent to MatMul |
| GlobalAveragePool | Y | Y | Y | Y | C: static rank>=3 (N,C,*), output [N,C,1,...,1]; CMSIS: int8/int16; 4D NCHW, N=1 |
| GlobalLpPool | Y | N | Y | Y | C: static rank>=3 (N,C,*); p>0 |
| GlobalMaxPool | Y | Y | Y | Y | C: static rank>=3 (N,C,*), output [N,C,1,...,1]; CMSIS: int8/int16; 4D NCHW, N=1 |
| Greater | Y | N | Y | Y | basic support |
| GreaterOrEqual | Y | N | Y | Y | basic support |
| HardSigmoid | Y | N | Y | Y | basic support |
| Hardmax | Y | N | Y | Y | basic support |
| Identity | Y | Y | Y | Y | basic support |
| If | N | N | N | N | not implemented |
| InstanceNormalization | Y | N | Y | Y | C: rank>=3; computes on N,C,* (fp32) |
| IsInf | Y | N | Y | Y | basic support |
| IsNaN | Y | N | Y | Y | basic support |
| LRN | Y | N | Y | Y | C: rank>=3; channel window normalization (fp32) |
| LSTM | N | N | N | N | not implemented |
| LeakyRelu | Y | N | Y | Y | basic support |
| Less | Y | N | Y | Y | basic support |
| LessOrEqual | Y | N | Y | Y | basic support |
| Log | Y | N | Y | Y | basic support |
| LogSoftmax | Y | N | Y | Y | basic support |
| Loop | N | N | N | N | not implemented |
| LpNormalization | Y | N | Y | Y | C: subset: fp32, static rank>=1 |
| LpPool | Y | N | Y | Y | C: static rank>=3 (N,C,*); p>0 |
| MatMul | Y | Y | Y | Y | CMSIS: int8; 2D subset |
| MatMulInteger | Y | N | Y | Y | C: current subset: 2D; optional zero_point must be constant scalar |
| Max | Y | N | Y | Y | basic support |
| MaxPool | Y | Y | Y | Y | C: static rank>=3 (N,C,*); CMSIS: int8/int16; 4D NCHW, N=1 |
| MaxRoiPool | N | N | N | N | not implemented |
| MaxUnpool | N | N | N | N | not implemented |
| Mean | Y | N | Y | Y | basic support |
| MeanVarianceNormalization | Y | N | Y | Y | C: subset: fp32, 4D NCHW, axes=[0,2,3] |
| Min | Y | N | Y | Y | basic support |
| Mod | Y | N | Y | Y | basic support |
| Mul | Y | Y | Y | Y | CMSIS: int8/int16; same input/output shape |
| Multinomial | N | N | N | N | not implemented |
| Neg | Y | N | Y | Y | basic support |
| NegativeLogLikelihoodLoss | N | N | N | N | not implemented |
| NonMaxSuppression | Y | N | Y | Y | C: current subset: static 3D boxes/scores; output fixed [N,3], invalid rows=-1 |
| NonZero | Y | N | Y | Y | basic support |
| Not | Y | N | Y | Y | basic support |
| OneHot | Y | N | Y | Y | C: depth must be constant scalar; values must be constant length-2 tensor |
| Or | Y | N | Y | Y | basic support |
| PRelu | Y | N | Y | Y | basic support |
| Pad | Y | N | Y | Y | C: subset: mode=constant; static rank>=1 |
| Pow | Y | N | Y | Y | basic support |
| QLinearConv | Y | N | Y | Y | basic support |
| QLinearMatMul | Y | N | Y | Y | C: current subset: 2D; scale/zero_point must be constant scalar |
| QuantizeLinear | Y | N | Y | Y | basic support |
| RNN | N | N | N | N | not implemented |
| RandomNormal | N | N | N | N | not implemented |
| RandomNormalLike | N | N | N | N | not implemented |
| RandomUniform | N | N | N | N | not implemented |
| RandomUniformLike | N | N | N | N | not implemented |
| Range | Y | N | Y | Y | basic support |
| Reciprocal | Y | N | Y | Y | basic support |
| ReduceL1 | Y | N | Y | Y | basic support |
| ReduceL2 | Y | N | Y | Y | basic support |
| ReduceLogSum | Y | N | Y | Y | basic support |
| ReduceLogSumExp | Y | N | Y | Y | basic support |
| ReduceMax | Y | N | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceMean | Y | N | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceMin | Y | N | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceProd | Y | N | Y | Y | basic support |
| ReduceSum | Y | N | Y | Y | C: supports axes/keepdims (fp32) |
| ReduceSumSquare | Y | N | Y | Y | basic support |
| Relu | Y | Y | Y | Y | CMSIS: int8/int16; aligned quant params |
| Reshape | Y | Y | Y | Y | basic support |
| Resize | Y | N | Y | Y | basic support |
| ReverseSequence | Y | N | Y | Y | C: current subset: static shape; sequence_lens constant |
| RoiAlign | Y | N | Y | Y | C: current subset: static 4D NCHW + static rois/batch_indices shape; mode=avg/max (fp32) |
| Round | Y | N | Y | Y | basic support |
| Scan | N | N | N | N | not implemented |
| Scatter | Y | N | Y | Y | C: compat subset mapped to ScatterElements semantics; reduction=none only |
| ScatterElements | Y | N | Y | Y | C: supports reduction=none/add/mul/max/min |
| ScatterND | Y | N | Y | Y | C: supports reduction=none/add/mul/max/min |
| Selu | Y | N | Y | Y | basic support |
| SequenceAt | N | N | N | N | not implemented |
| SequenceConstruct | N | N | N | N | not implemented |
| SequenceEmpty | N | N | N | N | not implemented |
| SequenceErase | N | N | N | N | not implemented |
| SequenceInsert | N | N | N | N | not implemented |
| SequenceLength | N | N | N | N | not implemented |
| Shape | Y | N | Y | Y | basic support |
| Shrink | Y | N | Y | Y | C: current subset: fp32/int8/int16 |
| Sigmoid | Y | N | Y | Y | basic support |
| Sign | Y | N | Y | Y | basic support |
| Sin | Y | N | Y | Y | basic support |
| Sinh | Y | N | Y | Y | basic support |
| Size | Y | N | Y | Y | basic support |
| Slice | Y | N | Y | Y | C: subset: steps=1; static rank>=1 |
| Softmax | Y | N | Y | Y | C: supports static rank>=1; axis can be any valid axis |
| SoftmaxCrossEntropyLoss | N | N | N | N | not implemented |
| Softplus | Y | N | Y | Y | basic support |
| Softsign | Y | N | Y | Y | basic support |
| SpaceToDepth | Y | N | Y | Y | C: subset: 4D NCHW; H/W divisible by blocksize |
| Split | Y | N | Y | Y | basic support |
| SplitToSequence | N | N | N | N | not implemented |
| Sqrt | Y | N | Y | Y | basic support |
| Squeeze | Y | N | Y | Y | basic support |
| StringNormalizer | N | N | N | N | not implemented |
| Sub | Y | N | Y | Y | basic support |
| Sum | Y | N | Y | Y | basic support |
| Tan | Y | N | Y | Y | basic support |
| Tanh | Y | N | Y | Y | basic support |
| TfIdfVectorizer | N | N | N | N | not implemented |
| ThresholdedRelu | Y | N | Y | Y | basic support |
| Tile | Y | N | Y | Y | basic support |
| TopK | Y | N | Y | Y | basic support |
| Transpose | Y | N | Y | Y | C: supports static rank>=1; perm must be valid |
| Unique | N | N | N | N | not implemented |
| Unsqueeze | Y | N | Y | Y | basic support |
| Upsample | Y | N | Y | Y | basic support |
| Where | Y | N | Y | Y | C: broadcast semantics; quantized path requires aligned quant params |
| Xor | Y | N | Y | Y | basic support |

## Quant Operator Set (`codegen.py::quant_ops`)

```text
Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, BitShift, Cast, Ceil, Celu, Clip, Compress, Concat, Constant, ConstantOfShape, Conv, ConvInteger, ConvTranspose, Cos, Cosh, CumSum, DepthToSpace, DequantizeLinear, Det, Div, Dropout, DynamicQuantizeLinear, Einsum, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, Gather, GatherElements, GatherND, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HardSigmoid, Hardmax, Identity, InstanceNormalization, IsInf, IsNaN, LRN, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, LpNormalization, LpPool, MatMul, MatMulInteger, Max, MaxPool, Mean, MeanVarianceNormalization, Min, Mod, Mul, Neg, NonMaxSuppression, NonZero, Not, OneHot, Or, PRelu, Pad, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, ReverseSequence, RoiAlign, Round, Scatter, ScatterElements, ScatterND, Selu, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, Softplus, Softsign, SpaceToDepth, Split, Sqrt, Squeeze, Sub, Sum, Tan, Tanh, ThresholdedRelu, Tile, TopK, Transpose, Unsqueeze, Upsample, Where, Xor
```
