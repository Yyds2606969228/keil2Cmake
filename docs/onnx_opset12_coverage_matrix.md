# ONNX Opset12 覆盖矩阵（C 后端）

- English version: `onnx_opset12_coverage_matrix_EN.md`
- 生成时间（UTC）: `2026-02-24 12:30:20Z`
- 统计范围: `domain in ('', 'ai.onnx')` 且 `since_version <= 12`
- 后端规则: 仅统计 C 后端
- 默认命令: `--weights flash --emit c`
- 量化判断: 基于模型 Q/DQ 与张量 dtype 自动推断
- 权重存储: 保持 ONNX 原始 dtype（`float/int8/int16/int32/int64`）

## 快速结论

- C 后端覆盖：`162/162`（100.0%）
- 量化覆盖（按 `quant_ops` 口径）：`162/162`（100.0%）
- 受约束算子：`22` 个；建议优先关注该清单

## 覆盖统计

| 指标 | 数量 | 占比 |
|---|---:|---:|
| Opset12 算子总数 | 162 | 100.0% |
| C 后端已覆盖 | 162 | 100.0% |
| 量化覆盖（`quant_ops ∩ C`） | 162 | 100.0% |
| 完整支持（无额外约束） | 140 | 86.4% |
| 受约束支持 | 22 | 13.6% |
| 未实现 | 0 | 0.0% |

## 等级说明

- `完整支持`：已实现且无额外子集约束
- `受约束`：已实现，但存在输入形状/类型/属性等约束
- `未实现`：当前 C 后端未提供实现

## 受约束算子清单

`ConcatFromSequence`, `If`, `Loop`, `MaxRoiPool`, `Multinomial`, `NonMaxSuppression`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `ReverseSequence`, `RoiAlign`, `Scan`, `SequenceAt`, `SequenceConstruct`, `SequenceEmpty`, `SequenceErase`, `SequenceInsert`, `SequenceLength`, `SplitToSequence`, `StringNormalizer`, `TfIdfVectorizer`

## 按算子家族视图

| 家族 | 总数 | 完整支持 | 受约束 | 未实现 | 占比 |
|---|---:|---:|---:|---:|---:|
| 神经网络核心 | 36 | 36 | 0 | 0 | 22.2% |
| 逐元素与数学 | 36 | 36 | 0 | 0 | 22.2% |
| 归约与索引 | 14 | 14 | 0 | 0 | 8.6% |
| 张量形状与布局 | 34 | 34 | 0 | 0 | 21.0% |
| 逻辑与比较 | 9 | 9 | 0 | 0 | 5.6% |
| 量化与整型 | 8 | 8 | 0 | 0 | 4.9% |
| 循环网络 | 3 | 3 | 0 | 0 | 1.9% |
| 视觉检测 | 3 | 0 | 3 | 0 | 1.9% |
| 随机与采样 | 5 | 0 | 5 | 0 | 3.1% |
| 序列 | 9 | 0 | 9 | 0 | 5.6% |
| 控制流 | 3 | 0 | 3 | 0 | 1.9% |
| 文本 | 2 | 0 | 2 | 0 | 1.2% |

### 家族明细（按等级）

#### 神经网络核心 (36)

- 完整支持: `AveragePool`, `BatchNormalization`, `Celu`, `Conv`, `ConvTranspose`, `DepthToSpace`, `Dropout`, `Elu`, `Gemm`, `GlobalAveragePool`, `GlobalLpPool`, `GlobalMaxPool`, `HardSigmoid`, `Hardmax`, `InstanceNormalization`, `LRN`, `LeakyRelu`, `LogSoftmax`, `LpNormalization`, `LpPool`, `MaxPool`, `MaxUnpool`, `MeanVarianceNormalization`, `NegativeLogLikelihoodLoss`, `PRelu`, `Relu`, `Selu`, `Shrink`, `Sigmoid`, `Softmax`, `SoftmaxCrossEntropyLoss`, `Softplus`, `Softsign`, `SpaceToDepth`, `Tanh`, `ThresholdedRelu`
- 受约束: _无_
- 未实现: _无_

#### 逐元素与数学 (36)

- 完整支持: `Abs`, `Acos`, `Acosh`, `Add`, `Asin`, `Asinh`, `Atan`, `Atanh`, `Ceil`, `Clip`, `Cos`, `Cosh`, `Div`, `Erf`, `Exp`, `Floor`, `IsInf`, `IsNaN`, `Log`, `MatMul`, `Max`, `Mean`, `Min`, `Mod`, `Mul`, `Neg`, `Pow`, `Reciprocal`, `Round`, `Sign`, `Sin`, `Sinh`, `Sqrt`, `Sub`, `Sum`, `Tan`
- 受约束: _无_
- 未实现: _无_

#### 归约与索引 (14)

- 完整支持: `ArgMax`, `ArgMin`, `CumSum`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceMax`, `ReduceMean`, `ReduceMin`, `ReduceProd`, `ReduceSum`, `ReduceSumSquare`, `TopK`
- 受约束: _无_
- 未实现: _无_

#### 张量形状与布局 (34)

- 完整支持: `Cast`, `Compress`, `Concat`, `Constant`, `ConstantOfShape`, `Det`, `Einsum`, `Expand`, `EyeLike`, `Flatten`, `Gather`, `GatherElements`, `GatherND`, `Identity`, `NonZero`, `OneHot`, `Pad`, `Range`, `Reshape`, `Resize`, `Scatter`, `ScatterElements`, `ScatterND`, `Shape`, `Size`, `Slice`, `Split`, `Squeeze`, `Tile`, `Transpose`, `Unique`, `Unsqueeze`, `Upsample`, `Where`
- 受约束: _无_
- 未实现: _无_

#### 逻辑与比较 (9)

- 完整支持: `And`, `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Not`, `Or`, `Xor`
- 受约束: _无_
- 未实现: _无_

#### 量化与整型 (8)

- 完整支持: `BitShift`, `ConvInteger`, `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger`, `QLinearConv`, `QLinearMatMul`, `QuantizeLinear`
- 受约束: _无_
- 未实现: _无_

#### 循环网络 (3)

- 完整支持: `GRU`, `LSTM`, `RNN`
- 受约束: _无_
- 未实现: _无_

#### 视觉检测 (3)

- 完整支持: _无_
- 受约束: `MaxRoiPool`, `NonMaxSuppression`, `RoiAlign`
- 未实现: _无_

#### 随机与采样 (5)

- 完整支持: _无_
- 受约束: `Multinomial`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`
- 未实现: _无_

#### 序列 (9)

- 完整支持: _无_
- 受约束: `ConcatFromSequence`, `ReverseSequence`, `SequenceAt`, `SequenceConstruct`, `SequenceEmpty`, `SequenceErase`, `SequenceInsert`, `SequenceLength`, `SplitToSequence`
- 未实现: _无_

#### 控制流 (3)

- 完整支持: _无_
- 受约束: `If`, `Loop`, `Scan`
- 未实现: _无_

#### 文本 (2)

- 完整支持: _无_
- 受约束: `StringNormalizer`, `TfIdfVectorizer`
- 未实现: _无_


## 矩阵详情

### A. 受约束算子

| 算子 | C | 量化(int8/int16) | 等级 | 说明 |
|---|---:|---:|---|---|
| ConcatFromSequence | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| If | Y | Y | 受约束 | 已注册；C 代码生成前需先完成控制流 lowering。 |
| Loop | Y | Y | 受约束 | 已注册；C 代码生成前需先完成控制流 lowering。 |
| MaxRoiPool | Y | Y | 受约束 | 当前子集：float32/int8/int16 NCHW + float32 rois[num_rois,5]。 |
| Multinomial | Y | Y | 受约束 | 当前子集：float32/int8/int16 的 2D 输入 -> int32/int64 输出。 |
| NonMaxSuppression | Y | Y | 受约束 | 当前子集：静态 3D boxes/scores；输出固定 [N,3]，无效行为 -1。 |
| RandomNormal | Y | Y | 受约束 | 当前子集：float32/int8/int16 输出。 |
| RandomNormalLike | Y | Y | 受约束 | 当前子集：float32/int8/int16 输出。 |
| RandomUniform | Y | Y | 受约束 | 当前子集：float32/int8/int16 输出。 |
| RandomUniformLike | Y | Y | 受约束 | 当前子集：float32/int8/int16 输出。 |
| ReverseSequence | Y | Y | 受约束 | 当前子集：静态 shape；sequence_lens 为常量。 |
| RoiAlign | Y | Y | 受约束 | 当前子集：静态 4D NCHW + 静态 rois/batch_indices；mode=avg/max（fp32）。 |
| Scan | Y | Y | 受约束 | 已注册；C 代码生成前需先完成控制流 lowering。 |
| SequenceAt | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| SequenceConstruct | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| SequenceEmpty | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| SequenceErase | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| SequenceInsert | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| SequenceLength | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| SplitToSequence | Y | Y | 受约束 | 已注册；C 代码生成前需先完成序列 lowering。 |
| StringNormalizer | Y | Y | 受约束 | 当前子集：仅支持预分词后的数值张量。 |
| TfIdfVectorizer | Y | Y | 受约束 | 当前子集：int32/int64 unigram TF/TFIDF。 |

### B. 完整支持算子

| 算子 | C | 量化(int8/int16) | 等级 | 说明 |
|---|---:|---:|---|---|
| Abs | Y | Y | 完整支持 | 基础支持 |
| Acos | Y | Y | 完整支持 | 基础支持 |
| Acosh | Y | Y | 完整支持 | 基础支持 |
| Add | Y | Y | 完整支持 | 基础支持 |
| And | Y | Y | 完整支持 | 基础支持 |
| ArgMax | Y | Y | 完整支持 | 基础支持 |
| ArgMin | Y | Y | 完整支持 | 基础支持 |
| Asin | Y | Y | 完整支持 | 基础支持 |
| Asinh | Y | Y | 完整支持 | 基础支持 |
| Atan | Y | Y | 完整支持 | 基础支持 |
| Atanh | Y | Y | 完整支持 | 基础支持 |
| AveragePool | Y | Y | 完整支持 | 基础支持 |
| BatchNormalization | Y | Y | 完整支持 | 基础支持 |
| BitShift | Y | Y | 完整支持 | 基础支持 |
| Cast | Y | Y | 完整支持 | 基础支持 |
| Ceil | Y | Y | 完整支持 | 基础支持 |
| Celu | Y | Y | 完整支持 | 基础支持 |
| Clip | Y | Y | 完整支持 | 基础支持 |
| Compress | Y | Y | 完整支持 | 基础支持 |
| Concat | Y | Y | 完整支持 | 基础支持 |
| Constant | Y | Y | 完整支持 | 基础支持 |
| ConstantOfShape | Y | Y | 完整支持 | 基础支持 |
| Conv | Y | Y | 完整支持 | 基础支持 |
| ConvInteger | Y | Y | 完整支持 | 基础支持 |
| ConvTranspose | Y | Y | 完整支持 | 基础支持 |
| Cos | Y | Y | 完整支持 | 基础支持 |
| Cosh | Y | Y | 完整支持 | 基础支持 |
| CumSum | Y | Y | 完整支持 | 基础支持 |
| DepthToSpace | Y | Y | 完整支持 | 基础支持 |
| DequantizeLinear | Y | Y | 完整支持 | 基础支持 |
| Det | Y | Y | 完整支持 | 基础支持 |
| Div | Y | Y | 完整支持 | 基础支持 |
| Dropout | Y | Y | 完整支持 | 基础支持 |
| DynamicQuantizeLinear | Y | Y | 完整支持 | 基础支持 |
| Einsum | Y | Y | 完整支持 | 基础支持 |
| Elu | Y | Y | 完整支持 | 基础支持 |
| Equal | Y | Y | 完整支持 | 基础支持 |
| Erf | Y | Y | 完整支持 | 基础支持 |
| Exp | Y | Y | 完整支持 | 基础支持 |
| Expand | Y | Y | 完整支持 | 基础支持 |
| EyeLike | Y | Y | 完整支持 | 基础支持 |
| Flatten | Y | Y | 完整支持 | 基础支持 |
| Floor | Y | Y | 完整支持 | 基础支持 |
| GRU | Y | Y | 完整支持 | 基础支持 |
| Gather | Y | Y | 完整支持 | 基础支持 |
| GatherElements | Y | Y | 完整支持 | 基础支持 |
| GatherND | Y | Y | 完整支持 | 基础支持 |
| Gemm | Y | Y | 完整支持 | 基础支持 |
| GlobalAveragePool | Y | Y | 完整支持 | 基础支持 |
| GlobalLpPool | Y | Y | 完整支持 | 基础支持 |
| GlobalMaxPool | Y | Y | 完整支持 | 基础支持 |
| Greater | Y | Y | 完整支持 | 基础支持 |
| GreaterOrEqual | Y | Y | 完整支持 | 基础支持 |
| HardSigmoid | Y | Y | 完整支持 | 基础支持 |
| Hardmax | Y | Y | 完整支持 | 基础支持 |
| Identity | Y | Y | 完整支持 | 基础支持 |
| InstanceNormalization | Y | Y | 完整支持 | 基础支持 |
| IsInf | Y | Y | 完整支持 | 基础支持 |
| IsNaN | Y | Y | 完整支持 | 基础支持 |
| LRN | Y | Y | 完整支持 | 基础支持 |
| LSTM | Y | Y | 完整支持 | 基础支持 |
| LeakyRelu | Y | Y | 完整支持 | 基础支持 |
| Less | Y | Y | 完整支持 | 基础支持 |
| LessOrEqual | Y | Y | 完整支持 | 基础支持 |
| Log | Y | Y | 完整支持 | 基础支持 |
| LogSoftmax | Y | Y | 完整支持 | 基础支持 |
| LpNormalization | Y | Y | 完整支持 | 基础支持 |
| LpPool | Y | Y | 完整支持 | 基础支持 |
| MatMul | Y | Y | 完整支持 | 基础支持 |
| MatMulInteger | Y | Y | 完整支持 | 基础支持 |
| Max | Y | Y | 完整支持 | 基础支持 |
| MaxPool | Y | Y | 完整支持 | 基础支持 |
| MaxUnpool | Y | Y | 完整支持 | 基础支持 |
| Mean | Y | Y | 完整支持 | 基础支持 |
| MeanVarianceNormalization | Y | Y | 完整支持 | 基础支持 |
| Min | Y | Y | 完整支持 | 基础支持 |
| Mod | Y | Y | 完整支持 | 基础支持 |
| Mul | Y | Y | 完整支持 | 基础支持 |
| Neg | Y | Y | 完整支持 | 基础支持 |
| NegativeLogLikelihoodLoss | Y | Y | 完整支持 | 基础支持 |
| NonZero | Y | Y | 完整支持 | 基础支持 |
| Not | Y | Y | 完整支持 | 基础支持 |
| OneHot | Y | Y | 完整支持 | 基础支持 |
| Or | Y | Y | 完整支持 | 基础支持 |
| PRelu | Y | Y | 完整支持 | 基础支持 |
| Pad | Y | Y | 完整支持 | 基础支持 |
| Pow | Y | Y | 完整支持 | 基础支持 |
| QLinearConv | Y | Y | 完整支持 | 基础支持 |
| QLinearMatMul | Y | Y | 完整支持 | 基础支持 |
| QuantizeLinear | Y | Y | 完整支持 | 基础支持 |
| RNN | Y | Y | 完整支持 | 基础支持 |
| Range | Y | Y | 完整支持 | 基础支持 |
| Reciprocal | Y | Y | 完整支持 | 基础支持 |
| ReduceL1 | Y | Y | 完整支持 | 基础支持 |
| ReduceL2 | Y | Y | 完整支持 | 基础支持 |
| ReduceLogSum | Y | Y | 完整支持 | 基础支持 |
| ReduceLogSumExp | Y | Y | 完整支持 | 基础支持 |
| ReduceMax | Y | Y | 完整支持 | 基础支持 |
| ReduceMean | Y | Y | 完整支持 | 基础支持 |
| ReduceMin | Y | Y | 完整支持 | 基础支持 |
| ReduceProd | Y | Y | 完整支持 | 基础支持 |
| ReduceSum | Y | Y | 完整支持 | 基础支持 |
| ReduceSumSquare | Y | Y | 完整支持 | 基础支持 |
| Relu | Y | Y | 完整支持 | 基础支持 |
| Reshape | Y | Y | 完整支持 | 基础支持 |
| Resize | Y | Y | 完整支持 | 基础支持 |
| Round | Y | Y | 完整支持 | 基础支持 |
| Scatter | Y | Y | 完整支持 | 基础支持 |
| ScatterElements | Y | Y | 完整支持 | 基础支持 |
| ScatterND | Y | Y | 完整支持 | 基础支持 |
| Selu | Y | Y | 完整支持 | 基础支持 |
| Shape | Y | Y | 完整支持 | 基础支持 |
| Shrink | Y | Y | 完整支持 | 基础支持 |
| Sigmoid | Y | Y | 完整支持 | 基础支持 |
| Sign | Y | Y | 完整支持 | 基础支持 |
| Sin | Y | Y | 完整支持 | 基础支持 |
| Sinh | Y | Y | 完整支持 | 基础支持 |
| Size | Y | Y | 完整支持 | 基础支持 |
| Slice | Y | Y | 完整支持 | 基础支持 |
| Softmax | Y | Y | 完整支持 | 基础支持 |
| SoftmaxCrossEntropyLoss | Y | Y | 完整支持 | 基础支持 |
| Softplus | Y | Y | 完整支持 | 基础支持 |
| Softsign | Y | Y | 完整支持 | 基础支持 |
| SpaceToDepth | Y | Y | 完整支持 | 基础支持 |
| Split | Y | Y | 完整支持 | 基础支持 |
| Sqrt | Y | Y | 完整支持 | 基础支持 |
| Squeeze | Y | Y | 完整支持 | 基础支持 |
| Sub | Y | Y | 完整支持 | 基础支持 |
| Sum | Y | Y | 完整支持 | 基础支持 |
| Tan | Y | Y | 完整支持 | 基础支持 |
| Tanh | Y | Y | 完整支持 | 基础支持 |
| ThresholdedRelu | Y | Y | 完整支持 | 基础支持 |
| Tile | Y | Y | 完整支持 | 基础支持 |
| TopK | Y | Y | 完整支持 | 基础支持 |
| Transpose | Y | Y | 完整支持 | 基础支持 |
| Unique | Y | Y | 完整支持 | 基础支持 |
| Unsqueeze | Y | Y | 完整支持 | 基础支持 |
| Upsample | Y | Y | 完整支持 | 基础支持 |
| Where | Y | Y | 完整支持 | 基础支持 |
| Xor | Y | Y | 完整支持 | 基础支持 |

### C. 未实现算子

| 算子 | C | 量化(int8/int16) | 等级 | 说明 |
|---|---:|---:|---|---|

## 量化算子集合（`codegen.py::quant_ops`）

```text
Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, BitShift, Cast, Ceil, Celu, Clip, Compress, Concat, ConcatFromSequence, Constant, ConstantOfShape, Conv, ConvInteger, ConvTranspose, Cos, Cosh, CumSum, DepthToSpace, DequantizeLinear, Det, Div, Dropout, DynamicQuantizeLinear, Einsum, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, GRU, Gather, GatherElements, GatherND, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HardSigmoid, Hardmax, Identity, If, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, Loop, LpNormalization, LpPool, MatMul, MatMulInteger, Max, MaxPool, MaxRoiPool, MaxUnpool, Mean, MeanVarianceNormalization, Min, Mod, Mul, Multinomial, Neg, NegativeLogLikelihoodLoss, NonMaxSuppression, NonZero, Not, OneHot, Or, PRelu, Pad, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, RNN, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, ReverseSequence, RoiAlign, Round, Scan, Scatter, ScatterElements, ScatterND, Selu, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, SoftmaxCrossEntropyLoss, Softplus, Softsign, SpaceToDepth, Split, SplitToSequence, Sqrt, Squeeze, StringNormalizer, Sub, Sum, Tan, Tanh, TfIdfVectorizer, ThresholdedRelu, Tile, TopK, Transpose, Unique, Unsqueeze, Upsample, Where, Xor
```
