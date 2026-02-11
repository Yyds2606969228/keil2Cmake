# TinyML框架设计方案（评估与补充）

## 本次基线更新（2026-02-11）
1. 暂不考虑 ESP-NN 集成，当前范围聚焦 `C` 与 `CMSIS-NN`。
2. 运行时默认量化模式为 `int8`（CLI 默认 `--quant int8`）。
3. 参数/权重存储保持 ONNX 模型原始 dtype（`float32/int8/int16/int32/int64`），不做统一重编码。
4. 后端策略固定为：
   - `backend=cmsis-nn`：优先 CMSIS-NN；不支持或约束不满足时回退 C；若 C 也不支持则报错。
   - `backend=c`：不回退；C 不支持则直接报错。
5. Conv 的 `group` 限制已放开：C 后端支持 `group>=1`（需满足 `Cin=group*CperG` 与 `Cout%group==0`）；CMSIS-NN 仍为常规/深度可分子集，普通 grouped conv 自动回退 C。

## 当前完成度（实现态）
1. ONNX Opset12 覆盖矩阵已更新到 `docs/onnx_opset12_coverage_matrix.md`。
2. 当前统计（见覆盖矩阵）：
   - Opset12 算子总数：`162`
   - C 后端原生支持：`136`
   - CMSIS-NN 原生支持：`12`
   - `cmsis-nn` 后端可用（含回退 C）：`136`
   - C 后端量化覆盖（opset12 交集）：`136`
3. 覆盖矩阵已同步反映：
   - Conv 的 C 端 grouped 能力。
   - CMSIS-NN Conv 子集与回退行为。
   - 默认参数与后端回退规则。
   - 新增 `GatherND/ConvTranspose/ScatterElements/ScatterND/Squeeze/Unsqueeze/ReduceProd/ReduceL1/ReduceL2/ReduceSumSquare/BitShift/Compress/MatMulInteger/QLinearMatMul/ConvInteger/QLinearConv/LpNormalization/MeanVarianceNormalization/LpPool/GlobalLpPool/Det/ReverseSequence/NonMaxSuppression/RoiAlign` 的 C 端与量化覆盖状态。

## 已落地架构分层
1. `src/keil2cmake/tinyml/converter/`
   - ONNX 解析、shape 推断、IR 构建。
2. `src/keil2cmake/tinyml/backends/c/ops/`
   - C 基线算子实现，按“一个算子一个文件”组织。
3. `src/keil2cmake/tinyml/backends/cmsis_nn/ops/`
   - CMSIS-NN 算子适配，按“一个算子一个文件”组织。
4. `src/keil2cmake/tinyml/runtime/`
   - 一致性校验与参考执行路径（用于转换后校验）。
5. `src/keil2cmake/tinyml/codegen.py`
   - 后端选择、回退策略、代码生成、manifest 记录。
6. `src/keil2cmake/tinyml/project.py`
   - 项目产物编排，输出 `h/c/manifest` 或 `lib`。

## 运行时与数据策略
1. 输入输出：当前按单输入单输出模型路径生成。
2. 内存策略：以 arena 风格进行静态内存规划，不依赖运行期动态分配。
3. 权重策略：
   - `--weights flash`：常量形式驻留；
   - `--weights ram`：运行期准备阶段拷贝到 RAM。
4. 量化策略：
   - `int8/int16` 依赖 ONNX Q/DQ 图；
   - 默认优先 `int8`。

## 后端能力说明（当前实现）
1. C 后端：
   - `fp32/int8/int16` 主路径可用；
   - Conv 支持 `N>=1` 与 `group>=1`（满足通道约束）。
2. CMSIS-NN 原生覆盖（按当前实现）：
   - `Conv`（`int8`，`N=1`，常规/深度可分子集）
   - `MatMul`/`Gemm`（`int8`，2D 子集）
   - `Add`/`Mul`/`Relu`（`int8/int16`）
   - `MaxPool`/`AveragePool`/`GlobalAveragePool`/`GlobalMaxPool`（`int8/int16`）
   - `Identity`/`Reshape`（轻量直通）
3. CMSIS-NN 不支持或约束不满足时，自动回退 C（仅在 `backend=cmsis-nn` 时）。

## 工程命令与产物
1. 命令：
   - `keil2cmake onnx --model <model.onnx> --backend <c|cmsis-nn> --quant <fp32|int8|int16> --weights <flash|ram> --emit <c|lib> --output <dir>`
2. 产物：
   - `<model>.h`
   - `<model>.c` 或 `lib<model>.a`
   - `model.manifest.json`

## 与目标的差距（下一阶段）
1. 多输入/多输出模型的生成路径与 ABI 收敛。
2. CMSIS-NN 高价值算子继续扩展（在保持可回退前提下）。
3. 更多 Opset12 算子补齐（优先 ResNet/MobileNet 常见链路高频算子）。
4. 覆盖矩阵自动校验流程与 CI 联动（防止能力漂移）。

## 参考资料
1. ONNX：https://github.com/onnx/onnx
2. TFLM 内存管理：
   https://android.googlesource.com/platform/external/tensorflow/+/632ff3f6169ef18a6947c53bd6f3cb5bf7fc26a6/tensorflow/lite/micro/docs/memory_management.md
3. CMSIS-NN：
   https://arm-software.github.io/CMSIS_6/main/NN/index.html
4. onnx2c：
   https://github.com/kraiskil/onnx2c

## 增量更新（2026-02-11，算子补齐）
1. `Pad`：C 后端已从 `rank<=4` 放宽为“静态任意 rank（rank>=1）”，仍限定 `mode=constant`。
2. `Slice`：C 后端已从 `rank<=4` 放宽为“静态任意 rank（rank>=1）”，仍限定 `steps=1`。
3. 维度策略：保持“按 ONNX 解析出的静态输入/输出维度生成 C 代码”，不再将 `Pad/Slice` 固定到低维模板。
4. 测试补充：新增 5D `Pad/Slice` 用例，防止后续改动回退到低维限制实现。
5. 新增 C 后端算子：`CumSum`（axis 常量标量，支持 exclusive/reverse）、`Shrink`（float32/int8/int16）、`EyeLike`（当前 rank=2）。
6. 新增对应单测：`CumSum`、`Shrink`、`EyeLike`，并通过全量 `tests.test_tinyml` 回归。

## 增量更新（2026-02-11，继续补齐）
1. `BitShift`：新增 C 后端实现，支持整数 dtype、广播以及 `direction=LEFT/RIGHT`。
2. `Compress`：新增 C 后端实现，当前要求 `condition` 为常量 1D，支持 `axis`/无 `axis` 子集。
3. 新增 `BitShift/Compress` 的 shape 推断与单测，覆盖矩阵已同步刷新。
4. `MatMulInteger`：新增 C 后端实现，当前支持 2D 子集；可选 `zero_point` 要求为常量标量，输出 `int32/int64`。
5. `QLinearMatMul`：新增 C 后端实现，当前支持 2D 子集；`scale/zero_point` 要求为常量标量，支持 `int8/int16` 量化路径。
6. 新增 `MatMulInteger/QLinearMatMul` 的 shape 推断、校验与单测，覆盖矩阵已同步刷新。
7. `ScatterElements/ScatterND`：补齐 `reduction=none/add/mul/max/min` 语义；`Scatter` 仍按兼容子集映射到 `ScatterElements` 路径，且仅支持 `reduction=none`。
8. 测试补充：新增 `Scatter` 与 `ScatterElements` 的差异边界回归（`Scatter` 禁止 reduction），并新增 `int8/int16` 量化图下 `ScatterElements/ScatterND + reduction` 回归。
9. 新增 C 后端算子：`LpNormalization`、`MeanVarianceNormalization`、`LpPool`、`GlobalLpPool`（当前为 float32 子集实现）。
10. 新增 C 后端算子：`Det`（当前 2D 方阵 float32 子集）、`ReverseSequence`（当前静态 shape + 常量 sequence_lens 子集）；并新增 int8 量化图 `ReverseSequence` 回归。
11. 新增 C 后端量化算子：`ConvInteger` 与 `QLinearConv`，并补齐对应 shape 推断、量化参数注入（QLinear 系列）与单测；覆盖矩阵已同步刷新。
12. 新增 C 后端算子：`NonMaxSuppression`（当前静态 3D boxes/scores、常量阈值输入、固定输出 [N,3] 子集），并补齐 shape 推断、validator 语义与单测；覆盖矩阵已同步刷新。
13. 新增 C 后端算子：`RoiAlign`（当前静态 4D NCHW + 静态 rois/batch_indices shape，`mode=avg/max` 的 float32 子集），并补齐 shape 推断、validator 语义与单测；覆盖矩阵已同步刷新。

## 增量更新（2026-02-11，算子继续扩展）
1. 新增 C 后端 `Einsum` 子集实现：`ij,jk->ik`、`bij,bjk->bik`、`bij,jk->bik`、`ij,bjk->bik`。
2. 补齐 `Einsum` 的 shape 推断与 validator 语义（使用 `numpy.einsum` 参考路径）。
3. unsupported 基线更新为 `RandomUniformLike`，已不再与 `Einsum` 冲突。
4. 覆盖矩阵 `docs/onnx_opset12_coverage_matrix.md` 已刷新：
   - C 后端原生支持：`136`
   - C 后端量化覆盖（opset12 交集）：`136`
5. `NonMaxSuppression` 文档约束已更新：阈值输入支持常量或运行时输入，输出固定 `[N,3]`，无效项填 `-1`。

## 增量更新（2026-02-11，维度自动匹配增强）
1. `GlobalAveragePool/GlobalMaxPool/GlobalLpPool`：从固定 4D 放宽为 `rank>=3`，按 ONNX 解析出的输入 rank 自动生成输出 `[N,C,1,...,1]` 与对应循环。
2. `BatchNormalization`：从固定 4D 放宽为 `rank>=2`，按 `N,C,*` 自动计算 `inner` 并生成代码（float/quant 路径一致）。
3. `validator` 与 `shape inference` 已同步：支持上述 ND 形态，避免“生成通过但一致性校验失败”。
4. 新增 ND 回归用例（含量化）并通过全量回归：
   - `BatchNormalization` ND
   - `GlobalAveragePool` ND（fp32/int8）
   - `GlobalMaxPool` ND（fp32/int8）
   - `GlobalLpPool` ND（fp32）

## 增量更新（2026-02-11，池化 ND 扩展与文档同步）
1. `AveragePool/MaxPool/LpPool`：C 后端从固定 4D 放宽为 `rank>=3`（`N,C,*`），按 ONNX 解析出的真实输入/输出维度生成循环与索引。
2. 量化路径同步：`AveragePool/MaxPool` 的 `int8/int16` 路径支持 ND 形态（仍要求静态 shape）；`LpPool` 维持 float 子集实现。
3. `shape inference` 与 `validator` 已同步到 ND 规则，避免“代码可生成但运行时校验失败”的分歧。
4. 测试补充并通过：
   - `AveragePool` ND（fp32/int8）
   - `MaxPool` ND（fp32/int8）
   - `LpPool` ND（fp32）
5. 覆盖矩阵重生成：`docs/onnx_opset12_coverage_matrix.md` 已反映 `AveragePool/MaxPool/LpPool` 的 C 端 ND 约束，以及 CMSIS-NN 端仍为 `4D NCHW, N=1` 的限制。
