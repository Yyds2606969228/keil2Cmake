# TinyML框架设计方案（评估与补充）

## 参考开源项目（定位与取舍）
1. TensorFlow Lite Micro（TFLM）
2. onnx2c
3. NNoM
4. CMSIS-NN
5. ESP-NN

## 评估结论
1. 采用 ONNX 作为输入格式更通用，利于跨框架导出与互通（见参考资料 1）。
2. MCU 推理需要严格控制内存与依赖，TFLM 的单一 arena 思路可复用（见参考资料 2）。
3. 纯 C 后端作为基线是必要的；CMSIS-NN/ESP-NN 作为可选加速后端更稳妥（见参考资料 3、5）。

## 关键缺口与风险
1. 缺少算子覆盖检查与降级策略，可能出现“导出成功但无法落地”的问题。
2. 缺少运行时 ABI 与内存规划接口，难以工程化集成。
3. 缺少量化策略与后端门禁条件，性能与内存不可控。

## 目标与范围
1. 输入：ONNX 模型。
2. 输出：可链接的 C 源码或静态库 + 头文件 + 元数据。
3. 运行时：静态内存、无动态分配，适配 MCU。
4. 后端：C（默认）、CMSIS-NN（可选）、ESP-NN（可选）。

## 更新后的工作流程
1. 训练并导出 `model.onnx`。
2. 执行命令（建议新 CLI）：
   `keil2cmake onnx --model <onnx> --backend <c|cmsis-nn|esp-nn> --quant <int8|int16|fp32> --weights <flash|ram> --emit <lib|c>`
3. 生成 `model_name.h` 与 `model_name.c` 或 `libmodel_name.a`，并输出 `model.manifest.json`。
4. 集成产物并初始化运行时上下文。
5. 调用 `k2c_invoke()` 执行推理。

## 架构设计（最小可行分层）
1. `converter/`：ONNX 解析与校验，生成内部 IR。
2. `codegen/`：IR -> C 源码/静态库。
3. `runtime/`：统一 ABI、内存管理、上下文管理。
4. `kernels/`：纯 C 算子实现（基线后端）。
5. `backends/`：CMSIS-NN/ESP-NN 适配层。

## 运行时 ABI（建议）
```c
typedef struct k2c_ctx k2c_ctx_t;
typedef struct k2c_io_desc k2c_io_desc_t;
typedef struct k2c_model k2c_model_t;

const k2c_model_t* k2c_get_model(void);
int k2c_prepare(k2c_ctx_t* ctx, void* arena, size_t arena_bytes);
int k2c_invoke(k2c_ctx_t* ctx, const void* in, void* out);
const k2c_io_desc_t* k2c_get_input_desc(size_t* n);
const k2c_io_desc_t* k2c_get_output_desc(size_t* n);
```

## 内存与权重策略
1. 统一使用单一 arena，由用户提供内存块。
2. 编译期完成内存规划，运行期无动态分配。
3. 权重默认放在 flash（`const`），`--weights=ram` 时在 `prepare` 阶段拷贝至 RAM。

## 量化策略
1. 默认 int8（优先匹配 CMSIS-NN/ESP-NN 加速）。
2. int16/FP32 仅在资源允许时启用。

## 后端策略
1. `c`：纯 C 基线后端，稳定可用、零依赖。
2. `cmsis-nn`：仅当 Cortex-M 且 int8/int16 时启用。
3. `esp-nn`：仅当 ESP 平台时启用。

## 算子覆盖与降级策略
1. 转换阶段输出算子清单与支持情况。
2. 若算子不支持：拒绝并给出清单，或降级到纯 C（可配置）。
3. 输出 `model.manifest.json`（包含 opset、算子列表、arena 估算、后端命中情况）。

## 产物清单
1. `model_name.h`
2. `model_name.c` 或 `libmodel_name.a`
3. `model.manifest.json`

## 命名与目录
1. 项目命名：`onnx-for-mcu`
2. 存放位置：与 `src/` 并列的顶层目录

## 参考资料
1. ONNX 开放标准说明：https://github.com/onnx/onnx
2. TFLM 内存管理说明：https://android.googlesource.com/platform/external/tensorflow/+/632ff3f6169ef18a6947c53bd6f3cb5bf7fc26a6/tensorflow/lite/micro/docs/memory_management.md
3. CMSIS-NN 文档：https://arm-software.github.io/CMSIS_6/main/NN/index.html
4. onnx2c 项目主页：https://github.com/kraiskil/onnx2c
5. ESP-NN 项目主页：https://github.com/espressif/esp-nn
