# Keil2Cmake 发布说明（2026-02-25）

## 版本范围
- 发布日期：2026-02-25
- 影响模块：CLI、TinyML、一致性校验、打包产物

## 关键变更
1. TinyML 新增严格一致性校验开关：
   - `--no-strict-validation`
   - 默认启用严格模式；仅在需要放宽时显式传入 `--no-strict-validation`。
2. 严格模式下，一致性校验若为 `skipped` 将直接失败（与 `failed` 一样阻断产物）。
3. `emit=lib` 路径改为 `subprocess.run` 执行外部工具，失败信息更可追踪。
4. CLI 按需加载 TinyML 依赖，普通 `--help`/`openocd --help` 不再触发 ONNX Runtime 警告。
5. `Keil2Cmake.spec` 清理无效 `hiddenimports`，与当前代码结构保持一致。
6. 配置模块调整为读写分离：
   - `load_config()` 只读不落盘
   - `save_config()` 专职落盘

## 兼容性说明
1. `onnx` 子命令参数已演进为：
   - 保留：`--model --weights --emit --output`
   - 新增：`--no-strict-validation`（默认严格模式开启）
2. 默认行为已切换为“严格模式开启”；如需放宽可显式传入 `--no-strict-validation`。

## 验证结果
1. 单元测试：`245 passed`
2. 打包：使用 conda 环境 `keil2cmake` 完成 PyInstaller 打包
3. 新 exe 冒烟：
   - `Keil2Cmake.exe --help` 正常
   - `Keil2Cmake.exe openocd --help` 正常
   - `Keil2Cmake.exe onnx --help` 包含 `--no-strict-validation` 开关
   - 最小 ONNX 模型转换成功（含 strict 模式）
