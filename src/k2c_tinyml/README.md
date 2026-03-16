# K2C TinyML

该子项目用于将 ONNX 模型转换为可在 MCU 侧集成的 C 代码（`.c/.h`）或静态库（`.a`）。

## 快速开始

生成 C 源码：
```bash
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python scripts/K2CTinyML.py onnx --model model.onnx --output ./out --weights flash --emit c
```

生成静态库（需要 `arm-none-eabi-gcc/ar` 可用）：
```bash
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python scripts/K2CTinyML.py onnx --model model.onnx --output ./out --emit lib --toolchain-bin D:/Toolchains/arm-gcc/bin
```

运行测试：
```bash
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python -m unittest discover -s tests -v
```

生成 Opset12 覆盖矩阵：
```bash
uv run --with onnx python scripts/generate_opset12_coverage.py
```

输出文件：`docs/onnx_opset12_coverage_matrix.md`

