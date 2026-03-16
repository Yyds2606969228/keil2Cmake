# 工件一致性清单

## 1. 目标 (Objective)
确保本轮调试、分析与验证使用的是同一轮构建产物，避免 ELF、BIN、SVD 与配置文件错配。

## 2. 输入 (Input)
- ELF 路径
- BIN / HEX 路径
- SVD 路径
- openocd.cfg 路径
- 构建目录与 preset 信息

## 3. 步骤 (Steps)
1. 检查 ELF 是否来自当前构建目录。
2. 检查下载镜像是否与 ELF 同轮生成。
3. 检查 SVD 是否对应当前 MCU。
4. 检查 `openocd.cfg` 的 interface / target / transport 是否正确。

## 4. 输出 (Output)
- 一致 / 不一致 结论
- 可疑工件列表
- 下一步建议

## 5. 风险 (Risks)
- 使用旧 ELF 会导致根因分析完全失真。
- 使用错误 SVD 会导致寄存器语义解释错误。

## 6. 示例 (Example)
```text
结论：工件不一致
问题：当前 ELF 来自 build-A，当前 openocd.cfg 指向 build-B 产物
建议：重新登记本轮构建产物后再进入分析
```