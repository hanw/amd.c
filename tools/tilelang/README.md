# TileLang 实验（2026-10-10，amd-gpu-host，R9700 / gfx1201）

环境：容器 `stilldeadcode/vllm-radiance:0.9.3`（TileLang 0.1.10、PyTorch 2.11 + ROCm 7.14），运行时设置
`ROCM_PATH=/opt/rocm/core-7.14 LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib`，`tilelang.compile(..., target="auto")`。
实验时先停掉 ie-serve（它占 30.8 GB 显存）。

| 文件 | 内容 |
|---|---|
| `q4k_gemm_current_tiling.py` | `ie_gemm_q4kr` 现在的分块方式（只用于描述；TileLang 0.1.10 在 RDNA 上没有 int8 矩阵指令，不能运行） |
| `gemm_fp16_sweep.py` | PyTorch fp16 / int8 矩阵乘法，以及 TileLang fp16 矩阵乘法的分块扫描 |
| `q4k_dequant_fp16.py` | Q4_K（引擎的数据布局）在 kernel 内解压成 fp16，再用 fp16 矩阵指令；和 PyTorch 参考比较 |

形状 512 × 34816 × 5120（预填充一块 512 个 token，最大的 Q4_K 矩阵）：

| 实现 | 毫秒 | 万亿次/秒 |
|---|---|---|
| PyTorch fp16（hipBLASLt） | 1.269 | 143.8 |
| PyTorch int8（`torch._int_mm`） | 0.859 | 212.5 |
| TileLang fp16，512 × 64 × 32，256 线程 | 1.60 | 114 |
| TileLang fp16，128 × 128 × 32 | 4.08 | 44.7 |
| TileLang Q4_K → fp16，512 × 64 × 32 | 1.827 | 99.9（等效），相对误差 2.2e-4 |
| 引擎 `ie_gemm_q4kr`（int8，`IE_PROFILE=2`） | 约 1.96 | 约 93 |

要点：一个工作组覆盖本块全部 512 个 token 时最快，权重在每块里只从显存读一次。LDS 超过 64 KB 的配置启动失败。
