# amd-infer

一个从零写的 LLM（大语言模型）解码引擎，专门针对 AMD RDNA4 显卡（gfx1201，例如 RX 9070 XT）。
它用 C 语言写成。它的索引、量化和内存规划逻辑来自一个经过形式化验证的核心（`core/ie_core.h`）。

## 名词

- **GGUF**：llama.cpp 使用的模型文件格式。本引擎读 GGUF v3。
- **Q4_0**：4 位权重量化。每 32 个权重为一块，一块有一个 f16 缩放因子。
- **Q8**：8 位激活量化。每 32 个激活为一块，`d = amax/127`，`q = round(x/d)`。
- **GEMV**：矩阵乘向量。解码时，几乎全部时间都花在 GEMV 上。
- **wave**：RDNA 上 32 个线程一起执行（wave32）。
- **工作组（workgroup）**：256 个线程，也就是 8 个 wave。
- **LDS**：工作组内共享的片上内存。
- **arena**：一块连续内存。一步解码的所有激活缓冲区都放在里面。
- **验证核心**：`core/ie_core.h`。它的定律（`laws/ie_laws.cpp`）在 Lean 4 中被证明，对所有输入成立。
- **hsaco**：AMD GPU 的代码对象文件（ELF）。

## 文件

| 文件 | 内容 |
|---|---|
| `core/ie_core.h` | 验证核心（C 与 C++ 共用一个头文件）。本项目不修改它。 |
| `laws/ie_laws.cpp`、`proofs/` | 定律与 Lean 证明（另一位开发者维护）。 |
| `src/gguf.c/.h` | mmap 并解析 GGUF v3：所有类型的元数据（含数组）、张量信息、对齐、数据偏移。 |
| `src/model.c/.h` | Llama 与 Qwen2 模型；加载时把 Q4_0 重排成 GPU 布局；构造一步解码的算子列表。 |
| `src/plan.c` | 激活 arena 的静态规划（贪心），然后用 `ie_plan_check` 检查。 |
| `src/cpu.c` | CPU 后端。算法和工作划分与 GPU 内核相同。 |
| `src/hip.c` | GPU 后端。运行时用 dlopen 加载 `libamdhip64.so`。 |
| `src/main.c` | 命令行工具 `ie-run`。 |
| `kernels/ie_kernels.c` | GPU 内核，用 C 写（不是 HIP C++），由 clang 编译到 amdgcn。 |
| `tools/make_tiny_gguf.py` | 生成小的随机模型（GGUF v3），两种架构都有。 |
| `tools/ref_forward.py` | 独立的 numpy float64 参考前向计算。 |
| `tools/check_objdump.sh` | 检查 GEMV 内核的机器码。 |
| `tests/kernel_test.c` | CPU GEMV 与逐元素点积对比；重排后的半字节与 GGUF 对比。 |
| `tests/laws_test.cpp` | 在随机输入上运行所有定律（g++）。 |

## 构建

需要：gcc（或 clang）、clang 18 以上（带 amdgcn 目标）、ld.lld、llvm-objdump、python3 + numpy + `gguf`（pip）。
构建时不需要 ROCm。

```sh
make          # build/ie-run（CPU 引擎 + GPU 主机代码）和 build/ie_kernels.hsaco
make test     # 全部测试（见下文）
make proofs   # Lean 证明：proofs/check.sh（需要环境变量 CPP_LEAN、EDG_DIR，以及 lean）
```

内核编译过程：`clang -x c -std=c11 --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3` 先生成 LLVM IR，
再编译成目标文件，最后 `ld.lld -shared` 链接成 hsaco。
clang 18 在 C 语言里不接受 `amdgpu_flat_work_group_size` 属性，
所以 Makefile 在 IR 中加入 `"amdgpu-flat-work-group-size"="256,256"`。
所有内核都用 256 线程的工作组。

## 运行

```sh
build/ie-run MODEL.gguf [--backend cpu|gpu] [--tokens 1,2,3] [--n 32] [--ctx N] \
             [--dump-logits FILE] [--hsaco build/ie_kernels.hsaco] [--info]
```

- `--tokens`：提示词的 token id 列表。这一版没有 BPE 编码，只接受 id。
- `--n`：贪心生成的 token 数。
- `--dump-logits`：每一步写一行 `位置 token logit0 logit1 ...`。
- 输出：所有 token id，以及解码后的文本（gpt2 分词器用字节级解码，例如 Qwen2；llama 分词器处理 `▁` 和 `<0xXX>`）。
- 结束时打印每 token 时间、tokens/s，以及有效带宽（每个 token 读取的权重字节数 ÷ 时间）。

### 在 gfx1201 机器上运行（GPU）

1. 安装 ROCm 运行时，确保可以加载 `libamdhip64.so`（例如 `export LD_LIBRARY_PATH=/opt/rocm/lib`）。
2. 运行 `make`。GPU 主机上没有 clang 时，在另一台机器上运行 `make build/ie_kernels.hsaco`，再把这个文件复制过去。
3. 运行 `build/ie-run model.gguf --backend gpu --tokens 1,2,3 --n 64`。

如果找不到库，程序会退出并给出明确的提示。
GPU 后端只在 gfx1201 上编译。核函数需要 `v_dot4_i32_iu8` 指令和 wave32。

环境变量：

| 变量 | 作用 |
|---|---|
| `IE_PROFILE=1` | 每个核函数后记录一个 GPU 事件，结束时打印按核函数和按层的时间表 |
| `IE_PROFILE_CSV=文件` | 与 `IE_PROFILE=1` 一起用：把每个算子的时间写进 CSV 文件 |
| `IE_ATTN=old` | 用旧的注意力核函数 `ie_attn`（每个头一个工作组） |
| `IE_ATTN=check` | 每层同时运行新旧两个注意力核函数，结束时打印两者输出的最大相对差 |
| `--tokens-file 文件`、`--ppl`、`--ppl-first N`（命令行选项） | 从文件读 token；计算从第 N 个位置开始的困惑度 |
| `IE_NORM_FUSE=1` | 把 RMSNorm 合进前一个带残差的矩阵向量乘（最后完成的工作组做）。在 R9700 上更慢，所以默认关闭 |

硬件追踪：用 `rocprofv3 --kernel-trace` 运行引擎，再用 `tools/rocprof_ops.py` 把追踪结果对应到算子（见脚本开头的说明）。
矩阵向量乘基准测试：`tools/gemv_bench.c` 对不同形状单独计时（见文件开头的说明）。
`tools/gemv_rows_proto.c` 是一个原型（每个 wave 算 R 行），测试结果：R = 2 和 R = 1 一样快，R = 4 和 8 更慢，所以引擎没有用它。

支持的模型：Llama 和 Qwen2 架构。Q4_0 和 Q8_0 矩阵走整数路径（dot4 指令）。F32 和 F16 矩阵在加载时转成 f32。
Q6_K 矩阵在加载时先还原成浮点数，再按 Q8_0 重新量化（近似：Q8_0 每 32 个权重一个缩放，Q6_K 每 16 个一个；
Qwen2.5-1.5B 的输出层上，权重的均方根误差约为权重均方根的 0.5%）。
词嵌入如果是 Q8_0，也转成 f32。
验证核心限制矩阵大小：Q4_0 行数 < 2^18，每行块数 < 512（K ≤ 16352）；Q8_0 每行块数 < 256（K ≤ 8160）。
Q4_0 超出限制时程序退出；Q8_0 超出限制时改走 f32 路径。

## 设计

一步解码 = 一个固定的算子列表。两个后端执行同一个列表，并使用同一组 arena 偏移。
加载时，q、k、v 三个矩阵拼成一个矩阵，gate 和 up 拼成一个矩阵（种类不同时不拼）。
重排布局按行存放，所以拼接只是把行接在后面，索引映射仍然是验证核心的映射。
一个缓冲区可以放几个向量（例如 q、k、v），算子用字节偏移指向其中一个。每层有 9 个算子：

1. RMSNorm，同时写出 Q8 副本。
2. qkv 矩阵向量乘，加上 bias。
3. RoPE + 写 KV 缓存（一个算子）。
4. 注意力，同时写出 Q8 副本（头的大小是 32 的倍数且不超过 256 时）。
5. 输出投影的矩阵向量乘，加上残差。
6. RMSNorm，同时写出 Q8 副本。
7. gate+up 矩阵向量乘。
8. SwiGLU，同时写出 Q8 副本。
9. down 矩阵向量乘，加上残差。

层外还有词嵌入、最后的 RMSNorm、输出层和取最大值。Qwen2.5-0.5B（24 层）每个 token 共 220 个算子。

**Q4_0 和 Q8_0 重排**：加载时，每个矩阵被重排成 GPU 布局：Q4_0 每块 4 个 u32 字，Q8_0 每块 8 个 u32 字，加一个独立的 f16 缩放数组。
重排只用核心函数（`ie_q4_src_*`、`ie_q4_dst_*`、`ie_q8_src_*`、`ie_q8_dst_*`、`ie_pack4`）。

**GEMV**：网格是 `ie_gemv_ngroups(rows)` 个工作组（落在 253 到 256 之间时补到 257 个，见下）；wave v 计算行 `ie_gemv_row(g, v)`；
lane l 计算块 `ie_lane_blk(l, t)`（l, l+32, ...）；块点积是 `ie_q4q8_block` 或 `ie_q8q8_block`（各 8 条 dot4 指令）；
每个 lane 一次发出 2 个块的读取，再计算；最后按 `ie_tree` 的顺序做 wave 归约。
CPU 后端按完全相同的划分和顺序计算，所以两者只差浮点舍入。

已确认的硬件现象：在 R9700 上，矩阵向量乘的工作组数在 253 到 256 之间时，比 252 或 257 个慢 2 到 3 倍
（`tools/gemv_bench.c`：2048 行 × 48 块，256 个工作组 17.0 微秒，257 个 6.5 微秒）。原因未知。
引擎因此在这个区间补到 257 个工作组，多出的工作组没有行，直接结束。

**注意力**（`ie_attn_split`）：工作组 = 查询头 × 位置段（每段 32 个位置）。
每个 wave 对自己的位置做在线 softmax；8 个 wave 在 LDS 中合并；最后完成的工作组（原子计数器）合并各段的结果：
整个工作组并行求各段的最大值和总和，每段的权重只算一次并放在 LDS 中。
（每段 8 或 16 个位置的版本试过：生成 64 个 token 时没有更快，生成 512 个时更慢。）

**RMSNorm**：工作组数 = 维度 ÷ 256。每个工作组都计算完整的平方和（代码相同，所以值相同），
然后每个 wave 只归一化并量化一个 32 元素块。

**wave 内归约**：用 DPP 指令（`v_permlanex16` 和 `row_shl`），不用 LDS 的 `ds_bpermute`。
矩阵向量乘的求和顺序不变（先加 16 之外的 lane，再 8、4、2、1），lane 0 的结果和以前逐位相同。

**取最大值**：128 个工作组各自找最大值，最后完成的工作组合并。

**激活规划**：每个缓冲区有大小和生存期。贪心规划器选最低的 256 字节对齐偏移。
然后引擎填好 `Plan` 结构，运行 `ie_plan_check`；检查失败就退出。

**Qwen3.5 / Qwen3.8（GGUF 架构名 qwen35）**：每 4 层中，前 3 层是线性注意力层，第 4 层是全注意力层。
多 token 预测层（`nextn_predict_layers`）不用于逐个 token 的解码，加载时跳过。

- **线性注意力层**（Gated DeltaNet，门控增量规则）：`attn_qkv`、`attn_gate`（z）、`ssm_beta`、`ssm_alpha` 拼成一个矩阵，
  一次矩阵向量乘。然后一个核函数 `ie_gdn` 做完这一层的其余部分。每个值头一个工作组。步骤：
  1. 4 抽头因果卷积，再 SiLU。卷积状态是 4 个槽的环形缓冲区，第 pos 个输入写到槽 pos % 4。本步只读其他 3 个槽。
  2. q、k 每头做 L2 归一化：x / sqrt(Σx² + eps)。值头 h 用键头 h % 键头数。
  3. beta = sigmoid(b)，衰减 = exp(softplus(a + dt_bias) · A)。
  4. 状态 S（128 × 128，每个值头一个）：S ← 衰减 · S；delta = (v − Sᵀk) · beta；S ← S + k deltaᵀ；o = Sᵀq / sqrt(128)。
  5. 门控 RMSNorm：rmsnorm(o) · 权重 · silu(z)，同时写出 Q8 副本。

  位置 0 时，核函数不读旧状态（当作 0），所以新序列不需要清零状态。
- **全注意力层**：q 投影每头输出 [q | gate]。`ie_qkn_rope_kv` 对 q、k 每头做 RMSNorm，只旋转前 64 维（NEOX 配对），
  写 KV 缓存。注意力输出乘以 sigmoid(gate)（在注意力核函数的最后一步做）。KV 缓存只给全注意力层分配。
- **词嵌入**：Q8_0 词嵌入保持 Q8_0（27B 模型 1.3 GB，转成 f32 会是 5 GB），用 `ie_embed_q8` 取一行。
- **流式加载**：GPU 后端在加载时，每个矩阵重排后立刻复制到显存，然后释放主机内存（`ie_mat_sink`）。
  主机内存只需要放下一层。`IE_STREAM=0` 关闭。
- **验证核心的 Q8 尺寸**：每行最多 2047 块，并且行数 × 块数 < 2^26（27B 的 down 投影每行 544 块）。

## 测试结果

### 本机（`make test`，CPU）

- `laws_test`：所有定律在随机输入上通过。
- `kernel_test`：Q4_0 和 Q8_0 各 9 种形状。CPU GEMV 与逐元素 double 点积的最大相对误差：Q4_0 约 8e-8，Q8_0 约 5e-8。重排后的权重与 GGUF 一致。
- 两个小模型与 numpy 参考比较（模拟 Q8 激活量化）：相对误差最大约 5e-7。纯浮点模式约 2e-2 到 3e-2（Q8 激活量化本身的误差）。贪心 token 都相同。
- objdump：`ie_gemv_q4q8` 和 `ie_gemv_q8q8` 各有 8 条 `v_dot4_i32_iu8`（前者 A 无符号，后者 A 有符号），没有 scratch 内存。

### GPU 主机（Radeon AI PRO R9700，gfx1201，ROCm 7.1 / 7.14 运行时）

提示 20 个 token，正常运行（不计时模式），每个 token 的平均时间：

| 模型 | 生成 64 个 | 生成 512 个 | 生成 2048 个 | 每个 token 读取的权重 |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_0（输出层 Q8_0） | 1.40 毫秒（714 个/秒） | 1.47 毫秒 | 1.76 毫秒 | 346 MB |
| Qwen2.5-1.5B-Instruct Q4_0（输出层 Q6_K → Q8_0） | 2.63 毫秒（380 个/秒） | 2.71 毫秒 | 3.14 毫秒 | 985 MB |

1.5B 生成 64 个 token 时，有效带宽 374 GB/s（标称 640 GB/s 的 58%）。
硬件追踪中，大矩阵的核函数接近标称带宽：gate+up（17920 × 1536）585 GB/s，down（1536 × 8960）540 GB/s，
输出层（151936 × 1536，Q8_0）629 GB/s。剩下的时间主要是注意力、几个小矩阵（qkv、输出投影）和核函数之间的间隔。

与 llama.cpp 对比（同一块卡，同样的 GGUF 文件；llama.cpp b11222，Vulkan 后端，RADV 驱动，`llama-bench -p 0 -n 64,512 -r 5`；
本引擎用 1 个 token 的提示，各跑 5 次）。单位：每秒生成的 token 数。

| 模型 | 测试 | llama.cpp（Vulkan） | 本引擎 | 倍数 |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 生成 64 个 | 465 ± 36 | 716 | 1.54 |
| Qwen2.5-0.5B Q4_0 | 生成 512 个 | 533 ± 28 | 683 | 1.28 |
| Qwen2.5-1.5B Q4_0 | 生成 64 个 | 280 ± 8 | 381 | 1.36 |
| Qwen2.5-1.5B Q4_0 | 生成 512 个 | 328 ± 7 | 371 | 1.13 |

注意：llama.cpp 这里用的是 Vulkan 后端，不是 ROCm（HIP）后端，没有测 ROCm 版本。1.5B 的输出层，llama.cpp 直接读 Q6_K，
本引擎读重新量化后的 Q8_0（字节更多，而且是近似）。

Qwen2.5-3B-Instruct Q4_0（输出层 Q6_K → Q8_0，每个 token 读 1891 MB）：生成 64 个 token 226 个/秒（llama.cpp 196 ± 8），
生成 512 个 219 个/秒（llama.cpp 200.1 ± 0.4）；有效带宽 429 GB/s（标称值的 67%）。

**输出质量（困惑度）**：wikitext-2 测试集，用 llama.cpp 的分词器分词，前 4 段，每段 512 个 token，只给每段后半的 255 个 token 计分
（和 `llama-perplexity -c 512 --chunks 4` 相同）。本引擎用 `ie-run --tokens-file 段.txt --ppl --ppl-first 256` 逐段计算。

| 模型 | llama.cpp（Vulkan） | 本引擎（GPU） | 差别 |
|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 15.062 | 15.125 | +0.42% |
| Qwen2.5-1.5B Q4_0 | 10.144 | 10.146 | +0.02% |
| Qwen2.5-3B Q4_0 | 9.069 | 9.092 | +0.25% |

所以 CPU、GPU 和 numpy 参考之间约 2e-2 的 logits 相对差，没有让输出质量变得比 llama.cpp 差。

- 回答 "What is the capital of France? Answer in one sentence." 时，两个模型都输出 "The capital of France is Paris."。
- 两个小模型：GPU 与 CPU 的 logits 相对误差最大 7e-3，中位数约 1e-7 到 3e-7，贪心 token 相同。
- `IE_ATTN=check`：新旧注意力核函数在 48,456 次调用（位置到 2020）中，输出最大相对差 5.5e-5。
- 已知问题：Qwen2.5-0.5B 上，CPU、GPU 和 numpy 参考两两之间的 logits 相对误差约 2e-2（中位数）。
  注意力输出只差 1e-7 到 5e-5，logits 却差 2e-2，说明模型把微小的舍入差放大了。推测原因是 Q8 激活量化遇到离群值。这个问题还没有解决。

### Qwen3.5 / 3.8（qwen35 架构）

Qwen3.5-0.8B Q8_0（ggml-org/Qwen3.5-0.8B-GGUF，24 层，词嵌入与输出层共用）：

| 项目 | llama.cpp b11222（Vulkan） | 本引擎（GPU） |
|---|---|---|
| 困惑度（wikitext-2，4 × 512） | 15.795 | 15.826（+0.2%） |
| 生成 64 个 token | 233.7 ± 6.6 个/秒 | 415 个/秒 |
| 生成 512 个 token | 305.2 ± 13.4 个/秒 | 415 个/秒 |

- CPU 后端第 0 段困惑度 9.8839，GPU 9.8883。
- 硬件追踪：`ie_gdn` 每次 6.9 微秒（先发出状态读取，与卷积重叠；以前 20 微秒）。
- 聊天模板（`<|im_start|>`、空的 `<think>` 块）正常；`--stop` 在结束符处停止。
- 27B 的矩阵形状上（`gemv_bench`，`Q8=1`），Q8_0 矩阵向量乘达到 589 到 634 GB/s（标称值的 92% 到 99%）。

## 什么被验证、什么被测试、什么没有测试

**被验证（Lean 证明，对所有输入成立，`make proofs` 约 1 分钟）**：`core/ie_core.h` 中的函数满足 `laws/ie_laws.cpp` 的 22 条定律：
GEMV 网格和 lane 划分的覆盖与唯一性；Q4_0 和 Q8_0 重排地址在数组内且是双射；半字节和字节解包符合 GGUF；
Q4×Q8 和 Q8×Q8 的块点积公式等于精确的整数点积；跨 lane 求和与 wave 树归约等于普通求和（模 2^32）；
被 `ie_plan_check` 接受的规划：缓冲区在 arena 内、256 对齐、同时存活的缓冲区不重叠。

**只被测试**：GGUF 解析、模型加载、贪心规划器（它的输出由验证过的检查函数检查）、
两个后端的完整前向计算、所有浮点部分（rmsnorm、rope、softmax、swiglu、合并进核函数的 bias、残差和量化副本）、
注意力和取最大值核函数里的跨工作组同步（原子计数器）。

**没有测试**：长于 4020 个位置的上下文、Qwen2.5-0.5B 和 1.5B 以外的真实模型、头的大小超过 256 的模型（会走旧的注意力核函数）。

## 信任边界

C 代码调用的是同一个验证头文件 `core/ie_core.h`；cpp-lean 把同一个文件翻译成 Lean。
下面这些是被信任的，没有被证明：

1. C 语义与 Lean 模型之间的联系：EDG 前端 + cpfe-lean 翻译是正确的。
2. 编译器：gcc / clang（主机与 amdgcn）正确地编译了这些函数。
3. `v_dot4_i32_iu8` 指令的行为等于模型 `ie_dot4_us`（AMD RDNA4 ISA 文档）。
4. 调用方给出的前提：例如 `ie_sizes_ok` 成立、缓冲区真的有核心假设的长度。引擎在加载时检查大小，但缓冲区分配本身没有被证明。
5. 浮点计算不在验证范围内：只有整数部分（索引、点积、归约顺序、内存规划）被证明。
   GPU 上的浮点 wave 归约使用同样的顺序，但浮点加法不满足结合律，所以“与 CPU 只差舍入”是测试结论，不是证明结论。
