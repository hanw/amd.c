"""
ie_gemm_q4kr（kernels/ie_kernels.c）的分块方式，用 TileLang 表示。

用途：描述现在的分块，不是可以直接替换的实现。
未核实：这段代码没有编译或运行过（工作区没有 AMD 显卡，也没有装 TileLang）。
T.copy 直接写入寄存器片段、T.gemm 接受共享内存的切片、GemmWarpPolicy 的取值，
这三点以 TileLang 当前版本的文档为准。

记号：
  Tn  = 本块提示的 token 数（预填充每块最多 512）
  R   = 权重行数（输出维度）
  K   = 输入维度；nb = K // 32 个量化块；K // 256 个超级块
输出 Y[Tn, R] = X[Tn, K] · W[R, K]ᵀ，W 是 Q4_K，X 是 Q8（每 32 个元素一个缩放系数）。
"""
import tilelang
import tilelang.language as T

# ---- 分块参数：和 ie_kernels.c 里的宏一一对应 ----
BLOCK_R = 128   # GK_R = 8 个波 × 16 行（GK_WI = 1）
BLOCK_T = 64    # GK_T：每个工作组的 token 数（4 个 16×16 输出块，GK_WJ = 4）
BLOCK_K = 128   # GK_KB = 4 个量化块 × 32 个元素：每次循环处理的 K
QB = 32         # Q4_K 量化块：32 个权重共用一个 6 位缩放系数
THREADS = 256   # 8 个波（wave32）


@tilelang.jit
def q4k_gemm_current(Tn: int, R: int, K: int):
    nb = K // QB
    Wq: T.Tensor((R, K // 2), T.uint8)        # 4 位权重：每块 16 字节，低 4 位 = 权重 0..15，高 4 位 = 16..31
    Wsc: T.Tensor((R, nb), T.uint8)           # 每块的 6 位缩放系数 sc
    Wmn: T.Tensor((R, nb), T.uint8)           # 每块的 6 位最小值 m
    Wd: T.Tensor((R, K // 256), T.float16)    # 每个超级块的 d
    Wdmin: T.Tensor((R, K // 256), T.float16) # 每个超级块的 dmin
    Xq: T.Tensor((Tn, K), T.int8)             # Q8 激活值
    Xd: T.Tensor((Tn, nb), T.float32)         # 每块的激活缩放系数 dx
    Xsum: T.Tensor((Tn, nb), T.int32)         # 每块激活值之和 asum（最小值项用）
    Y = T.empty((Tn, R), T.float32)

    # 网格：x = 行块，y = token 块（kernel 里 token 块变化最快：wgid % ntt）
    with T.Kernel(T.ceildiv(R, BLOCK_R), T.ceildiv(Tn, BLOCK_T), threads=THREADS) as (bx, by):
        # 激活值：共享内存（LDS），两份轮换（num_stages = 2）
        # kernel 里每行补 16 字节（行宽 144 字节）以避开存储体冲突；TileLang 自动选布局
        X_s = T.alloc_shared((BLOCK_T, BLOCK_K), T.int8)
        Xd_s = T.alloc_shared((BLOCK_T, BLOCK_K // QB), T.float32)
        Sh_s = T.alloc_shared((BLOCK_T, BLOCK_K // QB), T.float16)    # dx·asum（fp16）
        # 权重：不经过 LDS，每个 lane 直接把自己那一行读进寄存器（kernel 里的 wa / wn）
        Wq_f = T.alloc_fragment((BLOCK_R, BLOCK_K // 2), T.uint8)
        W8_f = T.alloc_fragment((BLOCK_R, QB), T.int8)                # 一个量化块解出的 0..15
        Sw_s = T.alloc_shared((BLOCK_R, BLOCK_K // QB), T.float32)    # d·sc（每行每块）
        Mh_s = T.alloc_shared((BLOCK_R, BLOCK_K // QB), T.float16)    # −dmin·m（fp16）
        # 累加器：每个波 16 行 × 64 token = 4 个 16×16 的 f32 块，放在寄存器
        acc = T.alloc_fragment((BLOCK_T, BLOCK_R), T.float32)
        c_blk = T.alloc_fragment((BLOCK_T, BLOCK_R), T.int32)         # 一个量化块的整数点积

        T.clear(acc)
        for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):   # 下一段在算当前段时读入
            T.copy(Xq[by * BLOCK_T, ko * BLOCK_K], X_s)
            T.copy(Wq[bx * BLOCK_R, ko * (BLOCK_K // 2)], Wq_f)       # 显存 → 寄存器
            for t, k in T.Parallel(BLOCK_T, BLOCK_K // QB):
                b = ko * (BLOCK_K // QB) + k
                Xd_s[t, k] = Xd[by * BLOCK_T + t, b]
                Sh_s[t, k] = T.cast(Xd[by * BLOCK_T + t, b] * T.cast(Xsum[by * BLOCK_T + t, b], T.float32), T.float16)
            for r, k in T.Parallel(BLOCK_R, BLOCK_K // QB):
                b = ko * (BLOCK_K // QB) + k
                s = b // 8                                              # 所在超级块
                Sw_s[r, k] = T.cast(Wd[bx * BLOCK_R + r, s], T.float32) * T.cast(Wsc[bx * BLOCK_R + r, b], T.float32)
                Mh_s[r, k] = T.cast(-T.cast(Wdmin[bx * BLOCK_R + r, s], T.float32) * T.cast(Wmn[bx * BLOCK_R + r, b], T.float32), T.float16)

            # 每个量化块（32 个元素）单独做一次整数矩阵乘法，因为缩放系数每 32 个元素变一次
            for kb in T.serial(BLOCK_K // QB):
                for r, e in T.Parallel(BLOCK_R, QB):                   # 解出 4 位权重：lane 的前后两半分别取低 / 高 4 位
                    byte = Wq_f[r, kb * 16 + (e % 16)]
                    W8_f[r, e] = T.cast(T.if_then_else(e < 16, byte & 0x0F, byte >> 4), T.int8)
                T.clear(c_blk)
                # int8 × int8 → int32：K = 32，kernel 里是 2 条 v_wmma_i32_16x16x16_iu8 × 4 个 token 块
                # 8 个波沿行方向排开，每个波 16 行 × 64 token（FullCol：波只沿 N = 行方向划分）
                T.gemm(X_s[:, kb * QB:(kb + 1) * QB], W8_f, c_blk, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                # 每个输出元素 3 条向量运算：乘缩放、整数转浮点、乘加（这一步是现在的主要开销）
                for t, r in T.Parallel(BLOCK_T, BLOCK_R):
                    acc[t, r] += Sw_s[r, kb] * Xd_s[t, kb] * T.cast(c_blk[t, r], T.float32)

            # 最小值项：每段一次 fp16 小矩阵乘法，K = 4 个块（kernel 里补零到 16）
            # acc[t, r] += Σ_kb (dx·asum)[t, kb] · (−dmin·m)[r, kb]
            T.gemm(Sh_s, Mh_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

        T.copy(acc, Y[by * BLOCK_T, bx * BLOCK_R])                     # kernel 里这里还会加偏置或残差

    return Y


# ---- 这段描述没有表达出来的 kernel 细节 ----
# 1. 整数转浮点用的是加偏移技巧：累加器初值为 0x4B400000（12582912.0f 的位），
#    结果按位当作浮点数再减 12582912，省掉一条转换指令。
# 2. 每个量化块之后有 __builtin_amdgcn_sched_barrier(0)，禁止编译器让相邻块重叠，
#    否则寄存器会溢出。这也是延迟暴露的原因之一。
# 3. 权重的"下一段"读在寄存器里做双缓冲（wa / wn），不经过共享内存。
#    T.Pipelined 默认把数据放进共享内存；要表达"只进寄存器"需要手写或另加注解。
# 4. 资源：LDS 27,648 字节，每个波 182 个向量寄存器，没有溢出。
