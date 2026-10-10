import time, torch, tilelang
import tilelang.language as T
torch.manual_seed(0)
M, N, K = 512, 34816, 5120
nb = K // 32
dev = "cuda"
def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n
flop = 2.0 * M * N * K

# Q4_K in the engine layout: nibbles (N, nb*16) u8; sc | mn << 8 (N, nb) u16; [d, dmin] per super-block (N, nb/8*2) f16
Wq = torch.randint(0, 256, (N, nb * 16), device=dev, dtype=torch.uint8)
sc = torch.randint(0, 64, (N, nb), device=dev, dtype=torch.int32)
mn = torch.randint(0, 64, (N, nb), device=dev, dtype=torch.int32)
Wsm = (sc | (mn << 8)).to(torch.int16)  # u16 bits in an int16 tensor
dd = torch.empty(N, nb // 8, 2, device=dev, dtype=torch.float16)
dd[..., 0] = (torch.rand(N, nb // 8, device=dev) * 0.002 + 0.0005).half()
dd[..., 1] = (torch.rand(N, nb // 8, device=dev) * 0.002).half()
Wdd = dd.reshape(N, nb // 8 * 2).contiguous()
A = (torch.randn(M, K, device=dev) * 2).half()

def deq_ref():
    q = Wq.view(N, nb, 16).int()
    lo, hi = q & 15, q >> 4
    qv = torch.cat([lo, hi], dim=2).float()               # (N, nb, 32)
    d = dd[..., 0].float().repeat_interleave(8, dim=1)     # (N, nb)
    dm = dd[..., 1].float().repeat_interleave(8, dim=1)
    w = d[..., None] * sc[..., None].float() * qv - (dm * mn.float())[..., None]
    return w.reshape(N, K)
W = deq_ref()
ref = A.float() @ W.t()

def q4k_fp16(bM, bN, bK, st, th):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), Wq: T.Tensor((N, nb * 16), "uint8"), Wsm: T.Tensor((N, nb), "int16"),
             Wdd: T.Tensor((N, nb // 8 * 2), "float16"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=th) as (bx, by):
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bN, bK), "float16")
            Cl = T.alloc_fragment((bM, bN), "float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=st):
                T.copy(A[by * bM, k * bK], As)
                for n, e in T.Parallel(bN, bK):
                    r = bx * bN + n
                    b = k * (bK // 32) + e // 32
                    ee = e % 32
                    byte = T.cast(Wq[r, b * 16 + ee % 16], "int32")
                    q = T.if_then_else(ee < 16, byte & 15, byte >> 4)
                    sm = T.cast(Wsm[r, b], "int32") & 0xFFFF
                    d = T.cast(Wdd[r, (b // 8) * 2], "float32")
                    dm = T.cast(Wdd[r, (b // 8) * 2 + 1], "float32")
                    Bs[n, e] = T.cast(d * T.cast(sm & 63, "float32") * T.cast(q, "float32") - dm * T.cast(sm >> 8, "float32"), "float16")
                T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, C[by * bM, bx * bN])
    return main

C = torch.empty(M, N, device=dev, dtype=torch.float32)
for cfg in [(512, 64, 32, 2, 256), (512, 64, 64, 2, 256), (256, 64, 32, 2, 256), (512, 32, 32, 2, 256), (256, 128, 32, 2, 512)]:
    try:
        k = tilelang.compile(q4k_fp16(*cfg), target="auto")
        k(A, Wq, Wsm, Wdd, C)
        torch.cuda.synchronize()
        err = (C - ref).abs().max().item() / ref.abs().max().item()
        ms = bench(lambda: k(A, Wq, Wsm, Wdd, C))
        print(f"q4k->fp16 {cfg}: {ms:.3f} ms, {flop/ms/1e9:.1f} TFLOPS eq, rel err {err:.1e}", flush=True)
    except Exception as ex:
        print(f"q4k->fp16 {cfg} failed:", (str(ex).splitlines() or [repr(ex)])[-1][:200], flush=True)
