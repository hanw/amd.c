import time, torch, tilelang
import tilelang.language as T

M, N, K = 512, 34816, 5120   # tokens x rows x cols (the largest Q4_K matrix)
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
a = torch.randn(M, K, device=dev, dtype=torch.float16)
b = torch.randn(N, K, device=dev, dtype=torch.float16)
ms = 0.0 #
#print(f"torch fp16 matmul: {ms:.3f} ms, {flop/ms/1e9:.1f} TFLOPS")
ai = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
bi = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
try:
    raise Exception("skip")
    ms = bench(lambda: torch._int_mm(ai, bi))
    print(f"torch int8 _int_mm: {ms:.3f} ms, {flop/ms/1e9:.1f} TOPS")
except Exception as ex:
    print("torch int8 _int_mm failed:", str(ex)[:200])

def matmul(M, N, K, bM, bN, bK, stages, th=256):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), B: T.Tensor((N, K), "float16"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=th) as (bx, by):
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bN, bK), "float16")
            Cl = T.alloc_fragment((bM, bN), "float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by * bM, k * bK], As)
                T.copy(B[bx * bN, k * bK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, C[by * bM, bx * bN])
    return main

c = torch.empty(M, N, device=dev, dtype=torch.float32)
ref = (a.float() @ b.float().t())
for (bM, bN, bK, st, th) in [(512,64,32,2,512),(512,32,32,2,256),(512,64,64,2,256),(512,128,32,2,512),(512,64,32,3,256),(512,64,32,1,256),(512,32,64,2,256)]:
    try:
        t0 = time.time()
        k = tilelang.compile(matmul(M, N, K, bM, bN, bK, st, th), target="auto")
        ct = time.time() - t0
        k(a, b, c)
        err = (c - ref).abs().max().item() / ref.abs().max().item()
        ms = bench(lambda: k(a, b, c))
        print(f"tilelang fp16 {bM}x{bN}x{bK} st{st} th{th}: {ms:.3f} ms, {flop/ms/1e9:.1f} TFLOPS, rel err {err:.1e}")
    except Exception as ex:
        print(f"tilelang {bM}x{bN}x{bK} st{st} th{th} failed:", (str(ex).splitlines() or [repr(ex)])[-1][:200])
