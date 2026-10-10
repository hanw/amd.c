# int8 ie_gemm_q4kr vs fp16 ie_gemm_q4k_h / _h256 (our hsaco, via hipModuleLaunchKernel) vs TileLang, same harness
import ctypes, torch, sys
torch.manual_seed(0)
dev = "cuda"
hip = ctypes.CDLL("/opt/rocm/core-7.14/lib/libamdhip64.so")
mod = ctypes.c_void_p()
assert hip.hipModuleLoad(ctypes.byref(mod), b"/tl/ie_kernels.hsaco") == 0
def fn(name):
    f = ctypes.c_void_p(); assert hip.hipModuleGetFunction(ctypes.byref(f), mod, name.encode()) == 0, name; return f
def launch(f, grid, args):
    vals = []
    for kind, v in args:
        vals.append(ctypes.c_void_p(v) if kind == "p" else ctypes.c_uint32(v))
    arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in vals])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    r = hip.hipModuleLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, stream, arr, None)
    assert r == 0, r
def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

for (T, R, K) in [(512, 34816, 5120), (512, 5120, 17408), (512, 10240, 5120)]:
    nb = K // 32
    Wq = torch.randint(0, 256, (R, nb * 16), device=dev, dtype=torch.uint8)
    sc = torch.randint(0, 64, (R, nb), device=dev, dtype=torch.int32); mn = torch.randint(0, 64, (R, nb), device=dev, dtype=torch.int32)
    sm16 = (sc | (mn << 8)).to(torch.int16)
    dd = torch.empty(R, nb // 8, 2, device=dev, dtype=torch.float16)
    dd[..., 0] = (torch.rand(R, nb // 8, device=dev) * 0.002 + 0.0005).half(); dd[..., 1] = (torch.rand(R, nb // 8, device=dev) * 0.002).half()
    qs = torch.cat([sm16.view(torch.uint8).reshape(-1), dd.view(torch.uint8).reshape(-1)]).contiguous()   # u16 sc|mn<<8 [R][nb], then f16 [R][nb/8][2]
    x = torch.randn(T, K, device=dev) * 2
    xb = x.view(T, nb, 32); d = xb.abs().amax(-1) / 127.0
    q = torch.round(xb / d[..., None].clamp_min(1e-30)).clamp(-127, 127).to(torch.int8)
    asum = q.int().sum(-1).to(torch.int32)
    xs = 40 * nb                                             # bytes per token row: 32 nb int8, nb f32 d, nb u32 asum
    xq = torch.zeros(T, xs, device=dev, dtype=torch.uint8)
    xq[:, :32 * nb] = q.view(torch.uint8).reshape(T, -1)
    xq[:, 32 * nb:36 * nb] = d.float().contiguous().view(torch.uint8).reshape(T, -1)
    xq[:, 36 * nb:] = asum.contiguous().view(torch.uint8).reshape(T, -1)
    xdq = (q.float() * d[..., None]).reshape(T, K)
    qv = Wq.view(R, nb, 16).int(); qv = torch.cat([qv & 15, qv >> 4], 2).float()
    W = (dd[..., 0].float().repeat_interleave(8, 1)[..., None] * sc[..., None].float() * qv - (dd[..., 1].float().repeat_interleave(8, 1) * mn.float())[..., None]).reshape(R, K)
    ref = xdq @ W.t()
    y = torch.empty(T, R, device=dev, dtype=torch.float32)
    xh = torch.empty(T, K, device=dev, dtype=torch.float16)
    flop = 2.0 * T * R * K
    f_q4kr, f_h, f_h256, f_cv = fn("ie_gemm_q4kr"), fn("ie_gemm_q4k_h"), fn("ie_gemm_q4k_h256"), fn("ie_q8_f16")
    def run_q4kr():
        launch(f_q4kr, ((R + 127) // 128) * ((T + 63) // 64), [("p", Wq.data_ptr()), ("p", qs.data_ptr()), ("p", xq.data_ptr()), ("p", y.data_ptr()), ("u", R), ("u", nb), ("p", 0), ("p", 0), ("u", T), ("u", xs), ("u", R), ("u", 0)])
    def conv():
        hip.hipModuleLaunchKernel(f_cv, (K + 255) // 256, T, 1, 256, 1, 1, 0, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in [ctypes.c_void_p(xq.data_ptr()), ctypes.c_void_p(xh.data_ptr()), ctypes.c_uint32(nb), ctypes.c_uint32(xs), ctypes.c_uint32(2 * K)]]), None)
    def run_h(f, qt):
        launch(f, (R // 64) * ((T + qt - 1) // qt), [("p", Wq.data_ptr()), ("p", qs.data_ptr()), ("p", xh.data_ptr()), ("p", y.data_ptr()), ("u", R), ("u", nb), ("p", 0), ("p", 0), ("u", T), ("u", K), ("u", R), ("u", 0)])
    print(f"== T {T} rows {R} K {K}")
    for name, f in [("int8 ie_gemm_q4kr", run_q4kr), ("q8->f16 only", conv), ("fp16 ie_gemm_q4k_h (512)", lambda: (conv(), run_h(f_h, 512))), ("fp16 ie_gemm_q4k_h256", lambda: (conv(), run_h(f_h256, 256)))]:
        f(); torch.cuda.synchronize()
        err = (y - ref).abs().max().item() / ref.abs().max().item() if "only" not in name else 0
        ms = bench(f)
        print(f"  {name:28s} {ms:.3f} ms  {flop/ms/1e9:6.1f} TOPS-eq  rel err {err:.1e}", flush=True)
