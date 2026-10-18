# amd-infer: from-scratch LLM decode engine for AMD RDNA4 (gfx1201).
#   make        CPU engine + GPU host (build/ie-run), kernels (build/ie_kernels.hsaco)
#   make test   laws test, tiny GGUF models, CPU vs numpy reference, objdump check
#   make proofs Lean proofs (proofs/check.sh; needs CPP_LEAN, EDG_DIR, lean)

CC      ?= gcc
CXX     ?= g++
CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -D_GNU_SOURCE
LDLIBS  = -lm -ldl
PYTHON  ?= python3

GPU_CC     ?= clang
GPU_LD     ?= ld.lld
OBJDUMP    ?= llvm-objdump
GPU_ARCH   ?= gfx1201
GPU_CFLAGS = -x c -std=c11 -Wall -Wextra --target=amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH) -nogpulib -O3

B = build
ENGINE_SRC = src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c
SRC = $(ENGINE_SRC) src/main.c
HDR = src/common.h src/gguf.h src/model.h src/backend.h src/gen.h core/ie_core.h $(B)/ie_mk_gen.h prof/sm_profiler.h
CFLAGS += -I$(B)

all: $(B)/ie-run $(B)/ie-serve $(B)/smp_demo $(B)/smp_demo.hsaco $(B)/ie_kernels.hsaco $(B)/ie_mk.hsaco $(B)/ie_mk_mtp.hsaco

$(B):
	mkdir -p $(B)

$(B)/ie-run: $(SRC) $(HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# ie-serve: the OpenAI-compatible HTTP server (SERVE.md)
SERVE_SRC = $(ENGINE_SRC) src/tok.c src/json.c src/chat.c src/serve.c
SERVE_HDR = $(HDR) src/tok.h src/tok_unicode.h src/json.h src/chat.h
$(B)/ie-serve: $(SERVE_SRC) $(SERVE_HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ $(SERVE_SRC) $(LDLIBS) -lpthread

# The prompt cache of gen_run (also in make test, on the CPU)
$(B)/cache_check: tools/cache_check.c $(ENGINE_SRC) $(HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ tools/cache_check.c $(ENGINE_SRC) $(LDLIBS)

# The chat template: tools/chat_check.py TEMPLATE.txt build/chat_render (SERVE.md section 6)
$(B)/chat_render: tools/chat_render.c src/chat.c src/json.c src/chat.h src/json.h | $(B)
	$(CC) $(CFLAGS) -o $@ tools/chat_render.c src/chat.c src/json.c

# The tokenizer against llama.cpp's tokenizer tests (needs a llama.cpp checkout):
#   make tokcheck LLAMA_CPP=../llama.cpp
LLAMA_CPP ?= ../llama.cpp
$(B)/tok_check: tools/tok_check.c src/tok.c src/gguf.c src/common.c src/tok.h src/tok_unicode.h | $(B)
	$(CC) $(CFLAGS) -o $@ tools/tok_check.c src/tok.c src/gguf.c src/common.c $(LDLIBS)
tokcheck: $(B)/tok_check
	./$(B)/tok_check $(LLAMA_CPP)/models/ggml-vocab-qwen2.gguf $(LLAMA_CPP)/models/ggml-vocab-qwen2.gguf.inp $(LLAMA_CPP)/models/ggml-vocab-qwen2.gguf.out
	./$(B)/tok_check $(LLAMA_CPP)/models/ggml-vocab-qwen35.gguf $(LLAMA_CPP)/models/ggml-vocab-qwen35.gguf.inp $(LLAMA_CPP)/models/ggml-vocab-qwen35.gguf.out

# GPU kernels: C -> LLVM IR. clang 18 does not accept the attribute
# amdgpu_flat_work_group_size in C, so the IR gets
# "amdgpu-flat-work-group-size"="256,256" (all kernels run with 256 threads).
$(B)/ie_kernels.ll: kernels/ie_kernels.c core/ie_core.h | $(B)
	$(GPU_CC) $(GPU_CFLAGS) -S -emit-llvm -o $@.tmp $<
	sed -E '/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/' $@.tmp > $@
	rm -f $@.tmp

$(B)/ie_kernels.o: $(B)/ie_kernels.ll
	$(GPU_CC) -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH) -O3 -c -o $@ $<

$(B)/ie_kernels.hsaco: $(B)/ie_kernels.o
	$(GPU_LD) -shared -o $@ $<

# The megakernel (kernels/ie_mk.c, IE_MK=1): the kernels of a run as
# functions of one persistent kernel. Two builds: ie_mk.hsaco (decode steps,
# 192 VGPRs, no spills) and ie_mk_mtp.hsaco (verify runs and MTP drafts; the
# multi-token GEMVs with 4 rows per wave: the 8-row ones make it spill).
# At most 192 VGPRs and 16 KB of LDS: two workgroups per CU (128). tools/mk_gen.py writes the table of all these kernels (their
# argument layout) for kernels/ie_mk.c and src/hip.c.
MK_DECODE = ie_embed_q8,ie_embed_q4,ie_embed_f32,ie_rope_kv,ie_gemv_q4q8,ie_rmsnorm,ie_gemv_q4kq8,ie_gemv_q6kq8,ie_gemv_q8q8,ie_gemv_f32,ie_swiglu,ie_gdn1,ie_qkn_rope_kv,ie_attn_split,ie_attn_kv,ie_argmax,ie_copy_rows
MK_MTP = ie_embed_q8,ie_rmsnorm,ie_rmsnorm_t,ie_quant_q8,ie_swiglu,ie_gdn1,ie_qkn_rope_kv,ie_attn_split,ie_attn_split_t,ie_attn_kv,ie_argmax,ie_gemv_q4q8,ie_gemv_q4kq8,ie_gemv_q6kq8,ie_gemv_q8q8,ie_gemv_f32,ie_gemv_f32_t,ie_gemv_q4kq8_tr,ie_gemv_q6kq8_tr,ie_gemv_q8q8_tr,ie_copy_rows
$(B)/ie_mk_gen.h: tools/mk_gen.py kernels/ie_kernels.c Makefile | $(B)
	$(PYTHON) tools/mk_gen.py kernels/ie_kernels.c $@ DECODE=$(MK_DECODE) MTP=$(MK_MTP)
# (clang 21 at -O3 writes "amdgpu-waves-per-eu"="8,16": at most 192 VGPRs,
# so two workgroups per CU; the host checks the count with a launch at load)
$(B)/ie_mk.ll: kernels/ie_mk.c kernels/ie_kernels.c core/ie_core.h $(B)/ie_mk_gen.h prof/sm_profiler.h | $(B)
	$(GPU_CC) $(GPU_CFLAGS) -I$(B) -DMKS_DECODE -S -emit-llvm -o $@.tmp $<
	sed -E -e '/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/' $@.tmp > $@
	rm -f $@.tmp
$(B)/ie_mk_mtp.ll: kernels/ie_mk.c kernels/ie_kernels.c core/ie_core.h $(B)/ie_mk_gen.h prof/sm_profiler.h | $(B)
	$(GPU_CC) $(GPU_CFLAGS) -I$(B) -DMKS_MTP -S -emit-llvm -o $@.tmp $<
	sed -E -e '/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/' $@.tmp > $@
	rm -f $@.tmp
$(B)/ie_mk.o $(B)/ie_mk_mtp.o: $(B)/%.o: $(B)/%.ll
	$(GPU_CC) -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH) -O3 -c -o $@ $<
$(B)/ie_mk.hsaco $(B)/ie_mk_mtp.hsaco: $(B)/%.hsaco: $(B)/%.o
	$(GPU_LD) -shared -o $@ $<

$(B)/laws_test: tests/laws_test.cpp laws/ie_laws.cpp core/ie_core.h core/lean_mem.h | $(B)
	$(CXX) -std=c++17 -O2 -o $@ tests/laws_test.cpp

TEST_SRC = src/common.c src/gguf.c src/model.c src/cpu.c
$(B)/kernel_test: tests/kernel_test.c $(TEST_SRC) $(HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ tests/kernel_test.c $(TEST_SRC) $(LDLIBS)

MODELS = $(B)/tiny-qwen2.gguf $(B)/tiny-llama.gguf $(B)/tiny-llama-k.gguf
$(MODELS) &: tools/make_tiny_gguf.py | $(B)
	$(PYTHON) tools/make_tiny_gguf.py $(B)

test: all $(B)/laws_test $(B)/kernel_test $(B)/cache_check $(MODELS)
	@echo "== laws_test (the laws of laws/ie_laws.cpp on random inputs)"
	./$(B)/laws_test
	@echo "== kernel_test (CPU GEMV vs plain dot products, repack vs GGUF)"
	./$(B)/kernel_test
	@echo "== tiny qwen2: CPU backend vs numpy reference"
	./$(B)/ie-run $(B)/tiny-qwen2.gguf --tokens 5,200,17,42,299,256 --n 10 --dump-logits $(B)/qwen2.logits
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ref_forward.py $(B)/tiny-qwen2.gguf $(B)/qwen2.logits --mode q8 --tol-median 1e-5 --tol 2e-3
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ref_forward.py $(B)/tiny-qwen2.gguf $(B)/qwen2.logits --mode float --tol 5e-2
	@echo "== tiny llama: CPU backend vs numpy reference"
	./$(B)/ie-run $(B)/tiny-llama.gguf --tokens 1,300,77,259,3 --n 10 --dump-logits $(B)/llama.logits
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ref_forward.py $(B)/tiny-llama.gguf $(B)/llama.logits --mode q8 --tol-median 1e-5 --tol 2e-3
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ref_forward.py $(B)/tiny-llama.gguf $(B)/llama.logits --mode float --tol 5e-2
	@echo "== tiny llama-k (Q4_K, Q5_K, Q6_K): CPU backend vs numpy reference"
	./$(B)/ie-run $(B)/tiny-llama-k.gguf --tokens 1,300,77,259,3 --n 10 --dump-logits $(B)/llamak.logits
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ref_forward.py $(B)/tiny-llama-k.gguf $(B)/llamak.logits --mode q8 --tol-median 1e-5 --tol 2e-3
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ref_forward.py $(B)/tiny-llama-k.gguf $(B)/llamak.logits --mode float --tol 5e-2
	@echo "== prompt cache (gen_run): with and without the cache, the same answers (CPU)"
	./$(B)/cache_check $(B)/tiny-qwen2.gguf
	./$(B)/cache_check $(B)/tiny-llama-k.gguf
	@echo "== GPU backend without a GPU: must fail with a clear message"
	! ./$(B)/ie-run $(B)/tiny-qwen2.gguf --backend gpu --hsaco $(B)/ie_kernels.hsaco --n 1 2> $(B)/gpu.err
	cat $(B)/gpu.err
	@echo "== objdump: the GEMV kernel uses v_dot4_i32_iu8 and 128-bit global loads"
	sh tools/check_objdump.sh $(B)/ie_kernels.hsaco
	@echo "== make test: all passed"

proofs:
	sh proofs/check.sh

clean:
	rm -rf $(B) tools/__pycache__

.PHONY: all test proofs clean tokcheck

# prof/: the per-wave event profiler (sm_profiler.h, after leepoly/sm-profiler)
# and its demo: build/smp_demo build/smp_demo.hsaco -> demo_trace.json
$(B)/smp_demo: prof/demo.c prof/sm_profiler.c prof/sm_profiler.h | $(B)
	$(CC) $(CFLAGS) -o $@ prof/demo.c prof/sm_profiler.c -ldl
$(B)/smp_demo.hsaco: prof/demo_kernel.c prof/sm_profiler.h | $(B)
	$(GPU_CC) $(GPU_CFLAGS) -S -emit-llvm -o $(B)/smp_demo.ll.tmp $<
	sed -E '/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="1,1024" "target-cpu"/' $(B)/smp_demo.ll.tmp > $(B)/smp_demo.ll
	$(GPU_CC) -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH) -O3 -c -o $(B)/smp_demo.o $(B)/smp_demo.ll
	$(GPU_LD) -shared -o $@ $(B)/smp_demo.o
	rm -f $(B)/smp_demo.ll.tmp
