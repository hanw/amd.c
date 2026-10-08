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
ENGINE_SRC = src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c
SRC = $(ENGINE_SRC) src/main.c
HDR = src/common.h src/gguf.h src/model.h src/backend.h src/gen.h core/ie_core.h

all: $(B)/ie-run $(B)/ie-serve $(B)/ie_kernels.hsaco

$(B):
	mkdir -p $(B)

$(B)/ie-run: $(SRC) $(HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# ie-serve: the OpenAI-compatible HTTP server (SERVE.md)
SERVE_SRC = $(ENGINE_SRC) src/tok.c src/json.c src/serve.c
SERVE_HDR = $(HDR) src/tok.h src/tok_unicode.h src/json.h
$(B)/ie-serve: $(SERVE_SRC) $(SERVE_HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ $(SERVE_SRC) $(LDLIBS) -lpthread

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

$(B)/laws_test: tests/laws_test.cpp laws/ie_laws.cpp core/ie_core.h core/lean_mem.h | $(B)
	$(CXX) -std=c++17 -O2 -o $@ tests/laws_test.cpp

TEST_SRC = src/common.c src/gguf.c src/model.c src/cpu.c
$(B)/kernel_test: tests/kernel_test.c $(TEST_SRC) $(HDR) | $(B)
	$(CC) $(CFLAGS) -o $@ tests/kernel_test.c $(TEST_SRC) $(LDLIBS)

MODELS = $(B)/tiny-qwen2.gguf $(B)/tiny-llama.gguf $(B)/tiny-llama-k.gguf
$(MODELS) &: tools/make_tiny_gguf.py | $(B)
	$(PYTHON) tools/make_tiny_gguf.py $(B)

test: all $(B)/laws_test $(B)/kernel_test $(MODELS)
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
