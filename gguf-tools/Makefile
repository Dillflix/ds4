CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)
LLAMA_CPP_DIR ?= ../../deepseek-v4-quants/llama.cpp
LLAMA_CPP_BUILD ?= $(LLAMA_CPP_DIR)/build-remote
LLAMA_CPP_BIN ?= $(LLAMA_CPP_BUILD)/bin
LLAMA_CPP_CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 \
	-I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -lggml-blas -lggml-metal \
	-framework Foundation -framework Metal -framework Accelerate
else
NATIVE_CPU_FLAG ?= -march=native
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Quantization decisions sit on rounding boundaries. Disable device FMA
# contraction so the CUDA searches follow the scalar C reference ordering.
QUANT_NVCCFLAGS ?= -O3 -std=c++17 --fmad=false $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -ldl -lpthread
endif

CFLAGS ?= -O3 -Wall -Wextra -Werror=implicit-function-declaration -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean check-source-f16 quality-score quality-llama-score

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

deepseek4-quantize-cuda.o: deepseek4-quantize.c quants.h quants_cuda.h Makefile
	$(CC) $(CFLAGS) $(CPPFLAGS) -DDS4Q_USE_CUDA -c -o $@ deepseek4-quantize.c

quants-cuda-host.o: quants.c quants.h Makefile
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ quants.c

quants_cuda.o: quants_cuda.cu quants_cuda.h quants.h Makefile
	$(NVCC) $(QUANT_NVCCFLAGS) -c -o $@ quants_cuda.cu

deepseek4-quantize-cuda: deepseek4-quantize-cuda.o quants-cuda-host.o quants_cuda.o
	$(NVCC) $(QUANT_NVCCFLAGS) -o $@ $^ -lm

test_quants_cuda.o: test_quants_cuda.c quants.h quants_cuda.h Makefile
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ test_quants_cuda.c

test-quants-cuda: test_quants_cuda.o quants-cuda-host.o quants_cuda.o
	$(NVCC) $(QUANT_NVCCFLAGS) -o $@ $^ -lm

test-source-f16-checked: test_source_f16_checked.c quants.c quants.h Makefile
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ test_source_f16_checked.c quants.c -lm -pthread

check-source-f16: deepseek4-quantize test-source-f16-checked
	./test-source-f16-checked
	python3 test_source_f16_core_dense.py ./deepseek4-quantize

quality-score:
	$(MAKE) -C .. gguf-tools/quality-testing/score_official

quality-llama-score:
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/score_llama quality-testing/score_llama.cpp $(LLAMA_CPP_LDLIBS)

clean:
	rm -f deepseek4-quantize deepseek4-quantize-cuda deepseek4-quantize-cuda.o \
		quants-cuda-host.o quants_cuda.o test-quants-cuda test_quants_cuda.o \
		test-source-f16-checked \
		quality-testing/score_official quality-testing/score_official.o \
		quality-testing/score_llama
