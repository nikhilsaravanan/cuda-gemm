# Build with the CUDA toolkit (nvcc) on a machine with an NVIDIA GPU.
#
# Set ARCH to your GPU's compute capability:
#   sm_75  - Turing (GTX 16xx, RTX 20xx, T4)
#   sm_80  - Ampere (A100)
#   sm_86  - Ampere (RTX 30xx)
#   sm_89  - Ada Lovelace (RTX 40xx)
#   sm_90  - Hopper (H100)
# Example:  make ARCH=sm_89

NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS := -O3 -arch=$(ARCH) -lineinfo
LDFLAGS   := -lcublas

SRC := src/gemm_naive.cu src/bench.cu
BIN := bench_gemm

all: $(BIN)

$(BIN): $(SRC) src/gemm_kernels.h
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC) $(LDFLAGS)

run: $(BIN)
	./$(BIN) 1024 1024 1024

clean:
	rm -f $(BIN)

.PHONY: all run clean
