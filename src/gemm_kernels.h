#pragma once

#include <cuda_runtime.h>

// Every GEMM kernel in this project exposes a launcher with a uniform
// signature so the benchmark harness can time them interchangeably.
//
// Matrices are row-major: A is MxK, B is KxN, C is MxN.

void launch_naive_gemm(const float *A, const float *B, float *C,
                       int M, int N, int K, cudaStream_t stream);
