#include "gemm_kernels.h"

// Commit 1: the simplest possible GEMM.
// One thread computes one output element:
//   C[row][col] = sum_k A[row][k] * B[k][col]
//
// Why this is slow (the whole point of the later commits):
//  - Each of the M*N threads reads K elements of A and K elements of B
//    from global memory. Total traffic is 2*M*N*K floats for 2*M*N*K flops,
//    i.e. an arithmetic intensity of ~0.25 FLOP/byte. Hopelessly
//    memory-bound: the SMs spend all their time waiting on DRAM.
//  - Reads of B are strided: B[k][col] jumps by N elements every step of k,
//    so a warp's 32 threads touch 32 far-apart addresses. Uncoalesced -
//    each warp issues far more memory transactions than the data needs.
//  - Zero reuse: A[row][k] is read N times (once per output column) and
//    B[k][col] is read M times (once per output row).

__global__ void naive_gemm_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_naive_gemm(const float *A, const float *B, float *C,
                       int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    naive_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
