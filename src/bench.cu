// Benchmark harness: times each kernel in gemm_kernels.h against cuBLAS.
//
// Usage: ./bench_gemm [M N K]   (default 1024 1024 1024)
//
// Reports milliseconds, achieved GFLOPS, and each kernel as a percentage
// of cuBLAS. Also verifies correctness against the cuBLAS result.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gemm_kernels.h"

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t stat = (call);                                       \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)stat);                                   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

static void fill_random(std::vector<float> &v) {
    for (auto &x : v) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

// Time `launch` on a fresh stream; returns average milliseconds per call.
static float time_kernel_ms(const std::function<void(cudaStream_t)> &launch,
                            int iters = 20) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 5; ++i) launch(stream);  // warmup
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) launch(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ms / iters;
}

int main(int argc, char **argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    srand(42);

    printf("GEMM: C(%d x %d) = A(%d x %d) * B(%d x %d), fp32\n",
           M, N, M, K, K, N);
    const double gflops_total = 2.0 * M * N * K / 1e9;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
    fill_random(h_A);
    fill_random(h_B);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * M * K,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(float) * K * N,
                          cudaMemcpyHostToDevice));

    // Reference: cuBLAS. cuBLAS is column-major, our data is row-major, so
    // we use the transpose trick: the bytes of row-major C = A*B are
    // identical to column-major C' = B'*A' where B' = B and A' = A viewed
    // as column-major. Hence: swap A/B, swap M/N, keep OP_N.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    float cublas_ms = time_kernel_ms([&](cudaStream_t s) {
        CUBLAS_CHECK(cublasSetStream(handle, s));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha,
                                 d_B, N,  // KxN row-major -> leading dim N
                                 d_A, K,  // MxK row-major -> leading dim K
                                 &beta,
                                 d_C, N));  // MxN row-major -> leading dim N
    });
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C, sizeof(float) * M * N,
                          cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(handle));

    const double cublas_gflops = gflops_total / (cublas_ms / 1e3);
    printf("\n%-10s %10s %10s %12s\n", "kernel", "ms", "GFLOPS", "% of cuBLAS");
    printf("%-10s %10.3f %10.1f %12s\n", "cublas", cublas_ms, cublas_gflops, "-");

    // Kernels under test. New commits append entries here.
    struct Kernel {
        const char *name;
        void (*launch)(const float *, const float *, float *,
                       int, int, int, cudaStream_t);
    };
    Kernel kernels[] = {
        {"naive", launch_naive_gemm},
    };

    for (const auto &k : kernels) {
        float ms = time_kernel_ms(
            [&](cudaStream_t s) { k.launch(d_A, d_B, d_C, M, N, K, s); });
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float) * M * N,
                              cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (int i = 0; i < M * N; ++i)
            max_err = std::max(max_err, (double)fabs(h_C[i] - h_ref[i]));

        double gflops = gflops_total / (ms / 1e3);
        printf("%-10s %10.3f %10.1f %11.1f%%   (max err vs cublas: %.2e)\n",
               k.name, ms, gflops, 100.0 * cublas_ms / ms, max_err);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
