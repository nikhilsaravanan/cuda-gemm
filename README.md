# cuda-gemm

Learning GEMM the hard way: from a naive kernel up to CUTLASS, one commit at a time.

## The plan

1. **Naive GEMM** (this commit) — one thread per output element, plus a benchmark
   harness that times every kernel against cuBLAS and checks correctness.
2. **Shared-memory tiling** — block the K loop, stage A/B tiles in shared memory,
   kill the redundant global reads.
3. **Register tiling + vectorized loads** — more output elements per thread,
   `float4` loads for full memory throughput.
4. **Tensor cores** — warp-level `mma.sync` (or `wgmma` on Hopper). The big leap.
5. **CUTLASS** — the reference implementation. Map its template parameters onto
   every decision made by hand in commits 1-4, then dissect the generated SASS
   with Nsight Compute.

## Build & run

Requires the CUDA toolkit and an NVIDIA GPU.

```bash
make                # builds ./bench_gemm (set ARCH for your GPU, see Makefile)
make run            # runs the 1024^3 case
./bench_gemm 2048 2048 2048   # custom M N K
```

## Reading the numbers (commit 1)

Expect naive to land at a few percent of cuBLAS. That gap *is* the lesson:

- Each thread reads `2*K` floats from global memory to do `2*K` flops —
  arithmetic intensity of ~0.25 FLOP/byte. The kernel is memory-bound before
  it starts.
- Reads of B are strided (`B[k][col]` jumps by N per step), so warps issue
  far more DRAM transactions than the bytes they need (uncoalesced).
- A and B elements are re-read from DRAM over and over instead of being
  reused from faster memory.

Commits 2-4 attack these in order: reuse (shared memory), throughput
(vectorized loads), then raw math horsepower (tensor cores).

## References

- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's GEMM template library
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — the step-by-step walkthrough this project mirrors
- [LonghornSilicon/awesome-ai-accelerators](https://github.com/LonghornSilicon/awesome-ai-accelerators) — the curated map this learning path follows
