# transform-cublasdx

Benchmarks for the MRA (Multiresolution Analysis) transform operator — a batched
3D tensor-times-matrix contraction — targeting AMD (HIP/ROCm) and NVIDIA (CUDA)
GPUs.  Five optimization levels are provided so each technique can be studied and
profiled independently.

---

## Problem Description

The MRA transform applies a 1-D transformation matrix along each of the three
dimensions of a K×K×K coefficient tensor.  A single transform is:

```
for each dimension d in {0,1,2}:
    C ← A(T) × C    (in-place, cycling through a workspace)
```

where the core GEMM is:

```
C(i,j) = Σ_k  A(k,i) · B(k,j)

  A  (input/workspace):  K² × K   col-major   A[k,i] = A[k·K² + i]
  B  (transform matrix): K  × K   row-major   B[k,j] = B[k·K  + j]
  C  (output):           K² × K   row-major   C[i,j] = C[i·K  + j]
```

The benchmark runs `nfuncs` independent transforms per kernel launch, repeated
across `ntasks` task submissions, so GPU utilisation across a realistic workload
can be measured.

---

## Problem Sizes

| K  | Tensor (K³) elements | Tensor bytes (FP64) | GEMM A shape | GEMM B shape | FLOPs / transform |
|----|----------------------|---------------------|--------------|--------------|-------------------|
|  6 |  216                 |  1.7 KB             |  36 ×  6     |  6 ×  6      |  3 888            |
|  8 |  512                 |  4.1 KB             |  64 ×  8     |  8 ×  8      | 12 288            |
| 10 | 1 000                |  8.0 KB             | 100 × 10     | 10 × 10      | 30 000            |
| 12 | 1 728                | 13.8 KB             | 144 × 12     | 12 × 12      | 62 208            |
| 16 | 4 096                | 32.8 KB             | 256 × 16     | 16 × 16      | 196 608           |
| 20 | 8 000                | 64.0 KB             | 400 × 20     | 20 × 20      | 480 000           |
| 32 | 32 768               | 262 KB              |1024 × 32     | 32 × 32      | 6 291 456         |

> FLOPs per transform = 3 dimensions × 2 × K² × K × K (multiply-add pairs).

### Memory footprint (default benchmark: N=2048, nblocks=512)

| K  | Input batch (A)  | Transform matrix (B) | Output batch (C) | Workspace    | Total      |
|----|------------------|----------------------|------------------|--------------|------------|
|  6 | 3.5 MB           |   0.3 KB             | 3.5 MB           |  0.9 MB      |  8.0 MB    |
|  8 | 8.4 MB           |   0.5 KB             | 8.4 MB           |  2.1 MB      | 18.9 MB    |
| 16 | 67.1 MB          |   2.0 KB             | 67.1 MB          |  8.4 MB      | 142.6 MB   |
| 32 | 537 MB           |  16.0 KB             | 537 MB           | 67.1 MB      |  1.14 GB   |

---

## Optimization Levels

Select a level at runtime with `-l <N>` (default: L5 on CUDA, L3 on HIP).

### L1 — Global memory, thread-parallel over j  (`mxm.h`)

Each thread owns one output column `j`.  For every row `i`, it streams both
`A[:,i]` and `B[:,j]` from HBM and accumulates `C[i,j]` in a scalar register.
No shared memory is used.

- **Bottleneck**: A is re-loaded K times per row (once per thread owning a
  different column), wasting HBM bandwidth by factor K.
- **Shared memory**: 0 bytes
- **Threads**: 128 × 1 × 1

---

### L2 — B in LDS  (`mxm_level2.h`)

The entire K×K transform matrix B is loaded cooperatively into LDS once per
GEMM call.  All subsequent accesses to B hit L1/LDS instead of HBM.  A is
still streamed from global memory one element at a time.

- **Improvement**: Eliminates repeated HBM loads of B (factor K reduction in
  B traffic).
- **Shared memory**: K² × sizeof(T)  (max 8 KB at K=32)
- **Threads**: 128 × 1 × 1

---

### L3 — Register blocking (`mxm_level3.h`)

Each thread accumulates a full output row `C[i,:]` in a compile-time register
array `acc[K]`.  The k-loop loads `A[k,i]` once and FMAs it against all K
columns of B (already in LDS), eliminating redundant global loads of A.

Because K is a compile-time template parameter, each K value produces a
separate kernel binary with register pressure proportional to K (not `max(K)`).
This is critical on AMD where `__launch_bounds__` enforces a hard VGPR budget.

- **Improvement**: Reduces A HBM traffic by K×; hot loop is fully
  register-resident.
- **Shared memory**: K² × sizeof(T)  (B in LDS)
- **Threads**: 128 × 1 × 1
- **Register cost**: K doubles per thread (e.g. 32 doubles = 64 VGPRs at K=32)

---

### L4 — AMD MFMA intrinsics (`mxm_level4.h`)

On GFX90A / GFX940 (MI250X / MI300X), uses the `v_mfma_f64_16x16x4f64`
instruction via `__builtin_amdgcn_mfma_f64_16x16x4f64`.  One wavefront (64
threads) cooperatively computes a 16×16 output tile with a 4-deep contraction
in a single instruction, achieving near-peak FP64 throughput.

Thread layout for a single 16×16×4 MFMA tile:
```
A fragment (16×4 = 64 elements):  thread t  →  A[t/4][t%4]
B fragment ( 4×16= 64 elements):  thread t  →  B[t/16][t%16]
D output   (16×16 / 4 per thread):thread t  →  D[{(t/16)*4 + 0..3}][t%16]
```

For K values not divisible by 16 (K=6,8,10,12,20) the MFMA path is skipped
and the kernel falls back to L3 register blocking within the same binary.

- **Improvement**: Hardware matrix units; 2× theoretical peak vs scalar FMA.
- **Shared memory**: K² × sizeof(T)
- **Threads**: 64 × 1 × 1 (one wavefront)
- **Supported K with MFMA**: 16, 32  (others fall back to L3)

---

### L5 — cuBLASDx  (`mxm_cublasdx.h`)

NVIDIA-only path using the cuBLASDx device-side BLAS library.  Performs a
double-buffered block GEMM with Tensor Core acceleration, with software
pipelining to overlap LDS reads and HMMA instructions.

- **Improvement**: Tensor Core throughput (FP16/BF16 accumulation, TF32, or
  FP64 on Ampere+); double-buffered pipeline hides LDS latency.
- **Requires**: cuBLASDx headers; only compiled when `MRA_HAVE_CUBLASDX=1`.
- **Threads**: determined by cuBLASDx policy for the given K.

---

## Building

```bash
mkdir build && cd build

# AMD (Frontier / MI250X)
cmake .. -DMRA_HAVE_HIP=1 -DCMAKE_CXX_COMPILER=hipcc
make -j

# NVIDIA (with cuBLASDx)
cmake .. -DMRA_HAVE_CUDA=1 -DMRA_HAVE_CUBLASDX=1
make -j
```

---

## Running

```bash
# Options
./transformbench_hip  -K <int>   # transform order (default 16)
                      -N <int>   # number of functions in batch (default 2048)
                      -M <int>   # max concurrent blocks (default 512)
                      -n <int>   # task submissions per timing rep (default 500)
                      -r <int>   # timing repetitions (default 5)
                      -l <int>   # optimization level 1-5 (default: auto)

# Sweep all levels for a fixed K
for L in 1 2 3 4; do
  ./transformbench_hip -K 16 -N 2048 -n 100 -l $L
done

# Sweep all K values at a fixed level
for K in 6 8 10 12 16 20 32; do
  ./transformbench_hip -K $K -N 2048 -n 100 -l 3
done
```

Output format (one line per timing rep):
```
Transform;level=L3-regblk;nfuncs=2048;nblocks=512;K=16;tasks=100;threads={128,1,1};smem=2048;Time(us)=12345;GFlop=403.0;Gflop/s=32.6
```

---

## File Map

| File                    | Role                                              |
|-------------------------|---------------------------------------------------|
| `mxm.h`                 | L1 reference GEMM (global memory)                 |
| `mxm_level2.h`          | L2 GEMM (B in LDS)                               |
| `mxm_level3.h`          | L3 GEMM (register blocking, K-templated)          |
| `mxm_level4.h`          | L4 GEMM (AMD MFMA + L3 fallback, K-templated)     |
| `mxm_cublasdx.h`        | L5 GEMM (cuBLASDx, NVIDIA only)                  |
| `transform.h`           | L1 3D transform wrapper + kernel                  |
| `transform_level2.h`    | L2 3D transform wrapper + kernel                  |
| `transform_level3.h`    | L3 3D transform wrapper + kernel (K-templated)    |
| `transform_level4.h`    | L4 3D transform wrapper + kernel (K-templated)    |
| `transform_cublasdx.h`  | L5 3D transform wrapper + kernel                  |
| `transformbench.cu`     | Main benchmark driver (`-l` dispatch, timing)     |
| `util.h`                | Platform macros (CUDA/HIP), `CALL_KERNEL`, etc.   |
