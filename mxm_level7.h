#pragma once

#include "util.h"
#include "mxm_level3.h"

/**
 * Level 7: B resident in VGPRs, A loaded directly from HBM/LDS into VGPRs (transposed),
 *          no LDS used for A or B.
 *
 * Computes C[K²×K] = A^T[K²×K] × B[K×K] using v_mfma_f64_16x16x4f64.
 *
 * Block = 256 threads (4 wavefronts, one per SIMD/Matrix core).
 *
 * --- B register layout ---
 * B [K×K] row-major.  For the MFMA lane mapping: b_k = tid/16 (0..3), b_col = tid%16 (0..15).
 * Each lane pre-loads its K/4 relevant elements into b_reg[NSTEPS] and holds them in VGPRs
 * across all three GEMMs.
 *
 * --- A register layout ---
 * A [K×K²] col-major in source memory (global for GEMM 1, LDS for GEMMs 2&3).
 * Wave w handles rows [w*ROWS_PER_WF .. w*ROWS_PER_WF+ROWS_PER_WF) of A^T.
 * MFMA A-operand lane mapping: a_row = tid/4 (0..15), a_k = tid%4 (0..3).
 * The transposed read is direct — no LDS staging for A.
 *
 * --- LDS layout (single buffer, reused across GEMMs) ---
 * One buffer of K*(K²+1) doubles, laid out in padded col-major:
 *   element [k][i]  →  LDS offset  k*(K²+1) + i
 * The +1 padding ensures consecutive k-values land on different LDS banks.
 *
 * Bank conflict analysis (gfx90a: 32 banks × 4 bytes):
 *   bank_of_double_at_addr = (addr × 2) % 32
 *   Unpadded stride K²=256:  (256×2)%32 = 0  → all k-values alias → 16-way conflict
 *   Padded stride K²+1=257:  (257×2)%32 = 2  → k=0,1,2,3 → banks 0,2,4,6 → conflict-free ✓
 *   Period = 32/gcd(2,32) = 16; K=16 uses k=0..15 → no aliasing.
 *
 * --- Single-buffer reuse ---
 * Within gemm7_pass the full A pre-load into a_reg completes before any write to dst.
 * Combined with __syncthreads() between GEMMs, the same LDS buffer is safely reused:
 *   GEMM 1: global A → buf (LDS, padded col-major)
 *   GEMM 2: buf (LDS) → buf (LDS, in-place: reads complete before writes)
 *   GEMM 3: buf (LDS) → C (global, row-major)
 *
 * LDS size = K*(K²+1)*sizeof(T).  For K=16: 16×257×8 = 32,864 bytes (~32 KB).
 *
 * --- GEMM chain: global memory traffic ---
 * GEMM 1: reads A from global.
 * GEMMs 2&3: no global memory traffic for A.
 * GEMM 3: writes final C to global.
 * B remains in VGPRs throughout.
 *
 * Supported: K multiple of 16, gfx90a / gfx940.  Falls back to L3 otherwise.
 */

namespace mra {

namespace detail {

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))

typedef double mfma_d4 __attribute__((ext_vector_type(4)));

/**
 * One GEMM pass: dst = src^T × B (B already in b_reg[]).
 *
 * SRC_STRIDE   — k-stride for reading src (K² for global, K²+1 for padded LDS)
 * WRITE_ROW_MAJOR — true  → write dst as row-major [K²×K]: dst[i*K + j]
 *                   false → write dst as padded col-major:  dst[j*(K²+1) + i]
 *
 * The row-major path is used only for the final GEMM writing to global C.
 * The padded col-major path is used for all LDS intermediate writes, enabling
 * conflict-free reads in the subsequent GEMM.
 */
template <typename T, int K, int SRC_STRIDE, bool WRITE_ROW_MAJOR>
__device__ __forceinline__ void gemm7_pass(
    const T* __restrict__ src,
    T* __restrict__       dst,
    const double          b_reg[K / 4],
    const int             wave_row_offset,
    const int             a_row,
    const int             a_k,
    const int             d_row_grp,
    const int             d_col)
{
    constexpr int K2           = K * K;
    constexpr int K2_PAD       = K2 + 1;          /* padded k-stride in LDS     */
    constexpr int ROWS_PER_WF  = K2 / 4;
    constexpr int TILES_PER_WF = ROWS_PER_WF / 16;
    constexpr int NSTEPS       = K / 4;

    /* --- Pre-load A partition into registers (transposed, no LDS staging) ---
     * src[k * SRC_STRIDE + i] where k = s*4+a_k, i = wave_row_offset+t*16+a_row.
     * For global src: SRC_STRIDE = K²  (col-major, unpadded).
     * For LDS src:    SRC_STRIDE = K²+1 (padded col-major, conflict-free). */
    double a_reg[TILES_PER_WF][NSTEPS];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t)
        #pragma unroll
        for (int s = 0; s < NSTEPS; ++s)
            a_reg[t][s] = (double)src[(s * 4 + a_k) * SRC_STRIDE
                                      + wave_row_offset + t * 16 + a_row];

    /* --- Issue all MFMAs for all tiles before draining any accumulator ---
     * Separate AGPR sets per tile allow the matrix core to pipeline tiles
     * while the VALU/LDS units write completed tile results. */
    mfma_d4 acc[TILES_PER_WF];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t)
        acc[t] = {0.0, 0.0, 0.0, 0.0};

    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t)
        #pragma unroll
        for (int s = 0; s < NSTEPS; ++s)
            acc[t] = (mfma_d4)__builtin_amdgcn_mfma_f64_16x16x4f64(
                         a_reg[t][s], b_reg[s], (mfma_d4)acc[t], 0, 0, 0);

    /* --- Write results to dst ---
     * WRITE_ROW_MAJOR=true  (GEMM 3 → global C):
     *   dst[i*K + j]        row-major [K²×K], matches format expected by other levels.
     * WRITE_ROW_MAJOR=false (GEMMs 1&2 → LDS buf):
     *   dst[j*(K²+1) + i]   padded col-major, conflict-free for next GEMM's read. */
    #pragma unroll
    for (int t = 0; t < TILES_PER_WF; ++t) {
        const int base_row = wave_row_offset + t * 16 + d_row_grp;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if constexpr (WRITE_ROW_MAJOR)
                dst[(base_row + r) * K + d_col] = (T)acc[t][r];
            else
                dst[d_col * K2_PAD + (base_row + r)] = (T)acc[t][r];
        }
    }
}

/**
 * Three-GEMM chain for level 7.
 *
 * B loaded once into VGPRs, resident through all three GEMMs.
 * Single LDS buffer reused across GEMMs (reads complete before writes within each pass).
 *
 *   GEMM 1: gemm7_pass<SRC_STRIDE=K²,  WRITE_ROW_MAJOR=false> — global → LDS
 *   GEMM 2: gemm7_pass<SRC_STRIDE=K²+1, WRITE_ROW_MAJOR=false> — LDS → LDS (in-place)
 *   GEMM 3: gemm7_pass<SRC_STRIDE=K²+1, WRITE_ROW_MAJOR=true>  — LDS → global
 */
template <typename T, int K>
__device__ void mTxmq_level7_mfma(
    T* __restrict__       c,         /* output [K²×K] row-major, global          */
    const T* __restrict__ a,         /* input  [K×K²] col-major, global          */
    const T* __restrict__ b,         /* B      [K×K]  row-major, global          */
    T*                    buf)       /* LDS scratch: K*(K²+1) doubles            */
{
    constexpr int K2     = K * K;
    constexpr int K2_PAD = K2 + 1;
    constexpr int NSTEPS = K / 4;

    const int tid_block       = (int)threadIdx.x;
    const int warp_id         = tid_block / 64;
    const int tid             = tid_block % 64;

    /* MFMA lane indices — fixed for all three GEMMs. */
    const int a_row           = tid / 4;
    const int a_k             = tid % 4;
    const int b_k             = tid / 16;
    const int b_col           = tid % 16;
    const int d_row_grp       = (tid / 16) * 4;
    const int d_col           = tid % 16;
    const int wave_row_offset = warp_id * (K2 / 4);

    /* -----------------------------------------------------------------------
     * Load B into VGPRs once.
     * Lane tid holds b_reg[s] = B[(s*4 + b_k)*K + b_col] for s=0..NSTEPS-1.
     * ----------------------------------------------------------------------- */
    double b_reg[NSTEPS];
    #pragma unroll
    for (int s = 0; s < NSTEPS; ++s)
        b_reg[s] = (double)b[(s * 4 + b_k) * K + b_col];

    /* -----------------------------------------------------------------------
     * GEMM 1: A (global, unpadded K² stride) → buf (LDS, padded col-major)
     * ----------------------------------------------------------------------- */
    gemm7_pass<T, K, K2, false>(a, buf, b_reg,
                                wave_row_offset, a_row, a_k, d_row_grp, d_col);
    __syncthreads();

    /* -----------------------------------------------------------------------
     * GEMM 2: buf (LDS, padded K²+1 stride) → buf (LDS, in-place)
     * Full A pre-load into a_reg completes before any write, so same buffer
     * is safe to overwrite.
     * ----------------------------------------------------------------------- */
    gemm7_pass<T, K, K2_PAD, false>(buf, buf, b_reg,
                                    wave_row_offset, a_row, a_k, d_row_grp, d_col);
    __syncthreads();

    /* -----------------------------------------------------------------------
     * GEMM 3: buf (LDS, padded K²+1 stride) → c (global, row-major)
     * ----------------------------------------------------------------------- */
    gemm7_pass<T, K, K2_PAD, true>(buf, c, b_reg,
                                   wave_row_offset, a_row, a_k, d_row_grp, d_col);
}

#endif /* AMD MFMA guard */

} // namespace detail


/**
 * Public interface: executes the full three-GEMM transform chain.
 * Dispatches to mTxmq_level7_mfma on gfx90a/gfx940; falls back to L3 elsewhere.
 */
template <typename T, int K>
__device__ void mTxmq_level7_k(
    T* __restrict__       c,
    const T* __restrict__ a,
    const T* __restrict__ b)
{
    extern __shared__ char smem_level7[];
    T* buf = reinterpret_cast<T*>(smem_level7);

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx940__))
    if constexpr (K % 16 == 0) {
        detail::mTxmq_level7_mfma<T, K>(c, a, b, buf);
        return;
    }
#endif
    /* Fallback: L3. Load B into the LDS buffer and run one GEMM at a time. */
    for (int idx = (int)threadIdx.x; idx < K * K; idx += (int)blockDim.x)
        buf[idx] = b[idx];
    __syncthreads();
    detail::mTxmq_level3_impl<T, K, true>(c, a, buf);
    __syncthreads();
}

template <typename T>
inline size_type mTxmq_level7_shmem_size(int K) {
    /* Single padded col-major buffer: K columns of (K²+1) doubles each. */
    return static_cast<size_type>(K * (K * K + 1) * (int)sizeof(T));
}

template <typename T>
constexpr Dim3 mTxmq_level7_blockdim(int /*K*/) {
    return Dim3(256, 1, 1);
}

} // namespace mra
