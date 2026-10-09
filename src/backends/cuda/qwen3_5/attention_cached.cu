// Request-local suffix attention. The full-buffer prefix API remains in attention.cu.
#include "common.cuh"
#include "mma.cuh"
#include "ops.h"

namespace cs1 {
namespace {
namespace flash {

constexpr int D = 256, BM = 64, BN = 32, THREADS = 128;
constexpr int LDS = D + 8;  // shared row stride in elements: 528 bytes keeps ldmatrix conflict-free
constexpr int SMEM_BYTES = (BM + 2 * BN) * LDS * 2;

template <bool Gated>
__global__ void __launch_bounds__(THREADS)
    flash_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int ldv,
                 const bf16* __restrict__ gate, bf16* __restrict__ out, int Tq, int Tk, int Hq, int Hk,
                 float scale_log2) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* qs = reinterpret_cast<bf16*>(smem);
    bf16* ks = qs + BM * LDS;
    bf16* vs = ks + BN * LDS;
    const int h = blockIdx.y, hk = h / (Hq / Hk);
    const int q0 = (gridDim.x - 1 - blockIdx.x) * BM;  // the longest blocks first
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, t = lane % 4;
    const int row0 = q0 + warp * 16;  // this warp's first query
    const int offset = Tk - Tq;       // the position of query 0

    for (int c = tid; c < BM * (D / 8); c += THREADS) {
        const int r = c / (D / 8), col = (c % (D / 8)) * 8, row = q0 + r;
        cp_async16(qs + r * LDS + col, q + ((size_t)min(row, Tq - 1) * Hq + h) * D + col, row < Tq);
    }
    cp_async_commit();

    float o[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    const int kv_end = min(Tk, offset + q0 + BM);
    for (int k0 = 0; k0 < kv_end; k0 += BN) {
        for (int c = tid; c < BN * (D / 8); c += THREADS) {
            const int r = c / (D / 8), col = (c % (D / 8)) * 8, s = k0 + r;
            cp_async16(ks + r * LDS + col, k + ((size_t)min(s, Tk - 1) * Hk + hk) * D + col, s < Tk);
        }
        cp_async_commit();
        for (int c = tid; c < BN * (D / 8); c += THREADS) {
            const int r = c / (D / 8), col = (c % (D / 8)) * 8, s = k0 + r;
            cp_async16(vs + r * LDS + col, v + (size_t)min(s, Tk - 1) * ldv + (size_t)hk * D + col, s < Tk);
        }
        cp_async_commit();
        cp_async_wait<1>();  // Q and K
        __syncthreads();

        // keys past every query of this warp contribute nothing, and rows past the
        // last query are never stored
        const bool active = row0 < Tq && k0 <= offset + row0 + 15;
        float sc[BN / 8][4];
#pragma unroll
        for (int n = 0; n < BN / 8; n++) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
        if (active) {
#pragma unroll
            for (int kk = 0; kk < D; kk += 16) {
                uint32_t a[4];
                load_a(a, qs, LDS, warp * 16, kk, lane);
#pragma unroll
                for (int n = 0; n < BN / 8; n += 2) {
                    uint32_t b[4];
                    load_b_nk(b, ks, LDS, kk, n * 8, lane);
                    mma16816(sc[n], a, b[0], b[1]);
                    mma16816(sc[n + 1], a, b[2], b[3]);
                }
            }
        }
        uint32_t p[BN / 16][4];
        if (active) {
            // causal and length mask, then the online softmax in base 2
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = k0 + n * 8 + 2 * t + (e & 1), pos = offset + row0 + g + (e >> 1) * 8;
                    sc[n][e] = (key <= pos && key < Tk) ? sc[n][e] * scale_log2 : -INFINITY;
                    mx[e >> 1] = fmaxf(mx[e >> 1], sc[n][e]);
                }
            }
            float alpha[2], base[2];
#pragma unroll
            for (int r = 0; r < 2; r++) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
                const float mn = fmaxf(m[r], mx[r]);
                base[r] = mn == -INFINITY ? 0.f : mn;
                alpha[r] = exp2f(m[r] - base[r]);
                m[r] = mn;
                l[r] *= alpha[r];
            }
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    sc[n][e] = exp2f(sc[n][e] - base[e >> 1]);
                    l[e >> 1] += sc[n][e];
                }
            }
#pragma unroll
            for (int n = 0; n < D / 8; n++) {
                o[n][0] *= alpha[0];
                o[n][1] *= alpha[0];
                o[n][2] *= alpha[1];
                o[n][3] *= alpha[1];
            }
            // the score accumulators, two 8-key tiles at a time, are the A fragments of P*V
#pragma unroll
            for (int j = 0; j < BN / 16; j++) {
                p[j][0] = pack_bf16(sc[2 * j][0], sc[2 * j][1]);
                p[j][1] = pack_bf16(sc[2 * j][2], sc[2 * j][3]);
                p[j][2] = pack_bf16(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                p[j][3] = pack_bf16(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        cp_async_wait<0>();  // V
        __syncthreads();
        if (active) {
#pragma unroll
            for (int j = 0; j < BN / 16; j++) {
#pragma unroll
                for (int n = 0; n < D / 8; n += 2) {
                    uint32_t b[4];
                    load_b_kn(b, vs, LDS, j * 16, n * 8, lane);
                    mma16816(o[n], p[j], b[0], b[1]);
                    mma16816(o[n + 1], p[j], b[2], b[3]);
                }
            }
        }
        __syncthreads();  // before the next tile overwrites K and V
    }

    // the four lanes of a row each summed a quarter of its keys
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
    const float inv[2] = {1.f / l[0], 1.f / l[1]};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = row0 + g + r * 8;
        if (row >= Tq) continue;
        bf16* dst = out + ((size_t)row * Hq + h) * D + 2 * t;
#pragma unroll
        for (int n = 0; n < D / 8; n++) {
            float a = o[n][2 * r] * inv[r], b = o[n][2 * r + 1] * inv[r];
            if constexpr (Gated) {
                const size_t idx = ((size_t)row * Hq + h) * D + 2 * t + n * 8;
                // Match attention -> BF16 store -> BF16 sigmoid -> multiply.
                // Rounding before the multiply is required even without that store.
                a = round_bf16(a) * round_bf16(sigmoid(f32(gate[idx])));
                b = round_bf16(b) * round_bf16(sigmoid(f32(gate[idx + 1])));
            }
            *reinterpret_cast<uint32_t*>(dst + n * 8) = pack_bf16(a, b);
        }
    }
}

template <bool Gated>
int launch(const void* q, const void* k, const void* v, int ldv, const void* gate, void* out,
           int Tq, int Tk, int Hq, int Hk, int Dh, float scale, void* stream) {
    if (Dh != D || Hk <= 0 || Hq <= 0 || Hq % Hk != 0 || ldv % 8 != 0 || ldv < Hk * Dh || Tq < 0 || Tk < Tq)
        return cudaErrorInvalidValue;
    if (Tq == 0) return cudaSuccess;
    if (Gated && gate == nullptr) return cudaErrorInvalidValue;
    // Once per specialization (for the device current at the first call).
    static const cudaError_t configured = cudaFuncSetAttribute(
        flash_kernel<Gated>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (configured != cudaSuccess) return configured;
    constexpr float LOG2E = 1.4426950408889634f;
    flash_kernel<Gated><<<dim3((Tq + BM - 1) / BM, Hq), THREADS, SMEM_BYTES,
                         static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(k), static_cast<const bf16*>(v), ldv,
        static_cast<const bf16*>(gate), static_cast<bf16*>(out), Tq, Tk, Hq, Hk, scale * LOG2E);
    return cudaGetLastError();
}

}  // namespace flash
}  // namespace
}  // namespace cs1
using namespace cs1;

extern "C" int cs1_attention_gated_cached(const void* q, const void* k, const void* v, int ldv, const void* gate,
                                          void* out, int Tq, int Tk, int Hq, int Hk, int Dh, float scale,
                                          void* stream) {
    return flash::launch<true>(q, k, v, ldv, gate, out, Tq, Tk, Hq, Hk, Dh, scale, stream);
}
