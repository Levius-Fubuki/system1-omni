// bfloat16 GEMMs through cuBLASLt, float32 accumulation.
//
// Row-major y [M, N] = x [M, K] * w [N, K]^T is the column-major product
// y^T [N, M] = (w viewed as [K, N])^T * (x viewed as [K, M]); y's rows may be
// strided (ldy >= N), so one GEMM can fill a slice of a wider buffer.
//
// Algorithms: cs1_gemm_tune times cuBLASLt's candidates for a shape with L2 flushed
// before every call and keeps the fastest: the heuristic's shortlist, or (exhaustive)
// each algorithm id with each tile, stage count, custom option and swizzle it supports
// and split-K factors of 1 to 6, 8, 12 and 16, as far as cuBLASLt accepts them for the
// shape (a first pass of one call each keeps 12 to time properly).
// Many configurations are within noise of each other, so two searches often keep
// different ones with about the same speed.
// It replaces the heuristic's first choice only when it is more than 3% faster, so
// near ties rarely change between runs, and
// cs1_gemm_export / cs1_gemm_import let a caller keep the choices across runs, for the
// cuBLASLt version they were tuned with. A shape that was not tuned borrows the
// algorithm tuned for a nearby M with the same N, K and ldy (see plan_for), or takes
// the heuristic's first choice. Split-K reductions that accumulate into the output in place are
// excluded, since their order, and so the rounding, is not fixed.
#include <cublasLt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#include "ops.h"

namespace {

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool tuned = false;
};

using Key = std::tuple<int, int, int, int>;  // M, N, K, ldy

constexpr size_t FLUSH_BYTES = 256u << 20;

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::map<Key, Plan> plans;
    // tuning only: a buffer larger than L2, a sink for its reads, and two events
    void* flush = nullptr;
    int* sink = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
};

void release_tuning(Gemm& g) {
    if (g.flush) cudaFree(g.flush);
    if (g.sink) cudaFree(g.sink);
    if (g.e0) cudaEventDestroy(g.e0);
    if (g.e1) cudaEventDestroy(g.e1);
    g.flush = nullptr;
    g.sink = nullptr;
    g.e0 = g.e1 = nullptr;
}

int status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + (int)s; }

void destroy(Plan& p) {
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    p = Plan{};
}

int describe(int M, int N, int K, int ldy, Plan& p) {
    cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    if ((s = cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, ldy)) != CUBLAS_STATUS_SUCCESS) return status(s);
    return 0;
}

int heuristics(Gemm& g, const Plan& p, int want, std::vector<cublasLtMatmulHeuristicResult_t>& out) {
    cublasLtMatmulPreference_t pref;
    cublasStatus_t s = cublasLtMatmulPreferenceCreate(&pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.workspace_bytes,
                                         sizeof(g.workspace_bytes));
    const uint32_t schemes = CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
                                         sizeof(schemes));
    out.resize(want);
    int found = 0;
    s = cublasLtMatmulAlgoGetHeuristic(g.handle, p.op, p.a, p.b, p.c, p.c, pref, want, out.data(), &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    out.resize(found);
    out.erase(std::remove_if(out.begin(), out.end(),
                             [](const cublasLtMatmulHeuristicResult_t& r) { return r.state != CUBLAS_STATUS_SUCCESS; }),
              out.end());
    if (out.empty()) return status(CUBLAS_STATUS_NOT_SUPPORTED);
    return 0;
}

bool usable(Gemm& g, const Plan& p, const cublasLtMatmulAlgo_t& algo) {
    cublasLtMatmulHeuristicResult_t r{};
    return cublasLtMatmulAlgoCheck(g.handle, p.op, p.a, p.b, p.c, p.c, &algo, &r) == CUBLAS_STATUS_SUCCESS &&
           r.workspaceSize <= g.workspace_bytes;
}

bool reduces_in_place(const cublasLtMatmulAlgo_t& algo) {
    uint32_t red = CUBLASLT_REDUCTION_SCHEME_NONE;
    size_t n = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &n);
    return red == CUBLASLT_REDUCTION_SCHEME_INPLACE;
}

template <typename T>
std::vector<T> cap_array(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t bytes = 0;
    cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &bytes);
    std::vector<T> v(bytes / sizeof(T));
    if (bytes) cublasLtMatmulAlgoCapGetAttribute(&algo, attr, v.data(), bytes, &bytes);
    return v;
}

template <typename T>
T cap(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoCapAttributes_t attr) {
    T v{};
    size_t n;
    cublasLtMatmulAlgoCapGetAttribute(&algo, attr, &v, sizeof v, &n);
    return v;
}

// The configurations cuBLASLt accepts for the shape, within the workspace: each
// algorithm id with each tile, stage count, custom option and swizzle it supports, and
// split-K factors from `splits` (reduced in the compute or the output type, not in
// place). Other attributes stay at their defaults.
int every_config(Gemm& g, const Plan& p, std::vector<cublasLtMatmulAlgo_t>& out) {
    int ids[256], nids = 0;
    const cublasStatus_t s = cublasLtMatmulAlgoGetIds(g.handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF,
                                                      CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 256, ids, &nids);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const int splits[] = {1, 2, 3, 4, 5, 6, 8, 12, 16};
    const uint32_t schemes[] = {CUBLASLT_REDUCTION_SCHEME_NONE, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                                CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE};
    for (int i = 0; i < nids; i++) {
        cublasLtMatmulAlgo_t base;
        if (cublasLtMatmulAlgoInit(g.handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                                   CUDA_R_16BF, ids[i], &base) != CUBLAS_STATUS_SUCCESS)
            continue;
        auto tiles = cap_array<uint32_t>(base, CUBLASLT_ALGO_CAP_TILE_IDS);
        auto stages = cap_array<uint32_t>(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
        if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
        if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
        const int splitk_ok = cap<int>(base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
        const uint32_t red_mask = cap<uint32_t>(base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
        const int swizzle_ok = cap<int>(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
        const int custom_max = cap<int>(base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
        for (uint32_t tile : tiles)
            for (uint32_t stage : stages)
                for (int custom = 0; custom <= custom_max; custom++)
                    for (int swz = 0; swz <= swizzle_ok; swz++)
                        for (int sk : splits) {
                            if (sk > 1 && !splitk_ok) break;
                            for (uint32_t red : schemes) {
                                if ((sk == 1) != (red == CUBLASLT_REDUCTION_SCHEME_NONE)) continue;
                                if (red != CUBLASLT_REDUCTION_SCHEME_NONE && !(red_mask & red)) continue;
                                cublasLtMatmulAlgo_t a = base;
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                                                     sizeof tile);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage,
                                                                     sizeof stage);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom,
                                                                     sizeof custom);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz,
                                                                     sizeof swz);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk,
                                                                     sizeof sk);
                                cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red,
                                                                     sizeof red);
                                if (usable(g, p, a)) out.push_back(a);
                            }
                        }
    }
    return 0;
}

// The plan for a shape, created on first use.
int plan_for(Gemm& g, int M, int N, int K, int ldy, Plan*& out) {
    const Key key{M, N, K, ldy};
    auto it = g.plans.find(key);
    if (it != g.plans.end()) {
        out = &it->second;
        return 0;
    }
    Plan p;
    int rc = describe(M, N, K, ldy, p);
    if (rc != 0) {
        destroy(p);
        return rc;
    }
    // Borrow a tuned algorithm: the one for the smallest tuned M above, if that M is at
    // most twice this one; the one for the largest tuned M below, if no larger M was
    // tuned; else take the heuristic's first choice.
    const Plan* above = nullptr;
    const Plan* below = nullptr;
    int above_m = 0, below_m = 0;
    for (auto& kv : g.plans) {
        const auto [m, n, k, l] = kv.first;
        if (n != N || k != K || l != ldy || !kv.second.tuned) continue;
        if (m > M && (!above || m < above_m)) above = &kv.second, above_m = m;
        if (m < M && (!below || m > below_m)) below = &kv.second, below_m = m;
    }
    if (above && above_m <= 2 * M && usable(g, p, above->algo)) {
        p.algo = above->algo;
    } else if (!above && below && usable(g, p, below->algo)) {
        p.algo = below->algo;
    } else {
        std::vector<cublasLtMatmulHeuristicResult_t> cands;
        rc = heuristics(g, p, 1, cands);
        if (rc != 0) {
            destroy(p);
            return rc;
        }
        p.algo = cands[0].algo;
    }
    out = &g.plans.emplace(key, p).first->second;
    return 0;
}

// With CUA_S1_GEMM_LOG set, print each tuned choice to stderr.
void log_choice(int M, int N, int K, const cublasLtMatmulAlgo_t& a, float ms, float first_ms, size_t pick) {
    static const bool on = std::getenv("CUA_S1_GEMM_LOG") != nullptr;
    if (!on) return;
    int tile = 0, stages = 0, splitk = 0, inner = 0, id = 0;
    size_t n;
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(int), &n);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(int), &n);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(int), &n);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(int), &n);
    cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(int), &n);
    fprintf(stderr, "gemm %5d x %5d x %5d: candidate %zu, algo %d tile %d stages %d splitK %d inner %d, %.1f us (first %.1f us)\n",
            M, N, K, pick, id, tile, stages, splitk, inner, ms * 1e3f, first_ms * 1e3f);
}

// Read a buffer larger than L2, so the next call finds none of its operands cached.
__global__ void flush_l2(const int4* p, size_t n, int* sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc ^= p[i].x ^ p[i].w;
    if (acc == 0x7fffffff) *sink = acc;
}

}  // namespace

extern "C" void* cs1_gemm_create(size_t workspace_bytes) {
    Gemm* g = new Gemm();
    if (cublasLtCreate(&g->handle) != CUBLAS_STATUS_SUCCESS ||
        (workspace_bytes > 0 && cudaMalloc(&g->workspace, workspace_bytes) != cudaSuccess)) {
        if (g->handle) cublasLtDestroy(g->handle);
        delete g;
        return nullptr;
    }
    g->workspace_bytes = workspace_bytes;
    return g;
}

extern "C" void cs1_gemm_destroy(void* gemm) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g) return;
    for (auto& kv : g->plans) destroy(kv.second);
    release_tuning(*g);
    if (g->workspace) cudaFree(g->workspace);
    cublasLtDestroy(g->handle);
    delete g;
}

extern "C" int cs1_gemm_tune(void* gemm, const void* x, const void* w, void* y, int M, int N, int K, int ldy,
                             int exhaustive, void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M <= 0 || N <= 0 || K <= 0 || ldy < N) return cudaErrorInvalidValue;
    const Key key{M, N, K, ldy};
    auto it = g->plans.find(key);
    if (it != g->plans.end() && it->second.tuned) return 0;
    if (it != g->plans.end()) {
        destroy(it->second);
        g->plans.erase(it);
    }
    Plan p;
    int rc = describe(M, N, K, ldy, p);
    std::vector<cublasLtMatmulHeuristicResult_t> shortlist;
    if (rc == 0) rc = heuristics(*g, p, 16, shortlist);
    if (rc != 0) {
        destroy(p);
        return rc;
    }
    // candidates: the heuristic's shortlist first, then (exhaustive) those of every_config
    std::vector<cublasLtMatmulAlgo_t> cands;
    for (auto& r : shortlist) cands.push_back(r.algo);
    if (exhaustive) {
        std::vector<cublasLtMatmulAlgo_t> all;
        rc = every_config(*g, p, all);
        if (rc != 0) {
            destroy(p);
            return rc;
        }
        for (auto& a : all)
            if (std::none_of(cands.begin(), cands.end(),
                             [&](const cublasLtMatmulAlgo_t& c) { return std::memcmp(&c, &a, sizeof a) == 0; }))
                cands.push_back(a);
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (!g->flush) {
        if (cudaMalloc(&g->flush, FLUSH_BYTES) != cudaSuccess || cudaMalloc(&g->sink, sizeof(int)) != cudaSuccess ||
            cudaMemsetAsync(g->flush, 0, FLUSH_BYTES, st) != cudaSuccess || cudaEventCreate(&g->e0) != cudaSuccess ||
            cudaEventCreate(&g->e1) != cudaSuccess) {
            release_tuning(*g);
            destroy(p);
            cudaGetLastError();
            return (int)cudaErrorMemoryAllocation;
        }
    }
    const float alpha = 1.f, beta = 0.f;
    // median of `reps` calls, each after an L2 flush; a huge value if the call fails
    auto time = [&](const cublasLtMatmulAlgo_t& algo, int reps) {
        if (cublasLtMatmul(g->handle, p.op, &alpha, w, p.a, x, p.b, &beta, y, p.c, y, p.c, &algo, g->workspace,
                           g->workspace_bytes, st) != CUBLAS_STATUS_SUCCESS) {
            cudaGetLastError();
            return 1e30f;
        }
        std::vector<float> times;
        for (int r = 0; r < reps; r++) {
            flush_l2<<<1024, 256, 0, st>>>(static_cast<const int4*>(g->flush), FLUSH_BYTES / sizeof(int4), g->sink);
            cudaEventRecord(g->e0, st);
            cublasLtMatmul(g->handle, p.op, &alpha, w, p.a, x, p.b, &beta, y, p.c, y, p.c, &algo, g->workspace,
                           g->workspace_bytes, st);
            cudaEventRecord(g->e1, st);
            if (cudaEventSynchronize(g->e1) != cudaSuccess) return 1e30f;
            float ms = 0.f;
            cudaEventElapsedTime(&ms, g->e0, g->e1);
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        return times[reps / 2];
    };
    // with many candidates, one timed call each picks the 12 to time properly
    std::vector<size_t> keep;
    if (cands.size() > 16) {
        std::vector<std::pair<float, size_t>> quick;
        for (size_t i = 0; i < cands.size(); i++) quick.push_back({time(cands[i], 1), i});
        std::sort(quick.begin(), quick.end());
        keep.push_back(0);  // the heuristic's first choice, the baseline
        for (size_t j = 0; j < quick.size() && keep.size() < 13; j++)
            if (quick[j].second != 0 && quick[j].first < 1e30f) keep.push_back(quick[j].second);
    } else {
        for (size_t i = 0; i < cands.size(); i++) keep.push_back(i);
    }
    // nine timed calls, or three for shapes that take over 2 ms
    const int reps = time(cands[0], 1) > 2.f ? 3 : 9;
    std::vector<float> median(cands.size(), 1e30f);
    for (size_t i : keep) median[i] = time(cands[i], reps);
    // the heuristic's first working choice, unless another is more than 3% faster
    size_t pick = 0;
    while (pick < median.size() && median[pick] >= 1e30f) pick++;
    if (pick == median.size()) {
        destroy(p);
        return status(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    const size_t first = pick;
    const size_t fastest = std::min_element(median.begin(), median.end()) - median.begin();
    if (median[fastest] < 0.97f * median[pick]) pick = fastest;
    p.algo = cands[pick];
    p.tuned = true;
    log_choice(M, N, K, p.algo, median[pick], median[first], pick);
    g->plans.emplace(key, p);
    return (int)cudaGetLastError();
}

extern "C" void cs1_gemm_tune_done(void* gemm) {
    if (gemm) release_tuning(*static_cast<Gemm*>(gemm));
}

extern "C" size_t cs1_gemm_export(void* gemm, Cs1GemmPlan* out, size_t cap) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g) return 0;
    size_t n = 0;
    for (auto& kv : g->plans) {
        if (!kv.second.tuned) continue;
        if (n < cap) {
            const auto [m, nn, k, l] = kv.first;
            out[n] = Cs1GemmPlan{m, nn, k, l, (uint64_t)cublasLtGetVersion(), {}};
            static_assert(sizeof(cublasLtMatmulAlgo_t) == sizeof(out[n].algo), "algo layout");
            std::memcpy(out[n].algo, &kv.second.algo, sizeof(out[n].algo));
        }
        n++;
    }
    return n;
}

extern "C" int cs1_gemm_import(void* gemm, const Cs1GemmPlan* plans, size_t n) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g) return cudaErrorInvalidValue;
    // check every plan before using any, so that a rejected file changes nothing
    std::vector<std::pair<Key, Plan>> checked;
    int rc = 0;
    for (size_t i = 0; i < n && rc == 0; i++) {
        const Cs1GemmPlan& r = plans[i];
        if (r.m <= 0 || r.n <= 0 || r.k <= 0 || r.ldy < r.n) {
            rc = cudaErrorInvalidValue;
            break;
        }
        if (r.cublaslt_version != (uint64_t)cublasLtGetVersion()) {
            rc = status(CUBLAS_STATUS_NOT_SUPPORTED);
            break;
        }
        Plan p;
        rc = describe(r.m, r.n, r.k, r.ldy, p);
        if (rc == 0) {
            std::memcpy(&p.algo, r.algo, sizeof(p.algo));
            if (reduces_in_place(p.algo) || !usable(*g, p, p.algo)) rc = status(CUBLAS_STATUS_NOT_SUPPORTED);
        }
        if (rc != 0) {
            destroy(p);
            break;
        }
        p.tuned = true;
        checked.emplace_back(Key{r.m, r.n, r.k, r.ldy}, p);
    }
    if (rc != 0) {
        for (auto& kv : checked) destroy(kv.second);
        return rc;
    }
    for (auto& [key, p] : checked) {
        auto it = g->plans.find(key);
        if (it != g->plans.end()) {
            destroy(it->second);
            it->second = p;
        } else {
            g->plans.emplace(key, p);
        }
    }
    return 0;
}

extern "C" size_t cs1_gemm_version(void) { return cublasLtGetVersion(); }

extern "C" int cs1_gemm(void* gemm, const void* x, const void* w, void* y, int M, int N, int K, int ldy,
                        void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M < 0 || N <= 0 || K <= 0 || ldy < N) return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    Plan* p = nullptr;
    const int rc = plan_for(*g, M, N, K, ldy, p);
    if (rc != 0) return rc;
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}
