// ascend/tests/ascend_parity_test.cpp - Numerical parity test for all Strata Ascend C (dav-m200) & Cube aclnnMm ops.
#include "ascend_ops.h"
#include "ref_quant.h"

#include "acl/acl.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_CHECK(expr)                                                                                 \
    do {                                                                                                \
        aclError _e = (expr);                                                                           \
        if (_e != ACL_SUCCESS) {                                                                        \
            std::fprintf(stderr, "ACL error %d at %s:%d: %s\n", (int)_e, __FILE__, __LINE__, #expr);    \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

namespace {

struct DevBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit DevBuf(size_t b) : bytes(((b + 63) / 64) * 64 + 64) {
        ACL_CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(ptr, bytes, 0, bytes));
    }
    ~DevBuf() {
        if (ptr) aclrtFree(ptr);
    }
    void upload(const void* h, size_t b) { ACL_CHECK(aclrtMemcpy(ptr, bytes, h, b, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void download(void* h, size_t b) const { ACL_CHECK(aclrtMemcpy(h, b, ptr, b, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

struct MaxDiff {
    float max_abs = 0.0f;
    double mse = 0.0;
};

MaxDiff compare_f32(const float* got, const float* exp, size_t n) {
    MaxDiff d{};
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(got[i] - exp[i]);
        if (std::isnan(got[i]) || std::isinf(got[i])) {
            d.max_abs = 1e30f;
            return d;
        }
        if (diff > d.max_abs) d.max_abs = diff;
        d.mse += (double)diff * (double)diff;
    }
    if (n > 0) d.mse = std::sqrt(d.mse / (double)n);
    return d;
}

float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

uint32_t float_okey_cpu(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

void norm_rope_cpu(const float* x, const float* w, const float* cs, float* out, int dim, float eps, bool do_rope) {
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) ss += x[i] * x[i];
    float rs = 1.0f / std::sqrt(ss / (float)dim + eps);
    for (int i = 0; i < dim; ++i) out[i] = x[i] * rs * w[i];
    if (do_rope) {
        for (int p = 0; p < 32; ++p) {
            float c = cs[p * 2], s = cs[p * 2 + 1];
            float a = out[p], b = out[p + 32];
            out[p] = a * c - b * s;
            out[p + 32] = a * s + b * c;
        }
    }
}

}  // namespace

int main() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    std::printf("=== Strata Ascend C (dav-m200) + Cube FRACTAL_NZ Parity Suite ===\n");
    ref::Rng rng(20261125ULL);

    // ------------------------------------------------------------------------
    // 1. Cast f32 <-> f16
    // ------------------------------------------------------------------------
    {
        const uint32_t N = 10240;
        std::vector<float> h_in(N), h_out(N), h_ref(N);
        for (uint32_t i = 0; i < N; ++i) {
            h_in[i] = rng.sym() * 3.0f;
            h_ref[i] = ref::f16_to_f32(ref::f32_to_f16(h_in[i]));
        }
        DevBuf d_f32(N * 4), d_f16(N * 2), d_back(N * 4);
        d_f32.upload(h_in.data(), N * 4);
        strata::ascend_ops::launch_f32_to_f16(stream, d_f32.ptr, d_f16.ptr, N);
        strata::ascend_ops::launch_f16_to_f32(stream, d_f16.ptr, d_back.ptr, N);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        d_back.download(h_out.data(), N * 4);
        auto md = compare_f32(h_out.data(), h_ref.data(), N);
        std::printf("[1] f32<->f16: max_abs=%.3e rmse=%.3e %s\n", md.max_abs, md.mse, md.max_abs < 1e-6f ? "PASS" : "FAIL");
        if (md.max_abs >= 1e-6f) return 1;
    }

    // ------------------------------------------------------------------------
    // 2. Cube aclnnMm with ACL_FORMAT_FRACTAL_NZ ([4, 2560] x [2560, 10240])
    // ------------------------------------------------------------------------
    {
        const int64_t M = 4, K = 2560, N = 10240;
        std::vector<uint16_t> hA(M * K), hB_kn(K * N), hC(M * N);
        std::vector<float> fA(M * K), fB_kn(K * N), fRef(M * N, 0.0f), fGot(M * N);
        for (int64_t i = 0; i < M * K; ++i) {
            uint16_t h = ref::f32_to_f16(rng.sym() * 0.5f);
            hA[i] = h; fA[i] = ref::f16_to_f32(h);
        }
        for (int64_t i = 0; i < K * N; ++i) {
            uint16_t h = ref::f32_to_f16(rng.sym() * 0.1f);
            hB_kn[i] = h; fB_kn[i] = ref::f16_to_f32(h);
        }
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < K; ++k) {
                float a = fA[m * K + k];
                const float* br = fB_kn.data() + k * N;
                float* cr = fRef.data() + m * N;
                for (int64_t n = 0; n < N; ++n) cr[n] += a * br[n];
            }

        DevBuf dA(M * K * 2), dB(K * N * 2), dC(M * N * 2), dWs(64 * 1024 * 1024);
        dA.upload(hA.data(), M * K * 2);
        dB.upload(hB_kn.data(), K * N * 2);

        int64_t bShape[2] = {K, N}, bStride[2] = {N, 1};
        aclTensor* tB_nd = aclCreateTensor(bShape, 2, ACL_FLOAT16, bStride, 0, ACL_FORMAT_ND, bShape, 2, dB.ptr);
        uint64_t ws_sz = 0;
        aclOpExecutor* exec_tr = nullptr;
        ACL_CHECK(aclnnTransMatmulWeightGetWorkspaceSize(tB_nd, &ws_sz, &exec_tr));
        ACL_CHECK(aclnnTransMatmulWeight(dWs.ptr, ws_sz, exec_tr, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        aclDestroyTensor(tB_nd);

        int64_t aShape[2] = {M, K}, aStride[2] = {K, 1};
        int64_t cShape[2] = {M, N}, cStride[2] = {N, 1};
        aclTensor* tA = aclCreateTensor(aShape, 2, ACL_FLOAT16, aStride, 0, ACL_FORMAT_ND, aShape, 2, dA.ptr);
        aclTensor* tB_nz = aclCreateTensor(bShape, 2, ACL_FLOAT16, bStride, 0, ACL_FORMAT_FRACTAL_NZ, bShape, 2, dB.ptr);
        aclTensor* tC = aclCreateTensor(cShape, 2, ACL_FLOAT16, cStride, 0, ACL_FORMAT_ND, cShape, 2, dC.ptr);
        aclOpExecutor* exec_mm = nullptr;
        ACL_CHECK(aclnnMmGetWorkspaceSize(tA, tB_nz, tC, 0, &ws_sz, &exec_mm));
        ACL_CHECK(aclnnMm(dWs.ptr, ws_sz, exec_mm, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        dC.download(hC.data(), M * N * 2);
        for (int64_t i = 0; i < M * N; ++i) fGot[i] = ref::f16_to_f32(hC[i]);
        auto md = compare_f32(fGot.data(), fRef.data(), M * N);
        std::printf("[2] aclnnMm FRACTAL_NZ: max_abs=%.3e rmse=%.3e %s\n", md.max_abs, md.mse, md.max_abs < 2e-2f ? "PASS" : "FAIL");
        aclDestroyTensor(tA); aclDestroyTensor(tB_nz); aclDestroyTensor(tC);
        if (md.max_abs >= 2e-2f) return 1;
    }

    // ------------------------------------------------------------------------
    // 3. Hyper-connection (gr_norm + gr_post_down + gr_post_up)
    // ------------------------------------------------------------------------
    {
        const uint32_t T = 4, HC = 4, N = 2560, WSTR = 336, LR = 320;
        const float eps = 1e-6f;
        std::vector<float> hR(T * HC * N), hBo(T * N), hInj(32, 0.0f), hWn(HC * N);
        for (auto& v : hR) v = rng.sym();
        for (auto& v : hBo) v = rng.sym() * 0.5f;
        for (uint32_t i = 0; i < T * HC; ++i) hInj[i] = rng.sym() * 2.0f;
        for (auto& v : hWn) v = 0.8f + 0.4f * rng.uni();

        std::vector<float> refR = hR, refXn32(T * HC * N);
        for (uint32_t idx = 0; idx < T * HC; ++idx) {
            uint32_t t = idx / HC, c = idx % HC;
            float g = 2.0f / (1.0f + std::exp(-hInj[idx] * 0.25f));
            float ss = 0.0f;
            for (uint32_t d = 0; d < N; ++d) {
                refR[idx * N + d] += g * hBo[t * N + d];
                ss += refR[idx * N + d] * refR[idx * N + d];
            }
            float rs = 1.0f / std::sqrt(ss / (float)N + eps);
            for (uint32_t d = 0; d < N; ++d)
                refXn32[idx * N + d] = refR[idx * N + d] * rs * hWn[c * N + d];
        }

        DevBuf dR(hR.size() * 4), dBo(hBo.size() * 4), dInj(32 * 4), dWn(hWn.size() * 4);
        DevBuf dXn32(T * HC * N * 4), dXn16(T * HC * N * 2);
        dR.upload(hR.data(), hR.size() * 4);
        dBo.upload(hBo.data(), hBo.size() * 4);
        dInj.upload(hInj.data(), 32 * 4);
        dWn.upload(hWn.data(), hWn.size() * 4);
        strata::ascend_ops::launch_gr_norm(stream, dR.ptr, dBo.ptr, dInj.ptr, dWn.ptr, dXn32.ptr, dXn16.ptr, T, 1, eps);

        // Post-down & Post-up
        std::vector<uint16_t> hDn16(T * WSTR), hGate16(T * HC * N);
        for (auto& v : hDn16) v = ref::f32_to_f16(rng.sym() * 2.0f);
        for (auto& v : hGate16) v = ref::f32_to_f16(rng.sym() * 2.0f);
        DevBuf dDn16(T * WSTR * 2), dLo16(T * LR * 2), dInjNext(32 * 4);
        DevBuf dGate16(T * HC * N * 2), dOut16(T * N * 2), dOut32(T * N * 4);
        dDn16.upload(hDn16.data(), hDn16.size() * 2);
        dGate16.upload(hGate16.data(), hGate16.size() * 2);
        strata::ascend_ops::launch_gr_post_down(stream, dDn16.ptr, dLo16.ptr, dInjNext.ptr, T);
        strata::ascend_ops::launch_gr_post_up(stream, dXn32.ptr, dGate16.ptr, dOut16.ptr, dOut32.ptr, T);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<float> gotR(hR.size()), gotXn32(refXn32.size()), gotInjNext(T * HC), gotOut32(T * N);
        std::vector<uint16_t> gotLo16(T * LR);
        dR.download(gotR.data(), gotR.size() * 4);
        dXn32.download(gotXn32.data(), gotXn32.size() * 4);
        dInjNext.download(gotInjNext.data(), T * HC * 4);
        dLo16.download(gotLo16.data(), T * LR * 2);
        dOut32.download(gotOut32.data(), T * N * 4);

        std::vector<float> refLo(T * LR), gotLo(T * LR), refInjNext(T * HC), refOut32(T * N);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t k = 0; k < LR; ++k) {
                refLo[t * LR + k] = ref::f16_to_f32(ref::f32_to_f16(silu_f(ref::f16_to_f32(hDn16[t * WSTR + k]) * 0.25f)));
                gotLo[t * LR + k] = ref::f16_to_f32(gotLo16[t * LR + k]);
            }
            for (uint32_t c = 0; c < HC; ++c) refInjNext[t * HC + c] = ref::f16_to_f32(hDn16[t * WSTR + LR + c]);
            for (uint32_t d = 0; d < N; ++d) {
                float m = 0.0f;
                for (uint32_t c = 0; c < HC; ++c)
                    m += refXn32[(t * HC + c) * N + d] * sigmoid_f(ref::f16_to_f32(hGate16[(t * HC + c) * N + d]));
                refOut32[t * N + d] = m * 0.25f;
            }
        }
        auto m1 = compare_f32(gotXn32.data(), refXn32.data(), refXn32.size());
        auto m2 = compare_f32(gotLo.data(), refLo.data(), refLo.size());
        auto m3 = compare_f32(gotInjNext.data(), refInjNext.data(), refInjNext.size());
        auto m4 = compare_f32(gotOut32.data(), refOut32.data(), refOut32.size());
        float mx = std::max({m1.max_abs, m2.max_abs, m3.max_abs, m4.max_abs});
        std::printf("[3] gr_norm/post_down/post_up: xn=%.2e lo=%.2e inj=%.2e out=%.2e %s\n",
                    m1.max_abs, m2.max_abs, m3.max_abs, m4.max_abs, mx < 2e-3f ? "PASS" : "FAIL");
        if (mx >= 2e-3f) return 1;
    }

    // ------------------------------------------------------------------------
    // 4. Fused GDN Recurrence (verify commit=0 + fold from_saved=1 commit=2)
    // ------------------------------------------------------------------------
    {
        const uint32_t T = 4, HK = 16, HV = 48, S = 128, CONV = 10240, ZV = 6144, INSTR = 16480;
        const float eps = 1e-6f;
        std::vector<float> hState(HV * S * S), hHist(CONV * 4, 0.0f), hCw(CONV * 4), hDt(64, 0.0f), hSa(64, 0.0f), hGm(S);
        std::vector<uint16_t> hIn16(T * INSTR);
        for (auto& v : hState) v = rng.sym() * 0.05f;
        for (uint32_t c = 0; c < CONV; ++c)
            for (uint32_t k = 0; k < 3; ++k) hHist[c * 4 + k] = rng.sym() * 0.2f;
        for (auto& v : hCw) v = rng.sym() * 0.3f;
        for (uint32_t h = 0; h < HV; ++h) { hDt[h] = rng.sym() * 0.5f; hSa[h] = -0.1f - 0.5f * rng.uni(); }
        for (auto& v : hGm) v = 0.8f + 0.4f * rng.uni();
        for (auto& v : hIn16) v = ref::f32_to_f16(rng.sym() * 0.5f);

        // CPU reference: run T=4 verify (no commit), and fold first 2 tokens committed
        std::vector<float> refYg(T * ZV), refState2 = hState, refHist2 = hHist;
        const float scale = 1.0f / std::sqrt(128.0f);
        for (uint32_t h = 0; h < HV; ++h) {
            uint32_t hk = h % HK;
            uint32_t ch_q = hk * S, ch_k = HK * S + hk * S, ch_v = 2 * HK * S + h * S;
            std::vector<float> st(hState.begin() + h * S * S, hState.begin() + (h + 1) * S * S);
            std::vector<float> hiQ(hHist.begin() + ch_q * 4, hHist.begin() + (ch_q + S) * 4);
            std::vector<float> hiK(hHist.begin() + ch_k * 4, hHist.begin() + (ch_k + S) * 4);
            std::vector<float> hiV(hHist.begin() + ch_v * 4, hHist.begin() + (ch_v + S) * 4);

            for (uint32_t t = 0; t < T; ++t) {
                const uint16_t* row = hIn16.data() + t * INSTR;
                float qs[S], ks[S], vs[S], zs[S];
                float q_ss = 0.0f, k_ss = 0.0f;
                for (uint32_t d = 0; d < S; ++d) {
                    float iq = ref::f16_to_f32(row[ch_q + d]);
                    float ik = ref::f16_to_f32(row[ch_k + d]);
                    float iv = ref::f16_to_f32(row[ch_v + d]);
                    zs[d] = ref::f16_to_f32(row[CONV + h * S + d]);
                    const float* cwq = &hCw[(ch_q + d) * 4];
                    const float* cwk = &hCw[(ch_k + d) * 4];
                    const float* cwv = &hCw[(ch_v + d) * 4];
                    float sq = hiQ[d*4]*cwq[0] + hiQ[d*4+1]*cwq[1] + hiQ[d*4+2]*cwq[2] + iq*cwq[3];
                    float sk = hiK[d*4]*cwk[0] + hiK[d*4+1]*cwk[1] + hiK[d*4+2]*cwk[2] + ik*cwk[3];
                    float sv = hiV[d*4]*cwv[0] + hiV[d*4+1]*cwv[1] + hiV[d*4+2]*cwv[2] + iv*cwv[3];
                    hiQ[d*4]=hiQ[d*4+1]; hiQ[d*4+1]=hiQ[d*4+2]; hiQ[d*4+2]=iq;
                    hiK[d*4]=hiK[d*4+1]; hiK[d*4+1]=hiK[d*4+2]; hiK[d*4+2]=ik;
                    hiV[d*4]=hiV[d*4+1]; hiV[d*4+1]=hiV[d*4+2]; hiV[d*4+2]=iv;
                    qs[d] = silu_f(sq); q_ss += qs[d]*qs[d];
                    ks[d] = silu_f(sk); k_ss += ks[d]*ks[d];
                    vs[d] = silu_f(sv);
                }
                float q_rs = (1.0f / std::sqrt(q_ss / S + eps / S)) * scale;
                float k_rs = (1.0f / std::sqrt(k_ss / S + eps / S)) * scale;
                for (uint32_t d = 0; d < S; ++d) { qs[d] *= q_rs; ks[d] *= k_rs; }

                float alpha = ref::f16_to_f32(row[CONV + ZV + h]);
                float beta  = ref::f16_to_f32(row[CONV + ZV + HV + h]);
                float vv = alpha + hDt[h];
                float ev = std::exp(vv), u = 1.0f + ev;
                float sp = vv > 20.0f ? vv : (u == 1.0f ? ev : std::log(u) * ev / (u - 1.0f));
                float g = std::exp(sp * hSa[h]);
                float be = sigmoid_f(beta);

                float kv[S] = {}, o[S] = {};
                for (uint32_t i = 0; i < S; ++i)
                    for (uint32_t j = 0; j < S; ++j) kv[j] += st[i * S + j] * ks[i];
                for (uint32_t j = 0; j < S; ++j) {
                    float delta = (vs[j] - g * kv[j]) * be;
                    for (uint32_t i = 0; i < S; ++i) {
                        st[i * S + j] = g * st[i * S + j] + ks[i] * delta;
                        o[j] += st[i * S + j] * qs[i];
                    }
                }
                float o_ss = 0.0f;
                for (uint32_t j = 0; j < S; ++j) { o[j] *= scale; o_ss += o[j] * o[j]; }
                float o_rs = 1.0f / std::sqrt(o_ss / S + eps);
                for (uint32_t j = 0; j < S; ++j)
                    refYg[t * ZV + h * S + j] = ref::f16_to_f32(ref::f32_to_f16((o[j] * o_rs * hGm[j]) * sigmoid_f(zs[j])));

                if (t + 1 == 2) {
                    std::copy(st.begin(), st.end(), refState2.begin() + h * S * S);
                    if (h < HK) {
                        std::copy(hiQ.begin(), hiQ.end(), refHist2.begin() + ch_q * 4);
                        std::copy(hiK.begin(), hiK.end(), refHist2.begin() + ch_k * 4);
                    }
                    std::copy(hiV.begin(), hiV.end(), refHist2.begin() + ch_v * 4);
                }
            }
        }

        DevBuf dSt(hState.size() * 4), dHi(hHist.size() * 4), dHo(hHist.size() * 4);
        DevBuf dIn16(hIn16.size() * 2), dQkvS(T * CONV * 4), dZS(T * ZV * 4), dAbS(T * 96 * 4);
        DevBuf dCw(hCw.size() * 4), dDt(64 * 4), dSa(64 * 4), dGm(S * 4), dYg(T * ZV * 2);
        dSt.upload(hState.data(), hState.size() * 4);
        dHi.upload(hHist.data(), hHist.size() * 4);
        dIn16.upload(hIn16.data(), hIn16.size() * 2);
        dCw.upload(hCw.data(), hCw.size() * 4);
        dDt.upload(hDt.data(), 64 * 4);
        dSa.upload(hSa.data(), 64 * 4);
        dGm.upload(hGm.data(), S * 4);

        // Verify pass (commit_at = 0, saves unpacked inputs)
        strata::ascend_ops::launch_gdn_step(stream, dSt.ptr, dHi.ptr, dHo.ptr,
                                            dIn16.ptr, dQkvS.ptr, dZS.ptr, dAbS.ptr,
                                            dCw.ptr, dDt.ptr, dSa.ptr, dGm.ptr,
                                            dYg.ptr, T, 0, 0, eps);
        // Fold pass (n_tok = 2, commit_at = 2, from_saved = 1)
        strata::ascend_ops::launch_gdn_step(stream, dSt.ptr, dHi.ptr, dHo.ptr,
                                            dIn16.ptr, dQkvS.ptr, dZS.ptr, dAbS.ptr,
                                            dCw.ptr, dDt.ptr, dSa.ptr, dGm.ptr,
                                            dYg.ptr, 2, 2, 1, eps);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<uint16_t> gotYg16(T * ZV);
        std::vector<float> gotYg(T * ZV), gotSt(hState.size()), gotHo(hHist.size());
        dYg.download(gotYg16.data(), T * ZV * 2);
        dSt.download(gotSt.data(), gotSt.size() * 4);
        dHo.download(gotHo.data(), gotHo.size() * 4);
        for (size_t i = 0; i < gotYg.size(); ++i) gotYg[i] = ref::f16_to_f32(gotYg16[i]);

        auto my = compare_f32(gotYg.data(), refYg.data(), refYg.size());
        auto ms = compare_f32(gotSt.data(), refState2.data(), refState2.size());
        auto mh = compare_f32(gotHo.data(), refHist2.data(), refHist2.size());
        float mx = std::max({my.max_abs, ms.max_abs, mh.max_abs});
        std::printf("[4] gdn_step (verify+fold): yg=%.2e state=%.2e hist=%.2e %s\n",
                    my.max_abs, ms.max_abs, mh.max_abs, mx < 2e-3f ? "PASS" : "FAIL");
        if (mx >= 2e-3f) return 1;
    }

    // ------------------------------------------------------------------------
    // 5. QSA (prep + select + online int8 attn_merge, including sparse >2051 test)
    // ------------------------------------------------------------------------
    {
        const uint32_t QH = 24, QKV = 2, QD = 256, IH = 4, ID = 128, INSTR = 13952;
        const uint32_t MAX_CELLS = 2200, MAX_BLOCKS = 600, QIDS_STRIDE = 2056;
        const float eps = 1e-6f;

        std::vector<float> hQn(QD), hKn(QD), hIqn(ID), hIkn(ID), hCs((MAX_CELLS + 32) * 64);
        for (auto& v : hQn) v = 0.8f + 0.4f * rng.uni();
        for (auto& v : hKn) v = 0.8f + 0.4f * rng.uni();
        for (auto& v : hIqn) v = 0.8f + 0.4f * rng.uni();
        for (auto& v : hIkn) v = 0.8f + 0.4f * rng.uni();
        for (uint32_t pos = 0; pos < MAX_CELLS + 32; ++pos) {
            for (uint32_t p = 0; p < 32; ++p) {
                float freq = (float)pos * std::pow(10000.0f, -2.0f * (float)p / 64.0f);
                hCs[pos * 64 + 2 * p]     = std::cos(freq);
                hCs[pos * 64 + 2 * p + 1] = std::sin(freq);
            }
        }

        const uint32_t T = 6;
        std::vector<uint16_t> hIn16(T * INSTR);
        for (auto& v : hIn16) v = ref::f32_to_f16(rng.sym() * 0.6f);

        DevBuf dIn16(T * INSTR * 2), dQn(QD * 4), dKn(QD * 4), dIqn(ID * 4), dIkn(ID * 4), dCs(hCs.size() * 4);
        DevBuf dQcur(T * QH * QD * 4), dQgate(T * QH * QD * 4), dQidx(T * IH * ID * 4);
        DevBuf dKq(MAX_CELLS * QKV * QD), dVq(MAX_CELLS * QKV * QD);
        DevBuf dKsc(MAX_CELLS * 16 * 2), dVsc(MAX_CELLS * 16 * 2);
        DevBuf dTail(4 * ID * 4), dDead(ID * 4), dPool(MAX_BLOCKS * ID * 4);
        DevBuf dScores(T * MAX_BLOCKS * 4), dQids(T * QIDS_STRIDE * 4), dXq(T * QH * QD * 2);

        dIn16.upload(hIn16.data(), hIn16.size() * 2);
        dQn.upload(hQn.data(), QD * 4); dKn.upload(hKn.data(), QD * 4);
        dIqn.upload(hIqn.data(), ID * 4); dIkn.upload(hIkn.data(), ID * 4);
        dCs.upload(hCs.data(), hCs.size() * 4);

        strata::ascend_ops::launch_qsa_prep(stream, dIn16.ptr, dQn.ptr, dKn.ptr, dIqn.ptr, dIkn.ptr, dCs.ptr,
                                            dQcur.ptr, dQgate.ptr, dQidx.ptr,
                                            dKq.ptr, dVq.ptr, dKsc.ptr, dVsc.ptr,
                                            dTail.ptr, dDead.ptr, dPool.ptr, 0, 0, T, eps);
        strata::ascend_ops::launch_qsa_select(stream, dQidx.ptr, dPool.ptr, dDead.ptr,
                                              dScores.ptr, dQids.ptr, 0, MAX_BLOCKS, T);
        strata::ascend_ops::launch_qsa_attn_merge(stream, dQcur.ptr, dQgate.ptr,
                                                  dKq.ptr, dVq.ptr, dKsc.ptr, dVsc.ptr,
                                                  dQids.ptr, dXq.ptr, 0, T);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        // CPU reference for T=6 at pos0=0
        std::vector<int8_t> refKq(T * QKV * QD), refVq(T * QKV * QD);
        std::vector<uint16_t> refKsc(T * 16, 0), refVsc(T * 16, 0);
        std::vector<float> refXq(T * QH * QD);
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<float> row(INSTR);
            for (uint32_t i = 0; i < INSTR; ++i) row[i] = ref::f16_to_f32(hIn16[t * INSTR + i]);
            const float* cs = hCs.data() + t * 64;

            for (uint32_t h = 0; h < QKV; ++h) {
                float kn[QD];
                norm_rope_cpu(&row[12288 + h * QD], hKn.data(), cs, kn, QD, eps, true);
                const float* vr = &row[12288 + 512 + h * QD];
                for (uint32_t g = 0; g < 4; ++g) {
                    float mk = 0.0f, mv = 0.0f;
                    for (uint32_t j = 0; j < 64; ++j) {
                        mk = std::max(mk, std::fabs(kn[g * 64 + j]));
                        mv = std::max(mv, std::fabs(vr[g * 64 + j]));
                    }
                    uint16_t sk = ref::f32_to_f16(mk / 127.0f), sv = ref::f32_to_f16(mv / 127.0f);
                    float fk = ref::f16_to_f32(sk), fv = ref::f16_to_f32(sv);
                    refKsc[t * 16 + h * 4 + g] = sk;
                    refVsc[t * 16 + h * 4 + g] = sv;
                    for (uint32_t j = 0; j < 64; ++j) {
                        refKq[(t * QKV + h) * QD + g * 64 + j] = fk > 0 ? (int8_t)std::clamp((int)std::nearbyintf(kn[g * 64 + j] / fk), -127, 127) : 0;
                        refVq[(t * QKV + h) * QD + g * 64 + j] = fv > 0 ? (int8_t)std::clamp((int)std::nearbyintf(vr[g * 64 + j] / fv), -127, 127) : 0;
                    }
                }
            }

            for (uint32_t r = 0; r < QH; ++r) {
                float q[QD];
                norm_rope_cpu(&row[r * 2 * QD], hQn.data(), cs, q, QD, eps, true);
                const float* gate = &row[r * 2 * QD + QD];
                uint32_t kvh = r / 12;
                std::vector<float> scores(t + 1);
                float mx = -1e30f;
                for (uint32_t c = 0; c <= t; ++c) {
                    float dot = 0.0f;
                    for (uint32_t g = 0; g < 4; ++g) {
                        float ks = ref::f16_to_f32(refKsc[c * 16 + kvh * 4 + g]);
                        for (uint32_t j = 0; j < 64; ++j)
                            dot += q[g * 64 + j] * ((float)refKq[(c * QKV + kvh) * QD + g * 64 + j] * ks);
                    }
                    scores[c] = dot * 0.0625f;
                    mx = std::max(mx, scores[c]);
                }
                float l = 0.0f;
                float acc[QD] = {};
                for (uint32_t c = 0; c <= t; ++c) {
                    float p = std::exp(scores[c] - mx);
                    l += p;
                    for (uint32_t g = 0; g < 4; ++g) {
                        float vs = ref::f16_to_f32(refVsc[c * 16 + kvh * 4 + g]);
                        for (uint32_t j = 0; j < 64; ++j)
                            acc[g * 64 + j] += p * ((float)refVq[(c * QKV + kvh) * QD + g * 64 + j] * vs);
                    }
                }
                for (uint32_t d = 0; d < QD; ++d)
                    refXq[(t * QH + r) * QD + d] = ref::f16_to_f32(ref::f32_to_f16((acc[d] / l) * sigmoid_f(gate[d])));
            }
        }

        std::vector<uint16_t> gotXq16(T * QH * QD);
        std::vector<float> gotXq(T * QH * QD);
        dXq.download(gotXq16.data(), T * QH * QD * 2);
        for (size_t i = 0; i < gotXq.size(); ++i) gotXq[i] = ref::f16_to_f32(gotXq16[i]);
        auto mq = compare_f32(gotXq.data(), refXq.data(), refXq.size());

        // Also test sparse radix selector at pos0 = 2101 (n_kv = 2102 > 2051)
        std::vector<float> hPool(MAX_BLOCKS * ID), hDead(ID), hQi(IH * ID);
        for (auto& v : hPool) v = rng.sym();
        for (auto& v : hDead) v = rng.sym();
        for (auto& v : hQi) v = rng.sym();
        dPool.upload(hPool.data(), hPool.size() * 4);
        dDead.upload(hDead.data(), hDead.size() * 4);
        dQidx.upload(hQi.data(), hQi.size() * 4);
        const uint32_t pos_sparse = 2101;
        strata::ascend_ops::launch_qsa_select(stream, dQidx.ptr, dPool.ptr, dDead.ptr,
                                              dScores.ptr, dQids.ptr, pos_sparse, MAX_BLOCKS, 1);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<int32_t> gotIds(QIDS_STRIDE);
        dQids.download(gotIds.data(), QIDS_STRIDE * 4);
        uint32_t n_kv = pos_sparse + 1, n_bid = n_kv >> 2, nb = n_bid + 1;
        std::vector<float> refSc(nb);
        for (uint32_t b = 0; b < nb; ++b) {
            const float* kp = (b < n_bid) ? &hPool[b * ID] : hDead.data();
            float s = 0.0f;
            for (uint32_t h = 0; h < IH; ++h) {
                float d = 0.0f;
                for (uint32_t i = 0; i < ID; ++i) d += hQi[h * ID + i] * kp[i];
                if (d > 0.0f) s += d;
            }
            if (b == n_bid && (n_kv & 3u) != 0) s += 1e9f;
            refSc[b] = s;
        }
        uint32_t prefix = 0, pmask = 0, need = 2051;
        for (int pass = 0; pass < 4; ++pass) {
            int shift = 24 - 8 * pass;
            uint32_t hist[256] = {};
            for (uint32_t b = 0; b < nb; ++b) {
                uint32_t k = float_okey_cpu(refSc[b]);
                if ((k & pmask) == prefix) hist[(k >> shift) & 255u] += (b < n_bid) ? 4u : (n_kv & 3u);
            }
            uint32_t acc = 0, d = 0;
            for (int v = 255; v >= 0; --v) {
                if (acc + hist[v] >= need) { d = (uint32_t)v; break; }
                acc += hist[v];
            }
            prefix |= (d << shift); pmask |= (255u << shift); need -= acc;
        }
        std::vector<int32_t> refIds;
        uint32_t eq_left = need;
        for (uint32_t b = 0; b < nb; ++b) {
            uint32_t k = float_okey_cpu(refSc[b]);
            uint32_t cells = (b < n_bid) ? 4u : (n_kv & 3u);
            uint32_t take = (k > prefix) ? cells : ((k == prefix && eq_left > 0) ? std::min(cells, eq_left) : 0u);
            if (k == prefix && eq_left >= take) eq_left -= take;
            for (uint32_t c = 0; c < take; ++c) refIds.push_back((int32_t)(4u * b + c));
        }
        bool ids_ok = (refIds.size() == 2051) && std::equal(refIds.begin(), refIds.end(), gotIds.begin());
        std::printf("[5] qsa (prep+select+attn): xq=%.2e radix_2051=%s %s\n",
                    mq.max_abs, ids_ok ? "EXACT" : "MISMATCH", (mq.max_abs < 2e-3f && ids_ok) ? "PASS" : "FAIL");
        if (mq.max_abs >= 2e-3f || !ids_ok) return 1;
    }

    // ------------------------------------------------------------------------
    // 6. MoE Router + Planar Q2_0 Routed Experts (gu + down) + Combine
    // ------------------------------------------------------------------------
    {
        const uint32_t T = 3, NEXP = 32, N = 2560, NFF = 640, INSTR = 1808, TOPK = 10;
        std::vector<uint16_t> hMoeIn(T * INSTR), hYsh(T * N);
        std::vector<float> hXf(T * N);
        for (auto& v : hMoeIn) v = ref::f32_to_f16(rng.sym() * 1.2f);
        for (auto& v : hYsh)   v = ref::f32_to_f16(rng.sym() * 0.4f);
        for (auto& v : hXf)    v = rng.sym() * 0.5f;

        // Generate real GGUF Q2_0 blocks for NEXP experts (gate, up, down) and split into planar qs + d
        const size_t gu_row_bytes = (N / 64) * 18;    // 40 * 18 = 720 B
        const size_t dn_row_bytes = (NFF / 64) * 18;  // 10 * 18 = 180 B
        std::vector<uint8_t> raw_gate(NEXP * NFF * gu_row_bytes), raw_up(NEXP * NFF * gu_row_bytes), raw_down(NEXP * N * dn_row_bytes);
        ref::random_blocks(ref::Q2_0, raw_gate.data(), NEXP * NFF, N, 101ULL, 0.04f);
        ref::random_blocks(ref::Q2_0, raw_up.data(),   NEXP * NFF, N, 202ULL, 0.04f);
        ref::random_blocks(ref::Q2_0, raw_down.data(), NEXP * N, NFF, 303ULL, 0.04f);

        std::vector<uint8_t> p_gu_qs(NEXP * 2 * NFF * 640);
        std::vector<uint16_t> p_gu_d(NEXP * 2 * NFF * 40);
        for (uint32_t e = 0; e < NEXP; ++e) {
            for (uint32_t mat = 0; mat < 2; ++mat) {
                const uint8_t* src = (mat == 0 ? raw_gate.data() : raw_up.data()) + (size_t)e * NFF * gu_row_bytes;
                for (uint32_t r = 0; r < NFF; ++r) {
                    const uint8_t* row = src + r * gu_row_bytes;
                    uint8_t* dqs = p_gu_qs.data() + (((size_t)e * 2 + mat) * NFF + r) * 640;
                    uint16_t* dd = p_gu_d.data()  + (((size_t)e * 2 + mat) * NFF + r) * 40;
                    for (uint32_t b = 0; b < 40; ++b) {
                        std::memcpy(&dd[b], row + b * 18, 2);
                        std::memcpy(dqs + b * 16, row + b * 18 + 2, 16);
                    }
                }
            }
        }

        std::vector<uint8_t> p_dn_qs(NEXP * N * 160);
        std::vector<uint16_t> p_dn_d(NEXP * N * 10);
        for (uint32_t e = 0; e < NEXP; ++e) {
            const uint8_t* src = raw_down.data() + (size_t)e * N * dn_row_bytes;
            for (uint32_t r = 0; r < N; ++r) {
                const uint8_t* row = src + r * dn_row_bytes;
                uint8_t* dqs = p_dn_qs.data() + ((size_t)e * N + r) * 160;
                uint16_t* dd = p_dn_d.data()  + ((size_t)e * N + r) * 10;
                for (uint32_t b = 0; b < 10; ++b) {
                    std::memcpy(&dd[b], row + b * 18, 2);
                    std::memcpy(dqs + b * 16, row + b * 18 + 2, 16);
                }
            }
        }

        DevBuf dMoeIn(T * INSTR * 2), dIds(T * 16 * 4), dWts(T * 16 * 4), dPlan(strata::ascend_ops::P_INTS * 4);
        DevBuf dShH(T * NFF * 2), dSg(8 * 4), dXf(T * N * 4);
        DevBuf dGuQs(p_gu_qs.size()), dGuD(p_gu_d.size() * 2), dDnQs(p_dn_qs.size()), dDnD(p_dn_d.size() * 2);
        DevBuf dH(128 * NFF * 4), dYe(128 * N * 4), dYsh(T * N * 2), dBo(T * N * 4);

        dMoeIn.upload(hMoeIn.data(), hMoeIn.size() * 2);
        dXf.upload(hXf.data(), hXf.size() * 4);
        dGuQs.upload(p_gu_qs.data(), p_gu_qs.size());
        dGuD.upload(p_gu_d.data(), p_gu_d.size() * 2);
        dDnQs.upload(p_dn_qs.data(), p_dn_qs.size());
        dDnD.upload(p_dn_d.data(), p_dn_d.size() * 2);
        dYsh.upload(hYsh.data(), hYsh.size() * 2);

        strata::ascend_ops::launch_moe_route_shexp(stream, dMoeIn.ptr, dIds.ptr, dWts.ptr, dPlan.ptr, dShH.ptr, dSg.ptr, T, NEXP);
        strata::ascend_ops::launch_moe_q2_gu(stream, dGuQs.ptr, dGuD.ptr, dXf.ptr, dPlan.ptr, dH.ptr, T);
        strata::ascend_ops::launch_moe_q2_down(stream, dDnQs.ptr, dDnD.ptr, dH.ptr, dPlan.ptr, dYe.ptr);
        strata::ascend_ops::launch_moe_combine(stream, dYe.ptr, dPlan.ptr, dYsh.ptr, dSg.ptr, dBo.ptr, T);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        std::vector<float> gotBo(T * N), refBo(T * N, 0.0f);
        dBo.download(gotBo.data(), T * N * 4);

        // CPU golden reference using ref::dequant(ref::Q2_0, ...)
        std::vector<float> wG(NFF * N), wU(NFF * N), wD(N * NFF);
        for (uint32_t t = 0; t < T; ++t) {
            const uint16_t* min_row = hMoeIn.data() + t * INSTR;
            std::vector<float> probs(NEXP);
            float mx = -1e30f;
            for (uint32_t e = 0; e < NEXP; ++e) { probs[e] = ref::f16_to_f32(min_row[e]); mx = std::max(mx, probs[e]); }
            float sum = 0.0f;
            for (uint32_t e = 0; e < NEXP; ++e) { probs[e] = std::exp(probs[e] - mx); sum += probs[e]; }
            for (uint32_t e = 0; e < NEXP; ++e) probs[e] /= sum;

            uint32_t top_e[TOPK]; float top_w[TOPK]; float ssum = 0.0f;
            for (uint32_t r = 0; r < TOPK; ++r) {
                uint32_t be = 0; float bv = -1e30f;
                for (uint32_t e = 0; e < NEXP; ++e) if (probs[e] > bv) { bv = probs[e]; be = e; }
                probs[be] = -1e30f;
                top_e[r] = be; top_w[r] = bv; ssum += bv;
            }
            float denom = std::max(ssum, 6.103515625e-05f);
            for (uint32_t r = 0; r < TOPK; ++r) top_w[r] /= denom;

            float gs = sigmoid_f(ref::f16_to_f32(min_row[512 + 2 * NFF]));
            for (uint32_t d = 0; d < N; ++d) refBo[t * N + d] = ref::f16_to_f32(hYsh[t * N + d]) * gs;

            const float* x = hXf.data() + t * N;
            for (uint32_t r = 0; r < TOPK; ++r) {
                uint32_t e = top_e[r];
                ref::dequant(ref::Q2_0, raw_gate.data() + (size_t)e * NFF * gu_row_bytes, wG.data(), NFF * N);
                ref::dequant(ref::Q2_0, raw_up.data()   + (size_t)e * NFF * gu_row_bytes, wU.data(), NFF * N);
                ref::dequant(ref::Q2_0, raw_down.data() + (size_t)e * N * dn_row_bytes,   wD.data(), N * NFF);
                float h[NFF];
                for (uint32_t i = 0; i < NFF; ++i) {
                    float g = 0.0f, u = 0.0f;
                    for (uint32_t k = 0; k < N; ++k) { g += wG[i * N + k] * x[k]; u += wU[i * N + k] * x[k]; }
                    h[i] = silu_f(g) * u;
                }
                for (uint32_t d = 0; d < N; ++d) {
                    float y = 0.0f;
                    for (uint32_t i = 0; i < NFF; ++i) y += wD[d * NFF + i] * h[i];
                    refBo[t * N + d] += y * top_w[r];
                }
            }
        }
        auto mm = compare_f32(gotBo.data(), refBo.data(), T * N);
        std::printf("[6] moe (route+q2_gu+q2_down+combine): max_abs=%.3e rmse=%.3e %s\n",
                    mm.max_abs, mm.mse, mm.max_abs < 2e-3f ? "PASS" : "FAIL");
        if (mm.max_abs >= 2e-3f) return 1;
    }

    // ------------------------------------------------------------------------
    // 7. Argmax over FP16 [4, 248320] vocabulary logits
    // ------------------------------------------------------------------------
    {
        const uint32_t T = 4, VOCAB = 248320;
        std::vector<uint16_t> hLog(T * VOCAB);
        for (auto& v : hLog) v = ref::f32_to_f16(rng.sym() * 4.0f);
        int32_t expected[4] = {17, 65432, 199999, 248319};
        for (uint32_t t = 0; t < T; ++t) hLog[t * VOCAB + expected[t]] = ref::f32_to_f16(12.5f + (float)t);

        DevBuf dLog(T * VOCAB * 2), dPv(T * 64 * 4), dPi(T * 64 * 4), dOut(8 * 4);
        dLog.upload(hLog.data(), hLog.size() * 2);
        strata::ascend_ops::launch_argmax_f16(stream, dLog.ptr, dPv.ptr, dPi.ptr, dOut.ptr, T, VOCAB);
        ACL_CHECK(aclrtSynchronizeStream(stream));

        int32_t got[8] = {};
        dOut.download(got, 8 * 4);
        bool ok = true;
        for (uint32_t t = 0; t < T; ++t) if (got[t] != expected[t]) ok = false;
        std::printf("[7] argmax_f16 [4, 248320]: got=[%d,%d,%d,%d] %s\n",
                    got[0], got[1], got[2], got[3], ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    std::printf("ALL 7 ASCEND PARITY SUITES PASSED!\n");
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
