// ascend/src/kernels/ascend_ops.h - Host launch declarations for Strata Ascend C (dav-m200) kernels.
#pragma once

#include <cstdint>

namespace strata::ascend_ops {

// MoePlan layout constants (matches metal/kernels/moe.metal P_* layout, padded to 8-int / 32-byte boundaries)
inline constexpr int P_NG = 0;
inline constexpr int P_IGU = 4;
inline constexpr int P_IDN = 8;
inline constexpr int P_MAXE = 128;
inline constexpr int P_GEXP = 16;
inline constexpr int P_GSTART = P_GEXP + P_MAXE;        // 144 (multiple of 8)
inline constexpr int P_ETOK = P_GSTART + P_MAXE + 8;    // 280 (multiple of 8)
inline constexpr int P_EW = P_ETOK + P_MAXE;            // 408 (multiple of 8)
inline constexpr int P_TOKENT = P_EW + P_MAXE;          // 536 (multiple of 8)
inline constexpr int P_INTS = P_TOKENT + P_MAXE + 8;    // 672 (multiple of 8 = 2688 bytes)

void launch_f32_to_f16(void* stream, const void* src_f32, void* dst_f16, uint32_t n);
void launch_f16_to_f32(void* stream, const void* src_f16, void* dst_f32, uint32_t n);

// Hyper-connection (gated residual) pre-norm + residual apply:
// R: [n_tok, 4, 2560] f32 (in/out if apply!=0), bo: [n_tok, 2560] f32, inj: [n_tok, 4] f32,
// w_norm: [4, 2560] f32, xn_f32: [n_tok, 4, 2560] f32, xn_f16: [n_tok, 10240] f16
void launch_gr_norm(void* stream, void* R, const void* bo, const void* inj, const void* w_norm,
                    void* xn_f32, void* xn_f16, uint32_t n_tok, uint32_t apply, float eps);

// Post-processing of [n_tok, 336] f16 output from W_hc_down_inj Cube matmul:
// cols 0..319 -> lo_f16 [n_tok, 320] = half(silu(v / 4)); cols 320..323 -> inj_next_f32 [n_tok, 4]
void launch_gr_post_down(void* stream, const void* down_f16, void* lo_f16, void* inj_next_f32, uint32_t n_tok);

// Post-processing of [n_tok, 10240] f16 output from W_hc_up Cube matmul:
// out[t, d] = 0.25f * sum_{c=0..3} xn_f32[t, c, d] * sigmoid(float(gate_f16[t, c, d]))
// Writes both out_f16 [n_tok, 2560] and out_f32 [n_tok, 2560].
void launch_gr_post_up(void* stream, const void* xn_f32, const void* gate_f16, void* out_f16, void* out_f32, uint32_t n_tok);

// Fused GDN (Gated DeltaNet) recurrence step across 48 value heads (16 q/k heads, state [48, 128, 128] f32,
// conv hist [10240, 4] f32 with [0..2] active taps for 16B/32B alignment):
// gdn_in_f16: [n_tok, 16480] f16 ([qkv 10240 | z 6144 | alpha 48 | beta 48]) when save_inputs!=0, also saves
// unpacked qkv_save [n_tok, 10240] f32, z_save [n_tok, 6144] f32, ab_save [n_tok, 96] f32 for speculative fold,
// OR reads from saved f32 arrays when from_saved!=0.
void launch_gdn_step(void* stream, void* state_f32, const void* hist_in_f32, void* hist_out_f32,
                     const void* gdn_in_f16, void* qkv_save_f32, void* z_save_f32, void* ab_save_f32,
                     const void* conv_w_f32, const void* dt_f32, const void* ssm_a_f32, const void* gamma_f32,
                     void* yg_f16, uint32_t n_tok, uint32_t commit_at, uint32_t from_saved, float eps);

// QSA (Query-Selected Sparse Attention):
// 1. qsa_prep: qsa_in_f16 [n_tok, 13952] ([qfull 12288 | kcur 512 | vcur 512 | iq 512 | ik 128])
//    -> qcur_f32 [n_tok, 24, 256], qgate_f32 [n_tok, 24, 256], qidx_f32 [n_tok, 4, 128],
//       kq_i8 [max_cells, 2, 256], vq_i8 [max_cells, 2, 256],
//       ksc_f16 [max_cells, 16] (first 8 used: 2 kv_heads x 4 groups, padded to 16 halfs = 32B!),
//       vsc_f16 [max_cells, 16], tail_f32 [4, 128], dead_f32 [128], pooled_f32 [max_blocks, 128]
void launch_qsa_prep(void* stream, const void* qsa_in_f16,
                     const void* qn_f32, const void* kn_f32, const void* iqn_f32, const void* ikn_f32,
                     const void* rope_cs_f32,
                     void* qcur_f32, void* qgate_f32, void* qidx_f32,
                     void* kq_i8, void* vq_i8, void* ksc_f16, void* vsc_f16,
                     void* tail_f32, void* dead_f32, void* pooled_f32,
                     uint32_t pos0, uint32_t pos_base, uint32_t n_tok, float eps);

// 2. qsa_select: scores blocks 0..n_bid and selects min(n_kv, 2051) cell IDs ascending into qids_i32 [n_tok, 2056]
void launch_qsa_select(void* stream, const void* qidx_f32, const void* pooled_f32, const void* dead_f32,
                       void* scores_f32, void* qids_i32,
                       uint32_t pos0, uint32_t max_blocks_stride, uint32_t n_tok);

// 3. qsa_attn_merge: 8 AI Cores (4 per KV head, 3 query heads per AI Core = 24 heads) online softmax over selected
//    int8 KV cells + * sigmoid(qgate) -> xq_f16 [n_tok, 6144]
void launch_qsa_attn_merge(void* stream, const void* qcur_f32, const void* qgate_f32,
                           const void* kq_i8, const void* vq_i8, const void* ksc_f16, const void* vsc_f16,
                           const void* qids_i32, void* xq_f16,
                           uint32_t pos0, uint32_t n_tok);

// MoE routing + shared expert SwiGLU:
// moe_in_f16: [n_tok, 1808] ([logits 512 | gsh 640 | ush 640 | sg 16])
// -> ids_i32 [n_tok, 16] (first 10 used), wts_f32 [n_tok, 16], plan_i32 [P_INTS],
//    sh_h_f16 [n_tok, 640], sg_f32 [max_tok_aligned]
void launch_moe_route_shexp(void* stream, const void* moe_in_f16,
                            void* ids_i32, void* wts_f32, void* plan_i32,
                            void* sh_h_f16, void* sg_f32,
                            uint32_t n_tok, uint32_t n_expert);

// Planar Q2_0 MoE Experts on 8 AI Cores:
// Expert layout in GM:
//   gu_qs:   [n_expert, 2, 640, 640] uint8_t (gate=0, up=1; each row 2560 weights -> 40 blocks x 16B = 640B)
//   gu_d:    [n_expert, 2, 640, 40]  half    (each 16 rows = 640 halfs = 1280B, 32B-aligned!)
//   down_qs: [n_expert, 2560, 160]   uint8_t (each row 640 weights -> 10 blocks x 16B = 160B)
//   down_d:  [n_expert, 2560, 10]    half    (each 16 rows = 160 halfs = 320B, 32B-aligned!)
// xf_f32: [n_tok, 2560] f32 -> h_f32: [max_entries, 640] f32 -> ye_f32: [max_entries, 2560] f32
void launch_moe_q2_gu(void* stream, const void* gu_qs, const void* gu_d,
                      const void* xf_f32, const void* plan_i32, void* h_f32, uint32_t n_tok);

void launch_moe_q2_down(void* stream, const void* down_qs, const void* down_d,
                        const void* h_f32, const void* plan_i32, void* ye_f32);

// Combine 10 routed expert outputs + gated shared expert output into bo_f32 [n_tok, 2560]:
void launch_moe_combine(void* stream, const void* ye_f32, const void* plan_i32,
                        const void* ysh_f16, const void* sg_f32, void* bo_f32, uint32_t n_tok);

// Argmax over vocabulary logits_f16 [n_tok, n_vocab] -> tok_out_i32 [8] (32B aligned):
void launch_argmax_f16(void* stream, const void* logits_f16, void* part_val_f32, void* part_idx_i32,
                       void* tok_out_i32, uint32_t n_tok, uint32_t n_vocab);

}  // namespace strata::ascend_ops
