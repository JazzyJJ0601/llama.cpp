#include "llama-graph.h"

#include "helix-paging.h"

#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <atomic>

// Helix Doppelgänger runtime statistics
struct helix_dg_stats {
    int layers_magnet = 0;
    int layers_sparse = 0;
    int layers_dense  = 0;
    bool printed = false;
};

static helix_dg_stats g_helix_stats;

static constexpr int HELIX_SPARSITY_N_LAYERS = 28;

struct helix_sparsity_runtime {
    uint64_t active_neurons = 0;
    uint64_t total_neurons  = 0;
    uint64_t mask_samples   = 0;
    uint64_t decode_active  = 0;
    uint64_t decode_total   = 0;
    uint64_t decode_mask_samples = 0;
    bool layer_magnet[HELIX_SPARSITY_N_LAYERS] = {};
    bool layer_gate[HELIX_SPARSITY_N_LAYERS]   = {};
};

static double helix_magnet_active_budget_pct() {
    // L3-L12: top-25% of FFN; L13-L27: top-20%
    return (10.0 * 25.0 + 15.0 * 20.0) / 25.0;
}

static ggml_tensor * helix_sparsity_anchor_tensor(
        ggml_context * ctx0,
        ggml_tensor * out,
        ggml_tensor * dep) {
    if (out == nullptr || dep == nullptr) {
        return out;
    }
    ggml_tensor * zero = ggml_scale(ctx0, ggml_sum(ctx0, dep), 0.0f);
    if (zero->type != out->type) {
        zero = ggml_cast(ctx0, zero, out->type);
    }
    return ggml_add(ctx0, out, ggml_repeat(ctx0, zero, out));
}

static helix_sparsity_runtime g_helix_sparsity;
static thread_local std::vector<uint8_t> g_helix_mask_buf;
static thread_local int64_t g_helix_layer_n_ff[HELIX_SPARSITY_N_LAYERS] = {};

static bool helix_env_flag_active(const char * name);

static float helix_magnet_env_width_frac() {
    const char * v = std::getenv("HELIX_MAGNET_WIDTH_FRAC");
    if (v == nullptr || v[0] == '\0') {
        v = std::getenv("HELIX_MAGNET_ACTIVE_FRAC");
    }
    if (v == nullptr || v[0] == '\0') {
        return -1.0f;
    }
    return (float) atof(v);
}

void helix_sparsity_reset() {
    g_helix_sparsity = {};
}

static void helix_sparsity_accumulate_mask(ggml_tensor * t, int il, bool is_magnet, bool count_nonzero) {
    if (il < 3 || il >= HELIX_SPARSITY_N_LAYERS || t == nullptr) {
        return;
    }

    const int64_t n_neurons = t->ne[0];
    const int64_t n_tokens  = t->ne[1] > 0 ? t->ne[1] : 1;
    if (n_neurons <= 0) {
        return;
    }

    const size_t row_bytes = (size_t) n_neurons * ggml_element_size(t);
    const size_t need      = row_bytes * (size_t) n_tokens;
    g_helix_mask_buf.resize(need);

    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(g_helix_mask_buf.data(), t->data, need);
    } else {
        ggml_backend_tensor_get(t, g_helix_mask_buf.data(), 0, need);
    }

    const int64_t tok = n_tokens - 1;
    const uint8_t * row = g_helix_mask_buf.data() + (size_t) tok * row_bytes;

    const float thresh = count_nonzero ? 1e-6f : 0.5f;
    int64_t active = 0;
    if (t->type == GGML_TYPE_F32) {
        const float * frow = (const float *) row;
        for (int64_t i = 0; i < n_neurons; ++i) {
            if (frow[i] > thresh) {
                ++active;
            }
        }
    } else {
        for (int64_t i = 0; i < n_neurons; ++i) {
            if (row[i] != 0) {
                ++active;
            }
        }
    }

    g_helix_sparsity.active_neurons += (uint64_t) active;
    g_helix_sparsity.total_neurons  += (uint64_t) n_neurons;
    ++g_helix_sparsity.mask_samples;
    if (n_tokens == 1) {
        g_helix_sparsity.decode_active += (uint64_t) active;
        g_helix_sparsity.decode_total  += (uint64_t) n_neurons;
        ++g_helix_sparsity.decode_mask_samples;
    }

    if (is_magnet) {
        g_helix_sparsity.layer_magnet[il] = true;
    } else {
        g_helix_sparsity.layer_gate[il] = true;
    }
}

bool helix_sparsity_get(struct helix_sparsity_stats * out) {
    if (out == nullptr) {
        return false;
    }

    int n_magnet = 0;
    int n_gate   = 0;
    for (int il = 3; il < HELIX_SPARSITY_N_LAYERS; ++il) {
        if (g_helix_sparsity.layer_magnet[il]) {
            ++n_magnet;
        } else if (g_helix_sparsity.layer_gate[il]) {
            ++n_gate;
        }
    }

    if (n_magnet == 0 && g_helix_stats.layers_magnet > 0) {
        n_magnet = g_helix_stats.layers_magnet;
    }
    if (n_gate == 0 && g_helix_stats.layers_sparse > 0) {
        n_gate = g_helix_stats.layers_sparse;
    }

    const bool engine_active = helix_env_flag_active("HELIX_DOPPELGANGER") &&
        (n_magnet > 0 || n_gate > 0 || g_helix_sparsity.mask_samples > 0 || g_helix_stats.printed);

    out->n_dense_layers    = 3;
    out->n_magnet_layers   = n_magnet;
    out->n_gate_layers     = n_gate;
    out->engine_active     = engine_active;
    out->mask_reads = g_helix_sparsity.mask_samples;
    out->decode_mask_reads = g_helix_sparsity.decode_mask_samples;
    out->active_neuron_measured = g_helix_sparsity.mask_samples > 0;

    if (g_helix_sparsity.decode_total > 0) {
        out->active_neuron_decode_pct = 100.0 * (double) g_helix_sparsity.decode_active /
            (double) g_helix_sparsity.decode_total;
    } else {
        out->active_neuron_decode_pct = 0.0;
    }

    if (n_magnet > 0) {
        const float wf = helix_magnet_env_width_frac();
        if (wf > 0.0f && wf <= 1.0f) {
            out->active_budget_pct = 100.0 * (double) wf;
        } else {
            out->active_budget_pct = helix_magnet_active_budget_pct();
        }
    } else if (n_gate > 0) {
        out->active_budget_pct = 35.0;
    } else {
        out->active_budget_pct = 0.0;
    }

    if (g_helix_sparsity.total_neurons > 0) {
        out->active_neuron_pct = 100.0 * (double) g_helix_sparsity.active_neurons /
            (double) g_helix_sparsity.total_neurons;
    } else {
        out->active_neuron_pct = 0.0;
    }

    const int n_sparse = n_magnet + n_gate;
    const int n_total  = out->n_dense_layers + n_sparse;
    const double sparse_coverage = n_total > 0
        ? 100.0 * (double) n_sparse / (double) n_total
        : 0.0;

    if (out->active_neuron_measured) {
        out->ffn_flop_saved_pct = (100.0 - out->active_neuron_pct) * sparse_coverage / 100.0;
    } else if (n_magnet > 0 || n_gate > 0) {
        out->ffn_flop_saved_pct = (100.0 - out->active_budget_pct) * sparse_coverage / 100.0;
    } else {
        out->ffn_flop_saved_pct = 0.0;
    }

    return engine_active;
}

void llama_helix_sparsity_reset(void) {
    helix_sparsity_reset();
}

bool llama_helix_sparsity_get(struct helix_sparsity_stats * out) {
    return helix_sparsity_get(out);
}

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d   (ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swq_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);

        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

        res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);
    }

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    helix_trace_set_ubatch_tokens(ubatch);
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs() {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_pre_norm != nullptr) {
        ggml_set_output(t_h_pre_norm);
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return { Qcur, Kcur, Vcur };
}


static std::vector<int32_t> helix_trace_token_ids;
static uint32_t helix_trace_batch_n_tokens = 0;

static constexpr uint32_t HELIX_ACT_N_EMBD   = 2048;
static constexpr uint32_t HELIX_ACT_N_LAYERS = 28;

static bool helix_env_flag_active(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    // Windows `set HELIX_TRACE=0` leaves the literal string "0" — treat as off.
    if (v[0] == '0' && v[1] == '\0') {
        return false;
    }
    if ((v[0] == 'f' || v[0] == 'F') && (strcmp(v + 1, "alse") == 0 || strcmp(v + 1, "ALSE") == 0)) {
        return false;
    }
    if ((v[0] == 'o' || v[0] == 'O') && (strcmp(v + 1, "ff") == 0 || strcmp(v + 1, "FF") == 0)) {
        return false;
    }
    if ((v[0] == 'n' || v[0] == 'N') && (strcmp(v + 1, "o") == 0 || strcmp(v + 1, "O") == 0)) {
        return false;
    }
    return true;
}

bool helix_env_flag_enabled(const char * name) {
    return helix_env_flag_active(name);
}

static bool helix_doppelganger_enabled() {
    return helix_env_flag_active("HELIX_DOPPELGANGER");
}

static float helix_magnet_layer_target_frac(int il) {
    const float frac_override = helix_magnet_env_width_frac();
    if (frac_override > 0.0f && frac_override <= 1.0f) {
        return frac_override;
    }
    const char * target = std::getenv("HELIX_MAGNET_TARGET_FRAC");
    if (target != nullptr && target[0] != '\0') {
        const float f = (float) atof(target);
        if (f > 0.0f && f <= 1.0f) {
            return f;
        }
    }
    return (il < 13) ? 0.25f : 0.20f;
}

static float helix_magnet_threshold_scale(int il) {
    // Calibrated so mean(|magnet|) * scale ≈ target active fraction (see train_magnet_field).
    return 0.68f * helix_magnet_layer_target_frac(il) / 0.22f;
}

static int64_t helix_magnet_active_k(int il, int64_t n_ff) {
    const float frac = helix_magnet_layer_target_frac(il);
    int64_t k = (int64_t) (frac * (float) n_ff + 0.5f);
    if (k < 1) {
        k = 1;
    }
    if (k > n_ff) {
        k = n_ff;
    }
    const char * vmin = std::getenv("HELIX_MAGNET_MIN_FRAC");
    const char * vmax = std::getenv("HELIX_MAGNET_MAX_FRAC");
    if (vmin != nullptr && vmin[0] != '\0') {
        const int64_t k_min = (int64_t) (atof(vmin) * (float) n_ff + 0.5f);
        if (k < k_min) {
            k = k_min;
        }
    }
    if (vmax != nullptr && vmax[0] != '\0') {
        const int64_t k_max = (int64_t) (atof(vmax) * (float) n_ff + 0.5f);
        if (k > k_max) {
            k = k_max;
        }
    }
    return k;
}

static bool helix_magnet_dynamic_enabled() {
    return helix_env_flag_active("HELIX_MAGNET_DYNAMIC");
}

static bool helix_magnet_staged_enabled() {
    return helix_env_flag_active("HELIX_MAGNET_STAGED");
}

static bool helix_magnet_sparse_mode_enabled() {
    if (!helix_doppelganger_enabled()) {
        return false;
    }
    if (helix_env_flag_active("HELIX_MAGNET_DENSE")) {
        return false;
    }
    return helix_env_flag_active("HELIX_MAGNET_SPARSE");
}

static bool helix_magnet_row_gather_enabled() {
    if (!helix_magnet_sparse_mode_enabled()) {
        return false;
    }
    return helix_env_flag_active("HELIX_MAGNET_GATHER") || helix_magnet_staged_enabled();
}

// Phase 2 demand paging: CPU-backed FFN + async row gather into VRAM scratchpad (see HELIX_DOPPELGANGER_PAGING.md).
static bool helix_magnet_paged_enabled() {
    return helix_doppelganger_enabled() && helix_env_flag_active("HELIX_MAGNET_PAGED");
}

static ggml_tensor * helix_sum_dim1_rows(
        ggml_context * ctx0,
        ggml_tensor * down_out,
        int64_t        n_embd_cur,
        int64_t        n_ffn_tokens) {
    // down_out [n_embd, k, n_tokens] -> sum over k in one op (avoid O(k) add nodes in graph)
    ggml_tensor * perm = ggml_cont(ctx0, ggml_permute(ctx0, down_out, 1, 0, 2, 3));
    ggml_tensor * sum  = ggml_sum_rows(ctx0, perm);
    return ggml_cont(ctx0, ggml_reshape_2d(ctx0, sum, n_embd_cur, n_ffn_tokens));
}

// Variable-width mask from magnet |scores| (per forward pass — ~8% on easy prompts, ~20% on hard).
static ggml_tensor * helix_magnet_build_dynamic_mask(
        ggml_context * ctx0,
        ggml_tensor * scores,
        int           il) {
    const float scale = helix_magnet_threshold_scale(il);
    ggml_tensor * scores_abs  = ggml_abs(ctx0, scores);
    ggml_tensor * scores_mean = ggml_mean(ctx0, scores_abs);
    ggml_tensor * thresh      = ggml_scale(ctx0, ggml_repeat(ctx0, scores_mean, scores_abs), scale);
    ggml_tensor * above       = ggml_sub(ctx0, scores_abs, thresh);
    return ggml_step(ctx0, above);
}

// Binary mask [n_ff, n_tokens]: ~active_k rows on (score >= k-th largest per token).
static ggml_tensor * helix_magnet_build_topk_mask(
        ggml_context * ctx0,
        ggml_tensor * scores,
        int           active_k) {
    GGML_ASSERT(active_k >= 1);

    const int64_t n_ff     = scores->ne[0];
    const int64_t n_tokens = scores->ne[1] > 0 ? scores->ne[1] : 1;

    ggml_tensor * sorted = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, scores, active_k));

    ggml_tensor * scores_3d = ggml_reshape_3d(ctx0, scores, 1, n_ff, n_tokens);
    ggml_tensor * top_scores = ggml_get_rows(ctx0, scores_3d, sorted);
    // k-th largest score per token (last row of top-k score block)
    const size_t kth_off = (size_t) (active_k - 1) * top_scores->nb[1];
    ggml_tensor * kth_score = ggml_view_2d(ctx0, top_scores, 1, n_tokens, top_scores->nb[2], kth_off);
    ggml_tensor * thresh = ggml_repeat(ctx0, kth_score, scores);
    ggml_tensor * above = ggml_sub(ctx0, scores, thresh);
    return ggml_step(ctx0, above);
}

// GGUF stores magnet A/B with reversed dims vs ggml_mul_mat (see embed_magnet_gguf.py ti_shape).
static ggml_tensor * helix_magnet_view_for_mul_mat(
        ggml_context * ctx0,
        ggml_tensor * weight,
        int64_t       rows,
        int64_t       cols) {
    ggml_tensor * w = ggml_cont(ctx0, ggml_cast(ctx0, weight, GGML_TYPE_F32));
    if (w->ne[0] == rows && w->ne[1] == cols) {
        return w;
    }
    if (w->ne[0] == cols && w->ne[1] == rows) {
        return ggml_view_2d(ctx0, w, rows, cols, cols * ggml_element_size(w), 0);
    }
    return w;
}

static bool helix_trace_enabled() {
    return helix_env_flag_active("HELIX_TRACE");
}

static bool helix_act_cache_enabled() {
    return helix_env_flag_active("HELIX_ACTIVATION_CACHE");
}

static const char * helix_act_cache_path() {
    const char * path = std::getenv("HELIX_ACTIVATION_CACHE");
    if (path != nullptr && path[0] != '\0' &&
        !(path[0] == '1' && path[1] == '\0')) {
        return path;
    }
    return "F:/NewAI/models/helix_official/native_activation_cache.bin";
}

#pragma pack(push, 1)
struct helix_act_header {
    char     magic[8];
    uint32_t version;
    uint32_t n_embd;
    uint32_t n_layers;
    uint64_t reserved;
};

struct helix_act_record_hdr {
    uint32_t layer_idx;
    uint32_t batch_n_tokens;
    uint32_t token_pos;
    int32_t  token_id;
};
#pragma pack(pop)

static void helix_trace_clear_stale_cache_once() {
    static bool cleared = false;
    if (cleared || !helix_trace_enabled()) {
        return;
    }
    cleared = true;
    const char * cache_path = helix_act_cache_path();
    if (remove(cache_path) == 0) {
        LLAMA_LOG("[HELIX] Cleared stale activation cache\n");
    }
}

static FILE * helix_act_cache_fp() {
    static FILE * fp = nullptr;
    static bool   header_written = false;
    if (!helix_act_cache_enabled()) {
        return nullptr;
    }
    if (fp == nullptr) {
        helix_trace_clear_stale_cache_once();
        const char * path = helix_act_cache_path();
        const bool truncate = helix_env_flag_active("HELIX_ACTIVATION_CACHE_TRUNCATE") || helix_trace_enabled();
        fp = fopen(path, truncate ? "wb" : "ab");
        if (fp == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to open HELIX_ACTIVATION_CACHE '%s'\n", __func__, path);
            return nullptr;
        }
        setvbuf(fp, nullptr, _IONBF, 0);
        if (truncate || ftell(fp) == 0) {
            helix_act_header hdr {};
            memcpy(hdr.magic, "HELIXAC1", 8);
            hdr.version  = 1;
            hdr.n_embd   = HELIX_ACT_N_EMBD;
            hdr.n_layers = HELIX_ACT_N_LAYERS;
            hdr.reserved = 0;
            if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
                LLAMA_LOG_ERROR("%s: failed to write activation cache header\n", __func__);
            } else {
                header_written = true;
                LLAMA_LOG_INFO("%s: writing native activation cache -> %s\n", __func__, path);
            }
        } else {
            header_written = true;
        }
    }
    (void) header_written;
    return fp;
}

static void helix_act_cache_write_gate_hidden(ggml_tensor * gate_scores, int il) {
    if (gate_scores == nullptr || gate_scores->src[1] == nullptr) {
        return;
    }
    if (il < 0 || il >= (int) HELIX_ACT_N_LAYERS) {
        return;
    }
    // Skip single-token decode steps; keep multi-token prefill batches.
    if (helix_trace_batch_n_tokens <= 1) {
        return;
    }

    FILE * fp = helix_act_cache_fp();
    if (fp == nullptr) {
        return;
    }

    ggml_tensor * hidden = gate_scores->src[1];
    if (hidden->type != GGML_TYPE_F32) {
        return;
    }

    const int64_t n_embd   = hidden->ne[0];
    const int64_t n_tokens = hidden->ne[1];
    if (n_embd != (int64_t) HELIX_ACT_N_EMBD) {
        return;
    }

    std::vector<float> row(n_embd);
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        ggml_backend_tensor_get(
            hidden, row.data(), tok * n_embd * sizeof(float), n_embd * sizeof(float));

        helix_act_record_hdr rec {};
        rec.layer_idx       = (uint32_t) il;
        rec.batch_n_tokens  = helix_trace_batch_n_tokens;
        rec.token_pos       = (uint32_t) tok;
        rec.token_id        = (tok < (int64_t) helix_trace_token_ids.size())
            ? helix_trace_token_ids[tok] : 0;

        if (fwrite(&rec, sizeof(rec), 1, fp) != 1 ||
            fwrite(row.data(), sizeof(float), (size_t) n_embd, fp) != (size_t) n_embd) {
            LLAMA_LOG_ERROR("%s: failed to write activation record (layer=%d tok=%lld)\n",
                    __func__, il, (long long) tok);
            return;
        }
    }
}

static int64_t helix_trace_focus_token_index(int64_t n_tokens) {
    // Qwen3 chat-template prefill for "Hello" (--single-turn): 9 tokens, payload at index 3.
    if (n_tokens == 9) {
        return 3;
    }
    return 0;
}

static bool helix_trace_want_raw_logits(int il, int64_t tok, int64_t n_tokens) {
    if (il != 0 && il != 19) {
        return false;
    }
    if (n_tokens == 9) {
        return tok == helix_trace_focus_token_index(n_tokens);
    }
    return tok == 0 || tok == n_tokens - 1;
}

static FILE * helix_trace_file() {
    static FILE * fp = nullptr;
    if (fp == nullptr) {
        if (!helix_trace_enabled()) {
            return nullptr;
        }
        helix_trace_clear_stale_cache_once();
        fp = fopen("F:/NewAI/helix_trace_cpp.log", "a");
        if (fp != nullptr) {
            setvbuf(fp, nullptr, _IONBF, 0);
        }
    }
    return fp;
}

static bool helix_trace_layer_wanted(int il) {
    return il == 0 || il == 19;
}

#define HELIX_TRACE_FPRINTF(...) do { \
        if (helix_trace_file() != nullptr) { \
            fprintf(helix_trace_file(), __VA_ARGS__); \
            fflush(helix_trace_file()); \
        } \
    } while (0)

void helix_trace_set_ubatch_tokens(const llama_ubatch * ubatch) {
    helix_trace_token_ids.clear();
    helix_trace_batch_n_tokens = 0;
    if ((!helix_trace_enabled() && !helix_act_cache_enabled()) ||
        ubatch == nullptr || ubatch->token == nullptr) {
        return;
    }
    helix_trace_batch_n_tokens = ubatch->n_tokens;
    helix_trace_token_ids.assign(ubatch->token, ubatch->token + ubatch->n_tokens);
    if (helix_trace_file() == nullptr) {
        return;
    }
    HELIX_TRACE_FPRINTF("[HELIX TRACE] batch n_tokens=%u token_ids:", ubatch->n_tokens);
    for (int32_t tid : helix_trace_token_ids) {
        HELIX_TRACE_FPRINTF(" %d", tid);
    }
    if (ubatch->n_tokens == 9) {
        HELIX_TRACE_FPRINTF(" | focus_token_index=3 (chat-template payload)");
    }
    HELIX_TRACE_FPRINTF("\n");
}

static int helix_trace_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }
    const char * dash = strrchr(name, '-');
    if (dash == nullptr || dash[1] == '\0') {
        return -1;
    }
    return atoi(dash + 1);
}

static bool helix_trace_name_matches(const char * name, const char * prefix) {
    if (name == nullptr || prefix == nullptr) {
        return false;
    }
    const size_t n = strlen(prefix);
    return strncmp(name, prefix, n) == 0 && name[n] == '-';
}

static const ggml_tensor * helix_trace_gate_scores_from_node(const ggml_tensor * t) {
    const ggml_tensor * node = t;
    for (int depth = 0; depth < 4 && node != nullptr; ++depth) {
        if (helix_trace_name_matches(node->name, "helix_gate_scores")) {
            return node;
        }
        node = node->src[0];
    }
    const ggml_tensor * sorted = t && t->src[0] ? t->src[0] : nullptr;
    const ggml_tensor * probs  = sorted && sorted->src[0] ? sorted->src[0] : nullptr;
    return probs && probs->src[0] ? probs->src[0] : nullptr;
}

struct helix_blk_q8_0 {
    ggml_fp16_t d;
    int8_t      qs[32];
};

static void helix_trace_gate_topk_from_scores(
        const ggml_tensor * gate_scores,
        int                   top_k,
        int64_t               tok,
        int32_t             & out_a,
        int32_t             & out_b) {
    out_a = -1;
    out_b = -1;
    if (gate_scores == nullptr || gate_scores->type != GGML_TYPE_F32 || top_k <= 0) {
        return;
    }

    const int64_t num_clusters = gate_scores->ne[0];
    const int64_t n_tokens     = gate_scores->ne[1];
    if (tok < 0 || tok >= n_tokens) {
        return;
    }

    std::vector<float> row((size_t) num_clusters);
    ggml_backend_tensor_get(
        gate_scores,
        row.data(),
        (size_t) tok * (size_t) num_clusters * sizeof(float),
        (size_t) num_clusters * sizeof(float));

    float maxv = row[0];
    for (int64_t c = 1; c < num_clusters; ++c) {
        maxv = std::max(maxv, row[c]);
    }
    double sum = 0.0;
    std::vector<double> probs((size_t) num_clusters);
    for (int64_t c = 0; c < num_clusters; ++c) {
        probs[(size_t) c] = exp((double) row[(size_t) c] - (double) maxv);
        sum += probs[(size_t) c];
    }
    for (int64_t c = 0; c < num_clusters; ++c) {
        probs[(size_t) c] /= sum;
    }

    std::vector<int64_t> order((size_t) num_clusters);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
        [&](int64_t a, int64_t b) { return probs[(size_t) a] > probs[(size_t) b]; });

    if (top_k > 0) {
        out_a = (int32_t) order[0];
    }
    if (top_k > 1) {
        out_b = (int32_t) order[1];
    }
}

static void helix_trace_dump_mul_mat_id_ids(ggml_tensor * ids, int il, int top_k, bool at_eval) {
    if (ids == nullptr || ids->type != GGML_TYPE_I32) {
        return;
    }

    const int64_t n_ids = ggml_nelements(ids);
    LLAMA_LOG("[HELIX TRACE] Layer %d MulMatID Expert IDs%s:\n", il, at_eval ? " (eval)" : " (graph build)");
    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d MulMatID Expert IDs%s:\n", il, at_eval ? " (eval)" : " (graph build)");
    LLAMA_LOG("  matmul path: ggml_mul_mat_id via build_lora_mm_id (not ggml_view_2d weight slicing)\n");
    HELIX_TRACE_FPRINTF("  matmul path: ggml_mul_mat_id via build_lora_mm_id (not ggml_view_2d weight slicing)\n");
    LLAMA_LOG("  ids tensor nelements: %lld\n", (long long) n_ids);
    HELIX_TRACE_FPRINTF("  ids tensor nelements: %lld\n", (long long) n_ids);
    LLAMA_LOG("  ids tensor shape: [%lld, %lld]\n", (long long) ids->ne[0], (long long) ids->ne[1]);
    HELIX_TRACE_FPRINTF("  ids tensor shape: [%lld, %lld]\n", (long long) ids->ne[0], (long long) ids->ne[1]);

    if (!at_eval) {
        LLAMA_LOG("  ids->data at build time: %s (values populated at eval before mul_mat_id)\n",
                ids->data ? "non-null" : "null");
        HELIX_TRACE_FPRINTF("  ids->data at build time: %s (values populated at eval before mul_mat_id)\n",
                ids->data ? "non-null" : "null");
        return;
    }

    std::vector<int32_t> id_data((size_t) n_ids);
    ggml_backend_tensor_get(ids, id_data.data(), 0, (size_t) n_ids * sizeof(int32_t));

    const int64_t top_k_dim = ids->ne[0];
    const int64_t n_tokens  = ids->ne[1];
    for (int i = 0; i < (int) n_ids && i < 8; ++i) {
        LLAMA_LOG("  ids[%d] = %d\n", i, id_data[(size_t) i]);
        HELIX_TRACE_FPRINTF("  ids[%d] = %d\n", i, id_data[(size_t) i]);
    }

    const ggml_tensor * gate_scores = helix_trace_gate_scores_from_node(ids);
    const int64_t focus = helix_trace_focus_token_index(n_tokens);
    int32_t expected_a = -1;
    int32_t expected_b = -1;
    if (gate_scores != nullptr) {
        helix_trace_gate_topk_from_scores(gate_scores, top_k, focus, expected_a, expected_b);
    }

    const int32_t ids_a = (top_k_dim > 0 && focus >= 0 && focus < n_tokens)
        ? id_data[(size_t) (focus * top_k_dim + 0)] : -1;
    const int32_t ids_b = (top_k_dim > 1 && focus >= 0 && focus < n_tokens)
        ? id_data[(size_t) (focus * top_k_dim + 1)] : -1;

    LLAMA_LOG("  Expected cluster IDs from gate (token %lld): %d %d\n",
            (long long) focus, expected_a, expected_b);
    HELIX_TRACE_FPRINTF("  Expected cluster IDs from gate (token %lld): %d %d\n",
            (long long) focus, expected_a, expected_b);
    LLAMA_LOG("  ids at focus token %lld: %d %d\n", (long long) focus, ids_a, ids_b);
    HELIX_TRACE_FPRINTF("  ids at focus token %lld: %d %d\n", (long long) focus, ids_a, ids_b);
}

static void helix_trace_dump_down_exps_raw_weights(const ggml_tensor * down_exps, int il) {
    if (down_exps == nullptr || down_exps->type != GGML_TYPE_Q8_0) {
        LLAMA_LOG("[HELIX TRACE] Layer %d Sidecar Raw Weights: skipped (type=%d)\n", il, down_exps ? (int) down_exps->type : -1);
        return;
    }

    helix_blk_q8_0 block{};
    ggml_backend_tensor_get(down_exps, &block, 0, sizeof(block));

    const float scale = ggml_fp16_to_fp32(block.d);
    LLAMA_LOG("[HELIX TRACE] Layer %d Sidecar Raw Weights:\n", il);
    LLAMA_LOG("  Block 0 scale (d): %f\n", scale);
    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d Sidecar Raw Weights:\n", il);
    HELIX_TRACE_FPRINTF("  Block 0 scale (d): %f\n", scale);
    for (int i = 0; i < 4; ++i) {
        const float val = (float) block.qs[i] * scale;
        LLAMA_LOG("  weight[%d] = %f (raw q8: %d)\n", i, val, (int) block.qs[i]);
        HELIX_TRACE_FPRINTF("  weight[%d] = %f (raw q8: %d)\n", i, val, (int) block.qs[i]);
    }
    LLAMA_LOG("  down_exps ne=[%lld,%lld,%lld]\n",
            (long long) down_exps->ne[0], (long long) down_exps->ne[1], (long long) down_exps->ne[2]);
    HELIX_TRACE_FPRINTF("  down_exps ne=[%lld,%lld,%lld]\n",
            (long long) down_exps->ne[0], (long long) down_exps->ne[1], (long long) down_exps->ne[2]);
}

static void helix_trace_dump_top_clusters(ggml_tensor * t, int il) {
    if (t == nullptr || t->type != GGML_TYPE_I32) {
        return;
    }

    const int64_t top_k      = t->ne[0];
    const int64_t n_tokens   = t->ne[1];
    const ggml_tensor * gate_scores = helix_trace_gate_scores_from_node(t);
    const int64_t num_clusters = gate_scores ? gate_scores->ne[0] : 0;
    const int64_t cluster_width = (num_clusters > 0) ? (6144 / num_clusters) : 0;

    std::vector<int32_t> host(ggml_nbytes(t) / sizeof(int32_t));
    ggml_backend_tensor_get(t, host.data(), 0, ggml_nbytes(t));

    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | top_k=%lld | n_tokens=%lld | clusters=%lld | cluster_width=%lld\n",
            il, (long long) top_k, (long long) n_tokens, (long long) num_clusters, (long long) cluster_width);

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        if (il == 19 && helix_trace_enabled() && helix_trace_want_raw_logits(il, tok, n_tokens)) {
            int32_t cid_a = (top_k > 0) ? host[0 + tok * top_k] : -1;
            int32_t cid_b = (top_k > 1) ? host[1 + tok * top_k] : -1;
            LLAMA_LOG("selected cluster IDs: %d %d\n", cid_a, cid_b);
            HELIX_TRACE_FPRINTF("selected cluster IDs: %d %d\n", cid_a, cid_b);
        }
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | Selected Cluster IDs:", il, (long long) tok);
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t cid = host[k + tok * top_k];
            const int64_t weight_offset = (int64_t) cid * cluster_width;
            HELIX_TRACE_FPRINTF(" %d (offset %lld)", cid, (long long) weight_offset);
        }
        HELIX_TRACE_FPRINTF("\n");
    }
}

static void helix_trace_dump_raw_logits_prefix(ggml_tensor * t, int il, int64_t tok, int count) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || count <= 0) {
        return;
    }

    const int64_t num_clusters = t->ne[0];
    const int64_t n_tokens     = t->ne[1];
    if (tok < 0 || tok >= n_tokens) {
        return;
    }

    const int n_print = (int) std::min<int64_t>(count, num_clusters);
    std::vector<float> row(num_clusters);
    ggml_backend_tensor_get(t, row.data(), tok * num_clusters * sizeof(float), num_clusters * sizeof(float));

    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | raw logits (pre-softmax) first %d:",
            il, (long long) tok, n_print);
    for (int i = 0; i < n_print; ++i) {
        HELIX_TRACE_FPRINTF(" [%d]=%.6f", i, row[i]);
    }
    HELIX_TRACE_FPRINTF("\n");
}

static void helix_trace_dump_hidden_prefix(ggml_tensor * gate_scores, int il, int64_t tok, int count) {
    if (gate_scores == nullptr || gate_scores->src[1] == nullptr) {
        return;
    }
    ggml_tensor * hidden = gate_scores->src[1];
    if (hidden->type != GGML_TYPE_F32) {
        return;
    }
    const int64_t n_embd = hidden->ne[0];
    const int64_t n_tokens = hidden->ne[1];
    if (tok < 0 || tok >= n_tokens) {
        return;
    }
    const int n_print = (int) std::min<int64_t>(count, n_embd);
    std::vector<float> row(n_embd);
    ggml_backend_tensor_get(hidden, row.data(), tok * n_embd * sizeof(float), n_embd * sizeof(float));
    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | hidden (pre-gate) first %d:",
            il, (long long) tok, n_print);
    for (int i = 0; i < n_print; ++i) {
        HELIX_TRACE_FPRINTF(" [%d]=%.6f", i, row[i]);
    }
    HELIX_TRACE_FPRINTF("\n");
}

static void helix_trace_dump_gate_scores(ggml_tensor * t, int il, int top_k) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || t->src[0] == nullptr) {
        return;
    }

    const int64_t num_clusters = t->ne[0];
    const int64_t n_tokens     = t->ne[1];
    const int64_t cluster_width = (num_clusters > 0) ? (6144 / num_clusters) : 0;

    std::vector<float> scores(num_clusters * n_tokens);
    ggml_backend_tensor_get(t, scores.data(), 0, scores.size() * sizeof(float));

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        const float * row = scores.data() + tok * num_clusters;

        if (il == 19 && helix_trace_enabled() && helix_trace_want_raw_logits(il, tok, n_tokens)) {
            LLAMA_LOG("=== LAYER 19 GEOMETRY TRACE (token %lld) ===\n", (long long) tok);
            LLAMA_LOG("layer_clusters: %lld\n", (long long) num_clusters);
            LLAMA_LOG("cluster_width: %lld\n", (long long) cluster_width);
            HELIX_TRACE_FPRINTF("=== LAYER 19 GEOMETRY TRACE (token %lld) ===\n", (long long) tok);
            HELIX_TRACE_FPRINTF("layer_clusters: %lld\n", (long long) num_clusters);
            HELIX_TRACE_FPRINTF("cluster_width: %lld\n", (long long) cluster_width);
            LLAMA_LOG("raw gate logits first 8:");
            HELIX_TRACE_FPRINTF("raw gate logits first 8:");
            const int n_print = (int) std::min<int64_t>(8, num_clusters);
            for (int i = 0; i < n_print; ++i) {
                LLAMA_LOG(" %.6f", row[i]);
                HELIX_TRACE_FPRINTF(" %.6f", row[i]);
            }
            LLAMA_LOG("\n");
            HELIX_TRACE_FPRINTF("\n");
        }

        if (helix_trace_want_raw_logits(il, tok, n_tokens)) {
            if (tok < (int64_t) helix_trace_token_ids.size()) {
                HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | token_id=%d\n",
                        il, (long long) tok, helix_trace_token_ids[tok]);
            }
            helix_trace_dump_hidden_prefix(t, il, tok, 8);
            helix_trace_dump_raw_logits_prefix(t, il, tok, 8);
        }

        // softmax for trace probabilities
        float maxv = row[0];
        for (int64_t c = 1; c < num_clusters; ++c) {
            maxv = std::max(maxv, row[c]);
        }
        double sum = 0.0;
        std::vector<double> probs(num_clusters);
        for (int64_t c = 0; c < num_clusters; ++c) {
            probs[c] = exp((double) row[c] - (double) maxv);
            sum += probs[c];
        }
        for (int64_t c = 0; c < num_clusters; ++c) {
            probs[c] /= sum;
        }

        std::vector<int64_t> order(num_clusters);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
            [&](int64_t a, int64_t b) { return probs[a] > probs[b]; });

        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | gate softmax top-%d:", il, (long long) tok, top_k);
        for (int k = 0; k < top_k; ++k) {
            const int64_t cid = order[k];
            const int64_t weight_offset = cid * cluster_width;
            HELIX_TRACE_FPRINTF(" %lld(prob=%.6f,logit=%.6f,offset=%lld)",
                    (long long) cid, probs[cid], row[cid], (long long) weight_offset);
        }
        HELIX_TRACE_FPRINTF("\n");
    }
}

static void helix_trace_dump_row_prefix(
        ggml_tensor * t,
        int il,
        int64_t tok,
        int count,
        const char * label) {
    if (t == nullptr || count <= 0 || label == nullptr) {
        return;
    }

    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (tok < 0 || tok >= n_tokens) {
        return;
    }

    const int n_print = (int) std::min<int64_t>(count, n_embd);
    std::vector<float> row(n_embd);

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, row.data(), tok * n_embd * sizeof(float), n_embd * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> row_f16(n_embd);
        ggml_backend_tensor_get(t, row_f16.data(), tok * n_embd * sizeof(ggml_fp16_t), n_embd * sizeof(ggml_fp16_t));
        for (int i = 0; i < n_print; ++i) {
            row[i] = ggml_fp16_to_fp32(row_f16[i]);
        }
    } else {
        return;
    }

    if (tok < (int64_t) helix_trace_token_ids.size()) {
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | token_id=%d | %s first %d:",
                il, (long long) tok, helix_trace_token_ids[tok], label, n_print);
    } else {
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d | token %lld | %s first %d:",
                il, (long long) tok, label, n_print);
    }
    for (int i = 0; i < n_print; ++i) {
        HELIX_TRACE_FPRINTF(" [%d]=%.6f", i, row[i]);
    }
    HELIX_TRACE_FPRINTF("\n");
}

static void helix_trace_dump_swiglu_prefix(
        ggml_tensor * t,
        int il,
        int64_t tok,
        int count,
        const char * label) {
    if (t == nullptr || count <= 0 || label == nullptr) {
        return;
    }

    const int64_t width    = t->ne[0];
    const int64_t top_k    = t->ne[1];
    const int64_t n_tokens = t->ne[2];
    if (tok < 0 || tok >= n_tokens || width <= 0 || top_k <= 0) {
        return;
    }

    const int n_print = (int) std::min<int64_t>(count, width);
    std::vector<float> row((size_t) width);

    for (int64_t expert = 0; expert < top_k; ++expert) {
        const size_t byte_offset = (size_t) tok * t->nb[2] + (size_t) expert * t->nb[1];

        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, row.data(), byte_offset, (size_t) width * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> row_f16((size_t) width);
            ggml_backend_tensor_get(t, row_f16.data(), byte_offset, (size_t) width * sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < width; ++i) {
                row[(size_t) i] = ggml_fp16_to_fp32(row_f16[(size_t) i]);
            }
        } else {
            return;
        }

        if (tok < (int64_t) helix_trace_token_ids.size()) {
            HELIX_TRACE_FPRINTF(
                "[HELIX TRACE] Layer %d | token %lld | token_id=%d | %s expert_slot=%lld first %d:",
                il, (long long) tok, helix_trace_token_ids[tok], label, (long long) expert, n_print);
        } else {
            HELIX_TRACE_FPRINTF(
                "[HELIX TRACE] Layer %d | token %lld | %s expert_slot=%lld first %d:",
                il, (long long) tok, label, (long long) expert, n_print);
        }
        for (int i = 0; i < n_print; ++i) {
            HELIX_TRACE_FPRINTF(" [%d]=%.6f", i, row[(size_t) i]);
        }
        HELIX_TRACE_FPRINTF("\n");
    }
}

static void helix_trace_dump_experts_down_out(
        ggml_tensor * t,
        int il,
        int64_t tok,
        int count) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || count <= 0) {
        return;
    }

    const int64_t n_embd   = t->ne[0];
    const int64_t top_k    = t->ne[1];
    const int64_t n_tokens = t->ne[2];
    if (tok < 0 || tok >= n_tokens || top_k <= 0) {
        return;
    }

    const int n_print = (int) std::min<int64_t>(count, n_embd);
    std::vector<float> row((size_t) n_embd);
    const size_t byte_offset = (size_t) tok * t->nb[2];
    ggml_backend_tensor_get(t, row.data(), byte_offset, (size_t) n_embd * sizeof(float));

    LLAMA_LOG("[HELIX TRACE] Layer %d post-down-proj (mul_mat_id output) token %lld first %d:",
            il, (long long) tok, n_print);
    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d post-down-proj (mul_mat_id output) token %lld first %d:",
            il, (long long) tok, n_print);
    for (int i = 0; i < n_print; ++i) {
        LLAMA_LOG(" [%d]=%.6f", i, row[(size_t) i]);
        HELIX_TRACE_FPRINTF(" [%d]=%.6f", i, row[(size_t) i]);
    }
    LLAMA_LOG("\n");
    HELIX_TRACE_FPRINTF("\n");
    LLAMA_LOG("[HELIX TRACE] Layer %d post-down-proj tensor shape: [%lld, %lld, %lld]\n",
            il, (long long) n_embd, (long long) top_k, (long long) n_tokens);
    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d post-down-proj tensor shape: [%lld, %lld, %lld]\n",
            il, (long long) n_embd, (long long) top_k, (long long) n_tokens);
}

static bool helix_trace_is_l19_swiglu_tensor(const char * name) {
    return name != nullptr && strcmp(name, "helix_swiglu_l19_tok3") == 0;
}

static bool helix_trace_is_l19_gate_raw_tensor(const char * name) {
    return name != nullptr && strcmp(name, "helix_gate_raw_l19") == 0;
}

static bool helix_trace_is_l19_up_raw_tensor(const char * name) {
    return name != nullptr && strcmp(name, "helix_up_raw_l19") == 0;
}

static void helix_trace_dump_l19_raw_linear(
        ggml_tensor * t,
        const char * label,
        int64_t tok,
        int count) {
    if (t == nullptr || label == nullptr || count <= 0) {
        return;
    }

    const int64_t width    = t->ne[0];
    const int64_t top_k    = t->ne[1];
    const int64_t n_tokens = t->ne[2];
    if (tok < 0 || tok >= n_tokens || width <= 0 || top_k <= 0) {
        return;
    }

    const int n_print = (int) std::min<int64_t>(count, width);
    std::vector<float> row((size_t) width);
    const size_t byte_offset = (size_t) tok * t->nb[2];

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, row.data(), byte_offset, (size_t) width * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> row_f16((size_t) width);
        ggml_backend_tensor_get(t, row_f16.data(), byte_offset, (size_t) width * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < width; ++i) {
            row[(size_t) i] = ggml_fp16_to_fp32(row_f16[(size_t) i]);
        }
    } else {
        return;
    }

    LLAMA_LOG("[HELIX TRACE] Layer 19 %s:", label);
    HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 %s:", label);
    for (int i = 0; i < n_print; ++i) {
        LLAMA_LOG(" val[%d]=%.6f", i, row[(size_t) i]);
        HELIX_TRACE_FPRINTF(" val[%d]=%.6f", i, row[(size_t) i]);
    }
    LLAMA_LOG("\n");
    HELIX_TRACE_FPRINTF("\n");
}

static bool helix_trace_is_ffn_out_tensor(const char * name) {
    return helix_trace_name_matches(name, "helix_ffn_out") ||
           helix_trace_name_matches(name, "ffn_out");
}

bool helix_runtime_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    if (helix_doppelganger_enabled() && t != nullptr && t->name[0] != '\0') {
        const bool is_magnet_mask = helix_trace_name_matches(t->name, "magnet_mask");
        const bool is_magnet_topk = helix_trace_name_matches(t->name, "magnet_topk");
        const bool is_dg_mask     = helix_trace_name_matches(t->name, "dg_mask");
        if (is_magnet_mask || is_magnet_topk || is_dg_mask) {
            if (ask) {
                return true;
            }
            const int il = helix_trace_layer_from_name(t->name);
            if (is_magnet_topk) {
                const int64_t n_ff = (il >= 0 && il < HELIX_SPARSITY_N_LAYERS) ? g_helix_layer_n_ff[il] : 0;
                const int64_t n_tokens = t->ne[1] > 0 ? t->ne[1] : 1;
                const int64_t k_sel    = t->ne[0];
                if (n_ff > 0) {
                    g_helix_sparsity.active_neurons += (uint64_t) k_sel * (uint64_t) n_tokens;
                    g_helix_sparsity.total_neurons  += (uint64_t) n_ff  * (uint64_t) n_tokens;
                    ++g_helix_sparsity.mask_samples;
                    if (n_tokens == 1) {
                        g_helix_sparsity.decode_active += (uint64_t) k_sel;
                        g_helix_sparsity.decode_total  += (uint64_t) n_ff;
                        ++g_helix_sparsity.decode_mask_samples;
                    }
                    g_helix_sparsity.layer_magnet[il] = true;
                }
            } else {
                helix_sparsity_accumulate_mask(t, il, is_magnet_mask, false);
            }
            return true;
        }
    }

    if (helix_trace_enabled() || helix_act_cache_enabled()) {
        return helix_trace_cb_eval(t, ask, user_data);
    }

    return false;
}

bool helix_trace_cb_eval(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!helix_trace_enabled() && !helix_act_cache_enabled()) {
        return false;
    }

    if (t == nullptr || t->name[0] == '\0') {
        return false;
    }

    const int il = helix_trace_layer_from_name(t->name);
    const bool is_top      = helix_trace_name_matches(t->name, "helix_top_clusters");
    const bool is_gate     = helix_trace_name_matches(t->name, "helix_gate_scores");
    const bool is_act_gate = il == 0 && helix_trace_enabled() && helix_trace_name_matches(t->name, "helix_act_gate");
    const bool is_act_up   = il == 0 && helix_trace_enabled() && helix_trace_name_matches(t->name, "helix_act_up");
    const bool is_l19_swiglu = helix_trace_enabled() && helix_trace_is_l19_swiglu_tensor(t->name);
    const bool is_l19_gate_raw = helix_trace_enabled() && helix_trace_is_l19_gate_raw_tensor(t->name);
    const bool is_l19_up_raw   = helix_trace_enabled() && helix_trace_is_l19_up_raw_tensor(t->name);
    const bool is_swiglu   = il == 0 && helix_trace_enabled() && helix_trace_name_matches(t->name, "helix_swiglu");
    const bool is_down     = il == 19 && helix_trace_enabled() &&
        (helix_trace_name_matches(t->name, "helix_down") ||
         helix_trace_name_matches(t->name, "helix_debug_down_out"));
    const bool is_ffn      = il == 0 && helix_trace_enabled() && helix_trace_is_ffn_out_tensor(t->name);
    if (!is_top && !is_gate && !is_ffn && !is_swiglu && !is_l19_swiglu && !is_l19_gate_raw && !is_l19_up_raw &&
            !is_act_gate && !is_act_up && !is_down) {
        return false;
    }

    // Never attach trace eval hooks during single-token decode (generation loop).
    if (helix_trace_enabled() && helix_trace_batch_n_tokens <= 1 &&
        (is_swiglu || is_l19_swiglu || is_l19_gate_raw || is_l19_up_raw || is_ffn || is_top ||
         is_act_gate || is_act_up || is_down)) {
        return false;
    }

    const bool act_layer = is_gate && helix_act_cache_enabled() &&
        il >= 0 && il < (int) HELIX_ACT_N_LAYERS;
    const bool trace_layer = helix_trace_enabled() &&
        (helix_trace_layer_wanted(il) || is_ffn || is_swiglu || is_l19_swiglu || is_l19_gate_raw ||
         is_l19_up_raw || is_act_gate || is_act_up || is_down);

    if (ask) {
        return act_layer || trace_layer;
    }

    if (act_layer) {
        helix_act_cache_write_gate_hidden(t, il);
    }

    if (is_ffn && helix_trace_file() != nullptr) {
        const int64_t n_tokens = t->ne[1];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(0, focus, n_tokens)) {
            helix_trace_dump_row_prefix(t, il, focus, 8, "post-ffn (sparse mlp output)");
        }
    }

    if (is_act_gate && helix_trace_file() != nullptr) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(0, focus, n_tokens)) {
            helix_trace_dump_swiglu_prefix(t, il, focus, 8, "gate_exps linear (pre-SiLU)");
        }
    }

    if (is_act_up && helix_trace_file() != nullptr) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(0, focus, n_tokens)) {
            helix_trace_dump_swiglu_prefix(t, il, focus, 8, "up_exps linear (pre-SiLU)");
        }
    }

    if (is_swiglu && helix_trace_file() != nullptr) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(0, focus, n_tokens)) {
            helix_trace_dump_swiglu_prefix(t, il, focus, 8, "SwiGLU activation intermediate");
        }
    }

    if (is_l19_swiglu) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(19, focus, n_tokens)) {
            helix_trace_dump_swiglu_prefix(t, 19, focus, 8, "Layer 19 pre-down SwiGLU");
        }
    }

    if (is_l19_gate_raw) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(19, focus, n_tokens)) {
            helix_trace_dump_l19_raw_linear(t, "Raw Gate Output", focus, 8);
        }
    }

    if (is_l19_up_raw) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(19, focus, n_tokens)) {
            helix_trace_dump_l19_raw_linear(t, "Raw Up Output", focus, 8);
        }
    }

    if (is_down) {
        const int64_t n_tokens = t->ne[2];
        const int64_t focus = helix_trace_focus_token_index(n_tokens);
        if (helix_trace_want_raw_logits(il, focus, n_tokens)) {
            helix_trace_dump_experts_down_out(t, il, focus, 8);
        }
    }

    if (!trace_layer || helix_trace_file() == nullptr) {
        return true;
    }

    if (!is_ffn && !is_swiglu && !is_l19_swiglu && !is_l19_gate_raw && !is_l19_up_raw && !is_act_gate && !is_act_up && !is_down) {
        if (is_top) {
            if (il == 19 && helix_trace_enabled()) {
                const int top_k = (int) t->ne[0];
                helix_trace_dump_mul_mat_id_ids(t, il, top_k, true);
            }
            helix_trace_dump_top_clusters(t, il);
        } else if (is_gate) {
            const int64_t num_clusters = t->ne[0];
            const int top_k = (il >= 19) ? 2 : 8;
            helix_trace_dump_gate_scores(t, il, top_k);
        }
    }

    return true;
}


ggml_tensor * llm_graph_context::build_helix_dnpa_sparse_ffn(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * ffn_up,
        ggml_tensor * ffn_gate,
        ggml_tensor * ffn_down,
        ggml_tensor * helix_ffn_gate_exps,
        ggml_tensor * helix_ffn_up_exps,
        ggml_tensor * helix_ffn_down_exps,
        ggml_tensor * helix_router_gate,
        ggml_tensor * helix_cluster_map,
        int           il,
        ggml_tensor * helix_shared_core_gate,
        ggml_tensor * helix_shared_core_up,
        ggml_tensor * helix_shared_core_down,
        ggml_tensor * helix_magnet_a,
        ggml_tensor * helix_magnet_b,
        ggml_tensor * helix_ffn_gate_live,
        ggml_tensor * helix_ffn_up_live,
        ggml_tensor * helix_ffn_down_live) const {
    // Force dense path for early layers — sparse sidecars not reliable for L0-2.
    if (il < 3) {
        if (!g_helix_stats.printed) g_helix_stats.layers_dense++;
        ggml_tensor * act_up   = ggml_mul_mat(ctx0, ffn_up,   cur);
        ggml_tensor * act_gate = ggml_mul_mat(ctx0, ffn_gate, cur);
        ggml_tensor * swiglu   = ggml_mul(ctx0, ggml_silu(ctx0, act_gate), act_up);
        return ggml_mul_mat(ctx0, ffn_down, swiglu);
    }

    // =========================================================================
    // MAGNET SPARSE MODE: True hardware-accelerated sparse FFN.
    // Uses the trained look-ahead predictor (scores = X @ A @ B) to identify
    // active neurons WITHOUT computing the full gate. Then gathers only the
    // active rows from gate/up/down for physically smaller matmuls.
    // When magnet tensors are absent, falls back to analytical gate masking.
    // =========================================================================
    if (helix_doppelganger_enabled() && ffn_gate != nullptr && ffn_up != nullptr && ffn_down != nullptr) {
        // Magnet path: predict active neurons via low-rank projection (6x cheaper than gate)
        if (helix_magnet_a != nullptr && helix_magnet_b != nullptr) {
            g_helix_sparsity.layer_magnet[il] = true;
            const ggml_type cur_type_mg = cur->type;
            ggml_tensor * cur_f32_mg = (cur->type == GGML_TYPE_F32) ? cur : ggml_cast(ctx0, cur, GGML_TYPE_F32);

            // Low-rank predictor: scores = X @ A @ B  -> [n_ff, n_tokens]
            const int64_t mag_rank = helix_magnet_a->ne[0] == cur_f32_mg->ne[0]
                ? helix_magnet_a->ne[1] : helix_magnet_a->ne[0];
            ggml_tensor * magnet_a_f32 = helix_magnet_view_for_mul_mat(
                ctx0, helix_magnet_a, cur_f32_mg->ne[0], mag_rank);
            ggml_tensor * hidden_proj = ggml_mul_mat(ctx0, magnet_a_f32, cur_f32_mg);
            ggml_tensor * magnet_b_f32 = helix_magnet_view_for_mul_mat(
                ctx0, helix_magnet_b, hidden_proj->ne[0], ffn_gate->ne[1]);
            cb(hidden_proj, "magnet_proj_a", il);
            ggml_tensor * magnet_scores = ggml_mul_mat(ctx0, magnet_b_f32, hidden_proj);
            cb(magnet_scores, "magnet_scores", il);

            const int64_t n_ff = ffn_gate->ne[1];
            const int64_t active_k = helix_magnet_active_k(il, n_ff);
            if (il >= 0 && il < HELIX_SPARSITY_N_LAYERS) {
                g_helix_layer_n_ff[il] = n_ff;
            }

            const bool sparse_mode   = helix_magnet_sparse_mode_enabled();
            const bool sparse_gather = helix_magnet_row_gather_enabled();
            const bool use_dynamic   = helix_magnet_dynamic_enabled();
            ggml_tensor * mg_out = nullptr;

            if (sparse_gather) {
                // Magnet top-k indices; matmul only on k rows (weights stay mmap'd — not all rows computed).
                const int64_t gather_k = helix_magnet_active_k(il, n_ff);
                const int64_t k_max    = helix_magnet_k_max(n_ff);
                // Dynamic + staged: rank by |score| (same threshold family as masked sparse).
                ggml_tensor * scores_for_pick = use_dynamic
                    ? ggml_abs(ctx0, magnet_scores)
                    : magnet_scores;
                ggml_tensor * selected = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, scores_for_pick, (int) gather_k));
                cb(selected, "magnet_topk", il);

                const int64_t n_embd_cur = cur_f32_mg->ne[0];
                const int64_t n_ffn_tokens = cur_f32_mg->ne[1];
                ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur_f32_mg, n_embd_cur, 1, n_ffn_tokens);

                const bool use_paged = helix_magnet_paged_enabled() &&
                    helix_ffn_gate_live != nullptr && helix_ffn_up_live != nullptr &&
                    helix_ffn_down_live != nullptr;

                ggml_tensor * gate_f32 = ggml_cont(ctx0, ggml_cast(ctx0, ffn_gate, GGML_TYPE_F32));
                ggml_tensor * up_f32   = ggml_cont(ctx0, ggml_cast(ctx0, ffn_up,   GGML_TYPE_F32));
                ggml_tensor * down_f32 = ggml_cont(ctx0, ggml_cast(ctx0, ffn_down, GGML_TYPE_F32));
                ggml_tensor * gate_as = ggml_reshape_3d(ctx0, gate_f32, n_embd_cur, 1, n_ff);
                ggml_tensor * up_as   = ggml_reshape_3d(ctx0, up_f32,   n_embd_cur, 1, n_ff);
                ggml_tensor * down_as = ggml_reshape_3d(ctx0, down_f32, 1, n_embd_cur, n_ff);

                ggml_tensor * gate_w = gate_as;
                ggml_tensor * up_w   = up_as;
                ggml_tensor * down_w = down_as;
                ggml_tensor * mm_ids = selected;

                if (use_paged) {
                    ggml_tensor * gate_live_3d = ggml_reshape_3d(ctx0, helix_ffn_gate_live, n_embd_cur, 1, k_max);
                    ggml_tensor * up_live_3d   = ggml_reshape_3d(ctx0, helix_ffn_up_live,   n_embd_cur, 1, k_max);
                    ggml_tensor * down_live_3d = ggml_reshape_3d(ctx0, helix_ffn_down_live, 1, n_embd_cur, k_max);

                    ggml_tensor * gate_packed = helix_graph_paged_pack_rows(ctx0, gate_as, selected, gate_live_3d, gather_k);
                    ggml_tensor * up_packed   = helix_graph_paged_pack_rows(ctx0, up_as,   selected, up_live_3d,   gather_k);
                    ggml_tensor * down_packed = helix_graph_paged_pack_rows(ctx0, down_as, selected, down_live_3d, gather_k);
                    cb(gate_packed, "magnet_paged_gate", il);
                    cb(up_packed, "magnet_paged_up", il);
                    cb(down_packed, "magnet_paged_down", il);

                    gate_w  = gate_live_3d;
                    up_w    = up_live_3d;
                    down_w  = down_live_3d;
                    mm_ids  = helix_graph_seq_indices(ctx0, gather_k, n_ffn_tokens);
                    cb(mm_ids, "magnet_paged_ids", il);
                }

                ggml_tensor * act_gate = build_lora_mm_id(gate_w, cur_3d, mm_ids);
                ggml_tensor * act_up   = build_lora_mm_id(up_w, cur_3d, mm_ids);
                cb(act_gate, "magnet_gate_sparse", il);
                cb(act_up, "magnet_up_sparse", il);

                ggml_tensor * gate_act_f32 = ggml_cont(ctx0, ggml_cast(ctx0, act_gate, GGML_TYPE_F32));
                ggml_tensor * up_act_f32   = ggml_cont(ctx0, ggml_cast(ctx0, act_up,   GGML_TYPE_F32));
                ggml_tensor * swiglu   = ggml_swiglu_split(ctx0, gate_act_f32, up_act_f32);
                cb(swiglu, "magnet_swiglu_sparse", il);

                ggml_tensor * down_out = build_lora_mm_id(down_w, swiglu, mm_ids);
                down_out = ggml_cont(ctx0, ggml_cast(ctx0, down_out, GGML_TYPE_F32));
                cb(down_out, "magnet_down_sparse", il);

                mg_out = helix_sum_dim1_rows(ctx0, down_out, n_embd_cur, n_ffn_tokens);
                cb(mg_out, "magnet_ffn_out", il);
            } else if (sparse_mode) {
                ggml_tensor * selected = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, magnet_scores, (int) active_k));
                cb(selected, "magnet_topk", il);

                ggml_tensor * mg_mask = use_dynamic
                    ? helix_magnet_build_dynamic_mask(ctx0, magnet_scores, il)
                    : helix_magnet_build_topk_mask(ctx0, magnet_scores, (int) active_k);
                cb(mg_mask, "magnet_mask", il);

                ggml_tensor * act_gate_mg = ggml_mul_mat(ctx0, ffn_gate, cur_f32_mg);
                ggml_tensor * act_up_mg   = ggml_mul_mat(ctx0, ffn_up, cur_f32_mg);
                ggml_tensor * z_full_mg   = ggml_mul(ctx0, ggml_silu(ctx0, act_gate_mg), act_up_mg);
                ggml_tensor * z_sparse_mg = ggml_mul(ctx0, z_full_mg, mg_mask);
                cb(z_sparse_mg, "magnet_swiglu_masked", il);
                mg_out = ggml_mul_mat(ctx0, ffn_down, z_sparse_mg);
                cb(mg_out, "magnet_ffn_out", il);
            } else {
                // Dense fallback (HELIX_MAGNET_DENSE=1): full SwiGLU; threshold mask for stats only.
                const float target_frac = (n_ff > 0) ? (float) active_k / (float) n_ff : 0.20f;
                const float scale = 0.68f * target_frac;

                ggml_tensor * scores_abs = ggml_abs(ctx0, magnet_scores);
                ggml_tensor * scores_mean = ggml_mean(ctx0, scores_abs);
                ggml_tensor * mg_threshold = ggml_scale(ctx0, ggml_repeat(ctx0, scores_mean, scores_abs), scale);
                ggml_tensor * mg_above = ggml_sub(ctx0, scores_abs, mg_threshold);
                ggml_tensor * mg_mask = ggml_step(ctx0, mg_above);
                cb(mg_mask, "magnet_mask", il);

                static const bool apply_magnet_mask = helix_env_flag_active("HELIX_MAGNET_APPLY_MASK");
                ggml_tensor * act_gate_mg = ggml_mul_mat(ctx0, ffn_gate, cur_f32_mg);
                ggml_tensor * gate_silu_mg = ggml_silu(ctx0, act_gate_mg);
                ggml_tensor * act_up_mg   = ggml_mul_mat(ctx0, ffn_up, cur_f32_mg);
                ggml_tensor * z_sparse_mg = nullptr;
                if (apply_magnet_mask) {
                    ggml_tensor * gate_masked_mg = ggml_mul(ctx0, gate_silu_mg, mg_mask);
                    cb(gate_masked_mg, "magnet_gate_masked", il);
                    z_sparse_mg = ggml_mul(ctx0, gate_masked_mg, act_up_mg);
                } else {
                    z_sparse_mg = ggml_mul(ctx0, gate_silu_mg, act_up_mg);
                    z_sparse_mg = helix_sparsity_anchor_tensor(ctx0, z_sparse_mg, magnet_scores);
                    z_sparse_mg = helix_sparsity_anchor_tensor(ctx0, z_sparse_mg, mg_mask);
                }
                mg_out = ggml_mul_mat(ctx0, ffn_down, z_sparse_mg);
                cb(mg_out, "magnet_ffn_out", il);
            }

            if (mg_out->type != cur_type_mg) {
                mg_out = ggml_cast(ctx0, mg_out, cur_type_mg);
            }

            if (il == 27 && !g_helix_stats.printed) {
                g_helix_stats.layers_dense  = 3;
                g_helix_stats.layers_magnet = 25;
                g_helix_stats.layers_sparse = 0;
                g_helix_stats.printed = true;
                for (int l = 3; l < HELIX_SPARSITY_N_LAYERS; ++l) {
                    g_helix_sparsity.layer_magnet[l] = true;
                }
            }

            {
                static bool dg_banner_shown = false;
                if (il == 27 && !dg_banner_shown) {
                    dg_banner_shown = true;
                    const float wf = helix_magnet_env_width_frac();
                    fprintf(stderr,
                        "\n[HELIX MAGNET] Engine active: 25 magnet layers | 3 dense | %s | "
                        "width dial: %s | per-token activation %% after each reply\n\n",
                        helix_magnet_paged_enabled() ? "paged gather (GPU scratchpad + H2D)"
                        : (helix_magnet_row_gather_enabled()
                            ? (use_dynamic ? "row gather + dynamic k" : "row gather (mul_mat_id)")
                            : (helix_magnet_sparse_mode_enabled() ? "top-k masked dense, gate-guided width"
                                                                  : "dense FFN (HELIX_MAGNET_DENSE=1)")),
                        wf > 0.0f ? "HELIX_MAGNET_WIDTH_FRAC env"
                                    : "default L3-12=25%% L13-27=20%%");
                }
            }

            if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
                HELIX_TRACE_FPRINTF("[HELIX MAGNET] Layer %d: rank=%lld active_k=%lld sparse=%d\n",
                        il, (long long) helix_magnet_a->ne[1], (long long) active_k,
                        (int) helix_magnet_row_gather_enabled());
            }
            return mg_out;
        }

        // Analytical gate-masking fallback (no magnet weights available)
        const ggml_type cur_type_dg = cur->type;
        ggml_tensor * cur_f32_dg = (cur->type == GGML_TYPE_F32) ? cur : ggml_cast(ctx0, cur, GGML_TYPE_F32);

        // Phase A: Full gate matmul — computes exact activation magnitudes
        ggml_tensor * act_gate_full = ggml_mul_mat(ctx0, ffn_gate, cur_f32_dg);
        cb(act_gate_full, "dg_gate_full", il);
        ggml_tensor * gate_silu = ggml_silu(ctx0, act_gate_full);
        cb(gate_silu, "dg_gate_silu", il);

        // Phase B: Top-P energy masking via adaptive threshold.
        // Threshold = mean(|gate|) * scale. Empirically scale=2.0 yields ~30-35% active
        // width for SwiGLU activation distributions, matching our verified Top-35% sweet spot.
        ggml_tensor * gate_abs = ggml_abs(ctx0, gate_silu);
        cb(gate_abs, "dg_gate_abs", il);

        ggml_tensor * gate_mean = ggml_mean(ctx0, gate_abs);
        cb(gate_mean, "dg_gate_mean", il);
        ggml_tensor * threshold = ggml_scale(ctx0, ggml_repeat(ctx0, gate_mean, gate_abs), 2.0f);
        cb(threshold, "dg_threshold", il);

        // Mask: step(|gate| - threshold) = 1 where neuron is active, 0 otherwise
        ggml_tensor * above_thresh = ggml_sub(ctx0, gate_abs, threshold);
        ggml_tensor * mask = ggml_step(ctx0, above_thresh);
        cb(mask, "dg_mask", il);
        g_helix_sparsity.layer_gate[il] = true;

        // Phase C: Apply mask to gate — zeroes inactive neurons
        ggml_tensor * gate_masked = ggml_mul(ctx0, gate_silu, mask);
        cb(gate_masked, "dg_gate_masked", il);

        // Phase D: Full up matmul (active channels will be kept, inactive zeroed by gate)
        ggml_tensor * act_up_full = ggml_mul_mat(ctx0, ffn_up, cur_f32_dg);
        cb(act_up_full, "dg_up_full", il);

        // SwiGLU element-wise: masked_gate * up (inactive channels → 0)
        ggml_tensor * z_sparse = ggml_mul(ctx0, gate_masked, act_up_full);
        cb(z_sparse, "dg_z_sparse", il);

        // Phase E: Down projection on sparse intermediate (zeros don't contribute)
        ggml_tensor * dg_out = ggml_mul_mat(ctx0, ffn_down, z_sparse);
        cb(dg_out, "dg_ffn_out", il);

        if (dg_out->type != cur_type_dg) {
            dg_out = ggml_cast(ctx0, dg_out, cur_type_dg);
        }

            if (il == 27 && !g_helix_stats.printed) {
            g_helix_stats.layers_dense = 3;
            g_helix_stats.layers_sparse = 25;
            g_helix_stats.printed = true;
            LLAMA_LOG("\n[HELIX DOPPELGANGER] Engine active: 25 gate-masked layers | 3 dense | "
                     "~35%% active width | ~57%% FFN FLOP reduction\n");
        }

        if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
            HELIX_TRACE_FPRINTF("[HELIX DOPPELGANGER] Layer %d: threshold=mean*2.0 gate_ne=[%lld,%lld]\n",
                    il,
                    (long long) ffn_gate->ne[0], (long long) ffn_gate->ne[1]);
        }

        return dg_out;
    }
    // =========================================================================
    // END DOPPELGÄNGER MODE — fall through to standard sparse sidecar path
    // =========================================================================

    if (helix_router_gate == nullptr || helix_cluster_map == nullptr ||
        helix_ffn_gate_exps == nullptr || helix_ffn_up_exps == nullptr || helix_ffn_down_exps == nullptr) {
        return nullptr;
    }

    if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
        HELIX_TRACE_FPRINTF("[HELIX TRACE] enter build_helix_dnpa_sparse_ffn layer %d\n", il);
    }

    // Router matmul must match PyTorch: logits = cur @ gate.T with gate [num_clusters, n_embd].
    // ggml_mul_mat(A, cur): out[c,t] = sum_h A[h,c] * cur[h,t]  =>  A is [n_embd, num_clusters].
    // Sidecar / gates.pt bytes are row-major gate[c,h] (stride n_embd between cluster rows).
    // GGUF may label that buffer ne=[num_clusters, n_embd]; a physical transpose scrambles coefficients.
    ggml_tensor * helix_gate_f32 = ggml_cont(ctx0, ggml_cast(ctx0, helix_router_gate, GGML_TYPE_F32));
    ggml_tensor * router   = nullptr;
    int64_t layer_clusters = 0;

    if (helix_router_gate->ne[0] == n_embd) {
        layer_clusters = helix_router_gate->ne[1];
        router = helix_gate_f32;
    } else if (helix_router_gate->ne[1] == n_embd) {
        layer_clusters = helix_router_gate->ne[0];
        router = ggml_view_2d(
            ctx0,
            helix_gate_f32,
            n_embd,
            layer_clusters,
            n_embd * ggml_element_size(helix_gate_f32),
            0);
    } else {
        return nullptr;
    }

    GGML_ASSERT(layer_clusters > 0);

    // Per-layer geometry: num_clusters from router gate; width from dense ffn_down intermediate dim.
    const int64_t num_clusters = layer_clusters;
    const int64_t n_intermediate_map = helix_cluster_map->ne[0];
    GGML_ASSERT(helix_cluster_map->type == GGML_TYPE_I32);
    const int64_t n_intermediate = (ffn_down != nullptr) ? ffn_down->ne[0] : n_intermediate_map;
    GGML_ASSERT(n_intermediate == n_intermediate_map);

    const int64_t neurons_per_cluster = (num_clusters > 0) ? (n_intermediate / num_clusters) : 0;
    const int64_t cluster_width = neurons_per_cluster;
    GGML_ASSERT(cluster_width > 0);
    GGML_ASSERT(cluster_width * num_clusters == n_intermediate);

    if (il == 19 && helix_trace_enabled()) {
        static bool l19_weights_dumped = false;
        LLAMA_LOG("=== LAYER 19 GEOMETRY TRACE ===\n");
        LLAMA_LOG("layer_clusters: %lld\n", (long long) num_clusters);
        LLAMA_LOG("cluster_width: %lld\n", (long long) neurons_per_cluster);
        LLAMA_LOG("ffn_down->ne[0]: %lld\n", ffn_down ? (long long) ffn_down->ne[0] : -1LL);
        LLAMA_LOG("ffn_down->ne[1]: %lld\n", ffn_down ? (long long) ffn_down->ne[1] : -1LL);
        LLAMA_LOG("helix_cluster_map->ne[0]: %lld\n", (long long) n_intermediate_map);
        HELIX_TRACE_FPRINTF("=== LAYER 19 GEOMETRY TRACE ===\n");
        HELIX_TRACE_FPRINTF("layer_clusters: %lld\n", (long long) num_clusters);
        HELIX_TRACE_FPRINTF("cluster_width: %lld\n", (long long) neurons_per_cluster);
        HELIX_TRACE_FPRINTF("ffn_down->ne[0]: %lld\n", ffn_down ? (long long) ffn_down->ne[0] : -1LL);
        HELIX_TRACE_FPRINTF("ffn_down->ne[1]: %lld\n", ffn_down ? (long long) ffn_down->ne[1] : -1LL);
        HELIX_TRACE_FPRINTF("helix_cluster_map->ne[0]: %lld\n", (long long) n_intermediate_map);
        if (!l19_weights_dumped) {
            l19_weights_dumped = true;
            helix_trace_dump_down_exps_raw_weights(helix_ffn_down_exps, il);
        }
    }
    // Gate/up now canonical: ne=[n_embd, W, K] (ne0=n_embd contraction axis).
    // Down keeps down-style ne=[W, n_embd, K] (ne0=W contraction axis).
    GGML_ASSERT(helix_ffn_gate_exps->ne[0] == n_embd);
    GGML_ASSERT(helix_ffn_gate_exps->ne[1] == cluster_width);
    GGML_ASSERT(helix_ffn_gate_exps->ne[2] == layer_clusters);
    GGML_ASSERT(helix_ffn_up_exps->ne[0] == n_embd);
    GGML_ASSERT(helix_ffn_up_exps->ne[1] == cluster_width);
    GGML_ASSERT(helix_ffn_up_exps->ne[2] == layer_clusters);
    GGML_ASSERT(helix_ffn_down_exps->ne[0] == cluster_width);
    GGML_ASSERT(helix_ffn_down_exps->ne[1] == n_embd);
    GGML_ASSERT(helix_ffn_down_exps->ne[2] == layer_clusters);

    // Path B: middle layers (3-18) route top-8; deep layers (19-27) route top-2.
    const int top_k = (il >= 19) ? 2 : 8;

    if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
        HELIX_TRACE_FPRINTF(
            "[HELIX TRACE BUILD] Layer %d | clusters=%lld | top_k=%d | cluster_width=%lld | "
            "router_raw_ne=[%lld,%lld] | gate_exps_ne=[%lld,%lld,%lld] | "
            "gate_input=post_ffn_rmsnorm\n",
            il,
            (long long) layer_clusters,
            top_k,
            (long long) cluster_width,
            (long long) helix_router_gate->ne[0],
            (long long) helix_router_gate->ne[1],
            (long long) helix_ffn_gate_exps->ne[0],
            (long long) helix_ffn_gate_exps->ne[1],
            (long long) helix_ffn_gate_exps->ne[2]);
    }

    // cur may be narrowed to n_outputs on the last layer via inp_out_ids — use its live width, not n_tokens.
    const int64_t n_ffn_tokens = cur->ne[1];
    GGML_ASSERT(n_ffn_tokens > 0);
    GGML_ASSERT(cur->ne[0] == n_embd);
    GGML_ASSERT(ggml_nelements(cur) == n_embd * n_ffn_tokens);

    const ggml_type cur_type = cur->type;

    // F16 norm weights (e.g. quantize-from-F16 GGUF) yield F16 activations; helix CUDA ops need F32.
    ggml_tensor * cur_f32 = ggml_cast(ctx0, cur, GGML_TYPE_F32);

    // hidden [n_embd, n_ffn_tokens] x router [n_embd, num_clusters] -> scores [num_clusters, n_ffn_tokens]
    ggml_tensor * router_f32 = router->type == GGML_TYPE_F32 ? router : ggml_cast(ctx0, router, GGML_TYPE_F32);
    ggml_tensor * gate_scores = ggml_mul_mat(ctx0, router_f32, cur_f32);
    cb(gate_scores, "helix_gate_scores", il);

    // Match Python reference: top-k on softmax probabilities, then argsort indices (cluster IDs).
    ggml_tensor * gate_probs = ggml_soft_max(ctx0, gate_scores);
    cb(gate_probs, "helix_gate_probs", il);
    ggml_tensor * selected_clusters = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, gate_probs, top_k));
    cb(selected_clusters, "helix_top_clusters", il);
    ggml_build_forward_expand(gf, selected_clusters);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur_f32, n_embd, 1, n_ffn_tokens);

    GGML_ASSERT(cluster_width % ggml_blck_size(helix_ffn_gate_exps->type) == 0);
    GGML_ASSERT(cluster_width % ggml_blck_size(helix_ffn_up_exps->type) == 0);
    GGML_ASSERT(cluster_width % ggml_blck_size(helix_ffn_down_exps->type) == 0);

    GGML_UNUSED(helix_cluster_map);

    // Gate/up are now stored canonically as ne=[n_embd, W, K] (ne0=n_embd = contraction axis),
    // exactly like stock llama.cpp ffn_gate_exps/ffn_up_exps. mul_mat_id contracts ne0 directly,
    // so NO runtime permute is needed and the Q8_0 path is preserved. The previous
    // cast->permute->cont path was geometrically correct (strides matched) but corrupted the
    // gate/up values; storing the correct layout on disk eliminates it. Down keeps [W, n_embd, K].
    ggml_tensor * gate_exps_permuted = helix_ffn_gate_exps;
    ggml_tensor * up_exps_permuted   = helix_ffn_up_exps;
    ggml_tensor * down_exps = helix_ffn_down_exps;

    if (il == 19 && helix_trace_enabled()) {
        ggml_set_name(gate_exps_permuted, "helix_gate_postpermute_l19");
        LLAMA_LOG("[HELIX TRACE] Layer 19 Post-Permute gate_exps shape: [%lld, %lld, %lld]\n",
                (long long) gate_exps_permuted->ne[0],
                (long long) gate_exps_permuted->ne[1],
                (long long) gate_exps_permuted->ne[2]);
        LLAMA_LOG("[HELIX TRACE] Layer 19 Post-Permute gate_exps strides: [%zu, %zu, %zu]\n",
                gate_exps_permuted->nb[0],
                gate_exps_permuted->nb[1],
                gate_exps_permuted->nb[2]);
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 Post-Permute gate_exps shape: [%lld, %lld, %lld]\n",
                (long long) gate_exps_permuted->ne[0],
                (long long) gate_exps_permuted->ne[1],
                (long long) gate_exps_permuted->ne[2]);
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 Post-Permute gate_exps strides: [%zu, %zu, %zu]\n",
                gate_exps_permuted->nb[0],
                gate_exps_permuted->nb[1],
                gate_exps_permuted->nb[2]);
        LLAMA_LOG("[HELIX TRACE] Layer 19 On-disk gate_exps shape: [%lld, %lld, %lld] strides: [%zu, %zu, %zu]\n",
                (long long) helix_ffn_gate_exps->ne[0],
                (long long) helix_ffn_gate_exps->ne[1],
                (long long) helix_ffn_gate_exps->ne[2],
                helix_ffn_gate_exps->nb[0],
                helix_ffn_gate_exps->nb[1],
                helix_ffn_gate_exps->nb[2]);
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 On-disk gate_exps shape: [%lld, %lld, %lld] strides: [%zu, %zu, %zu]\n",
                (long long) helix_ffn_gate_exps->ne[0],
                (long long) helix_ffn_gate_exps->ne[1],
                (long long) helix_ffn_gate_exps->ne[2],
                helix_ffn_gate_exps->nb[0],
                helix_ffn_gate_exps->nb[1],
                helix_ffn_gate_exps->nb[2]);
        for (int64_t expert_id : {15, 17, 28, 30}) {
            const size_t on_disk_off = (size_t) expert_id * helix_ffn_gate_exps->nb[2];
            const size_t permuted_off = (size_t) expert_id * gate_exps_permuted->nb[2];
            const int64_t dense_off = expert_id * cluster_width;
            LLAMA_LOG(
                "[HELIX TRACE] Layer 19 expert %lld byte offset: on_disk=%zu permuted_f32=%zu "
                "dense_ffn_intermediate=%lld (slot*cluster_width)\n",
                (long long) expert_id,
                on_disk_off,
                permuted_off,
                (long long) dense_off);
            HELIX_TRACE_FPRINTF(
                "[HELIX TRACE] Layer 19 expert %lld byte offset: on_disk=%zu permuted_f32=%zu "
                "dense_ffn_intermediate=%lld (slot*cluster_width)\n",
                (long long) expert_id,
                on_disk_off,
                permuted_off,
                (long long) dense_off);
        }
    }

    ggml_tensor * gate_exps = gate_exps_permuted;
    ggml_tensor * up_exps   = up_exps_permuted;

    if (il == 19 && helix_trace_enabled()) {
        helix_trace_dump_mul_mat_id_ids(selected_clusters, il, top_k, false);
        if (top_k == 1) {
            LLAMA_LOG("  output merge: ggml_view_2d byte offset=0 (single expert output, not id-based weight slice)\n");
            HELIX_TRACE_FPRINTF("  output merge: ggml_view_2d byte offset=0 (single expert output, not id-based weight slice)\n");
        } else {
            LLAMA_LOG("  output merge: ggml_view_2d byte offsets 0 and experts->nb[1] (sum two expert outputs)\n");
            HELIX_TRACE_FPRINTF("  output merge: ggml_view_2d byte offsets 0 and experts->nb[1] (sum two expert outputs)\n");
        }
    }

    ggml_tensor * act_gate = build_lora_mm_id(gate_exps, cur_3d, selected_clusters);
    cb(act_gate, "helix_act_gate", il);
    ggml_tensor * act_up = build_lora_mm_id(up_exps, cur_3d, selected_clusters);
    cb(act_up, "helix_act_up", il);
    if (il == 19 && helix_trace_enabled()) {
        ggml_set_name(act_gate, "helix_gate_raw_l19");
        ggml_set_name(act_up, "helix_up_raw_l19");
        LLAMA_LOG("[HELIX TRACE] Layer 19 Raw Gate and Up tagged\n");
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 Raw Gate and Up tagged\n");
    }

    ggml_tensor * gate_f32 = ggml_cont(ctx0, ggml_cast(ctx0, act_gate, GGML_TYPE_F32));
    ggml_tensor * up_f32   = ggml_cont(ctx0, ggml_cast(ctx0, act_up,   GGML_TYPE_F32));
    ggml_tensor * swiglu   = ggml_swiglu_split(ctx0, gate_f32, up_f32);
    cb(swiglu, "helix_swiglu", il);
    if (il == 19 && helix_trace_enabled()) {
        ggml_set_name(swiglu, "helix_swiglu_l19_tok3");
        LLAMA_LOG("[HELIX TRACE] Layer 19 Pre-Down SwiGLU tagged for capture\n");
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 Pre-Down SwiGLU tagged for capture\n");
    }

    ggml_tensor * experts = build_lora_mm_id(down_exps, swiglu, selected_clusters);
    experts = ggml_cont(ctx0, ggml_cast(ctx0, experts, GGML_TYPE_F32));
    cb(experts, "helix_down", il);
    if (il == 19 && helix_trace_enabled()) {
        ggml_format_name(experts, "helix_debug_down_out-%d", il);
        LLAMA_LOG("[HELIX TRACE] Layer 19 Post-Down-Proj tensor tagged for inspection\n");
        LLAMA_LOG("[HELIX TRACE] Expected shape: [%lld, %lld, %lld]\n",
                (long long) experts->ne[0], (long long) experts->ne[1], (long long) experts->ne[2]);
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer 19 Post-Down-Proj tensor tagged for inspection\n");
        HELIX_TRACE_FPRINTF("[HELIX TRACE] Expected shape: [%lld, %lld, %lld]\n",
                (long long) experts->ne[0], (long long) experts->ne[1], (long long) experts->ne[2]);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * out = nullptr;

    if (top_k == 1) {
        out = ggml_cont(ctx0, ggml_view_2d(ctx0, experts, n_embd, n_ffn_tokens, experts->nb[2], 0));
        cb(out, "helix_sparse_out", il);
        ggml_build_forward_expand(gf, out);
    } else {
        ggml_tensor * cur_experts[8] = { nullptr };
        GGML_ASSERT(top_k >= 2 && top_k <= 8);

        for (int i = 0; i < top_k; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_ffn_tokens, experts->nb[2], i * experts->nb[1]);
            cb(cur_experts[i], "helix_ffn_out_expert", il);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }

        out = cur_experts[0];
        for (int i = 1; i < top_k; ++i) {
            out = ggml_add(ctx0, out, cur_experts[i]);
            ggml_build_forward_expand(gf, out);
        }
        out = ggml_cont(ctx0, out);
        cb(out, "helix_sparse_out", il);
    }

    // Active Circuit Mirror: per-expert complement circuits routed by the SAME cluster IDs.
    // Each expert c has a tiny rank-R SwiGLU correction trained on what expert c specifically misses.
    // Uses mul_mat_id with same selected_clusters — different tokens get different corrections.
    if (helix_shared_core_gate != nullptr &&
        helix_shared_core_up   != nullptr &&
        helix_shared_core_down != nullptr &&
        helix_shared_core_gate->ne[2] == num_clusters) {
        // 3D expert-indexed layout: ne=[n_embd, rank, K] for gate/up, [rank, n_embd, K] for down
        ggml_tensor * mc_act_gate = build_lora_mm_id(helix_shared_core_gate, cur_3d, selected_clusters);
        cb(mc_act_gate, "helix_mc_gate", il);
        ggml_tensor * mc_act_up = build_lora_mm_id(helix_shared_core_up, cur_3d, selected_clusters);
        cb(mc_act_up, "helix_mc_up", il);

        ggml_tensor * mc_gate_f32 = ggml_cont(ctx0, ggml_cast(ctx0, mc_act_gate, GGML_TYPE_F32));
        ggml_tensor * mc_up_f32   = ggml_cont(ctx0, ggml_cast(ctx0, mc_act_up,   GGML_TYPE_F32));
        ggml_tensor * mc_swiglu   = ggml_swiglu_split(ctx0, mc_gate_f32, mc_up_f32);
        cb(mc_swiglu, "helix_mc_swiglu", il);

        ggml_tensor * mc_down = build_lora_mm_id(helix_shared_core_down, mc_swiglu, selected_clusters);
        mc_down = ggml_cont(ctx0, ggml_cast(ctx0, mc_down, GGML_TYPE_F32));
        cb(mc_down, "helix_mc_down", il);
        ggml_build_forward_expand(gf, mc_down);

        // Sum mirror expert outputs (same merge pattern as main sparse path)
        ggml_tensor * mc_out = nullptr;
        if (top_k == 1) {
            mc_out = ggml_cont(ctx0, ggml_view_2d(ctx0, mc_down, n_embd, n_ffn_tokens, mc_down->nb[2], 0));
        } else {
            ggml_tensor * mc_experts[8] = { nullptr };
            for (int i = 0; i < top_k; ++i) {
                mc_experts[i] = ggml_view_2d(ctx0, mc_down, n_embd, n_ffn_tokens, mc_down->nb[2], i * mc_down->nb[1]);
                ggml_build_forward_expand(gf, mc_experts[i]);
            }
            mc_out = mc_experts[0];
            for (int i = 1; i < top_k; ++i) {
                mc_out = ggml_add(ctx0, mc_out, mc_experts[i]);
                ggml_build_forward_expand(gf, mc_out);
            }
            mc_out = ggml_cont(ctx0, mc_out);
        }
        cb(mc_out, "helix_mc_out", il);

        out = ggml_add(ctx0, out, mc_out);
        cb(out, "helix_ffn_out_combined", il);
        ggml_build_forward_expand(gf, out);

        if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
            HELIX_TRACE_FPRINTF("[HELIX TRACE] Layer %d mirror_core: gate=[%lld,%lld,%lld] down=[%lld,%lld,%lld]\n",
                    il,
                    (long long) helix_shared_core_gate->ne[0], (long long) helix_shared_core_gate->ne[1],
                    (long long) helix_shared_core_gate->ne[2],
                    (long long) helix_shared_core_down->ne[0], (long long) helix_shared_core_down->ne[1],
                    (long long) helix_shared_core_down->ne[2]);
        }
    } else if (helix_shared_core_gate != nullptr &&
               helix_shared_core_up   != nullptr &&
               helix_shared_core_down != nullptr) {
        // Legacy 2D fallback: static un-routed adapter
        ggml_tensor * sc_gate_f32 = ggml_cont(ctx0, ggml_cast(ctx0, helix_shared_core_gate, GGML_TYPE_F32));
        ggml_tensor * sc_up_f32   = ggml_cont(ctx0, ggml_cast(ctx0, helix_shared_core_up,   GGML_TYPE_F32));
        ggml_tensor * sc_down_f32 = ggml_cont(ctx0, ggml_cast(ctx0, helix_shared_core_down, GGML_TYPE_F32));
        ggml_tensor * sc_act_gate = ggml_mul_mat(ctx0, sc_gate_f32, cur_f32);
        ggml_tensor * sc_act_up   = ggml_mul_mat(ctx0, sc_up_f32,   cur_f32);
        ggml_tensor * sc_swiglu   = ggml_swiglu_split(ctx0, sc_act_gate, sc_act_up);
        ggml_tensor * sc_out      = ggml_mul_mat(ctx0, sc_down_f32, sc_swiglu);
        sc_out = ggml_cont(ctx0, ggml_cast(ctx0, sc_out, GGML_TYPE_F32));
        out = ggml_add(ctx0, out, sc_out);
        cb(out, "helix_ffn_out_combined", il);
        ggml_build_forward_expand(gf, out);
    }

    cb(out, "helix_ffn_out", il);
    if (out->type != cur_type) {
        out = ggml_cast(ctx0, out, cur_type);
        cb(out, "helix_ffn_out_cast", il);
    }
    return out;
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il,
         ggml_tensor * helix_router_gate,
         ggml_tensor * helix_cluster_map,
         ggml_tensor * helix_ffn_gate_exps,
         ggml_tensor * helix_ffn_up_exps,
         ggml_tensor * helix_ffn_down_exps,
         ggml_tensor * helix_shared_core_gate,
         ggml_tensor * helix_shared_core_up,
         ggml_tensor * helix_shared_core_down,
         ggml_tensor * helix_magnet_a,
         ggml_tensor * helix_magnet_b,
         ggml_tensor * helix_ffn_gate_live,
         ggml_tensor * helix_ffn_up_live,
         ggml_tensor * helix_ffn_down_live) const {
    ggml_tensor * ffn_up   = up;
    ggml_tensor * ffn_gate = gate;
    ggml_tensor * ffn_down = down;

    const bool helix_qwen_ffn = ffn_down != nullptr && ffn_gate != nullptr && ffn_up != nullptr &&
        ffn_down->ne[0] == 6144 && ffn_down->ne[1] == 2048 &&
        ffn_gate->ne[0] == 2048 && ffn_gate->ne[1] == 6144;

    // Magnet-only GGUF (helix_magnet_a/b embedded, no DNPA sidecar tensors)
    if (helix_doppelganger_enabled() && helix_qwen_ffn &&
        helix_magnet_a != nullptr && helix_magnet_b != nullptr) {
        ggml_tensor * helix_out = build_helix_dnpa_sparse_ffn(
            ctx0, cur, ffn_up, ffn_gate, ffn_down,
            nullptr, nullptr, nullptr,
            nullptr, nullptr, il,
            nullptr, nullptr, nullptr,
            helix_magnet_a, helix_magnet_b,
            helix_ffn_gate_live, helix_ffn_up_live, helix_ffn_down_live);
        if (helix_out != nullptr) {
            return helix_out;
        }
    }

    // DNPA Helix sparse routing: triple sidecar MoE gate/up/down stacks
    if (helix_router_gate != nullptr && helix_cluster_map != nullptr &&
        helix_ffn_gate_exps != nullptr && helix_ffn_up_exps != nullptr && helix_ffn_down_exps != nullptr &&
        ffn_down != nullptr && ffn_gate != nullptr && ffn_up != nullptr &&
        helix_qwen_ffn) {
        if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
            HELIX_TRACE_FPRINTF("[HELIX TRACE] build_ffn -> sparse helix layer %d\n", il);
        }
        return build_helix_dnpa_sparse_ffn(
            ctx0, cur, ffn_up, ffn_gate, ffn_down,
            helix_ffn_gate_exps, helix_ffn_up_exps, helix_ffn_down_exps,
            helix_router_gate, helix_cluster_map, il,
            helix_shared_core_gate, helix_shared_core_up, helix_shared_core_down,
            helix_magnet_a, helix_magnet_b,
            helix_ffn_gate_live, helix_ffn_up_live, helix_ffn_down_live);
    }
    if (helix_trace_file() != nullptr && helix_trace_layer_wanted(il)) {
        HELIX_TRACE_FPRINTF(
            "[HELIX TRACE] build_ffn -> DENSE fallback layer %d | helix_gate=%p map=%p "
            "gate_exps=%p up_exps=%p down_exps=%p | down_ne=[%lld,%lld] gate_ne=[%lld,%lld]\n",
            il,
            (void *) helix_router_gate,
            (void *) helix_cluster_map,
            (void *) helix_ffn_gate_exps,
            (void *) helix_ffn_up_exps,
            (void *) helix_ffn_down_exps,
            ffn_down ? (long long) ffn_down->ne[0] : -1LL,
            ffn_down ? (long long) ffn_down->ne[1] : -1LL,
            ffn_gate ? (long long) ffn_gate->ne[0] : -1LL,
            ffn_gate ? (long long) ffn_gate->ne[1] : -1LL);
    }

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                // Step35: HF clamps gate (after SiLU) and up before multiplication
                if (arch == LLM_ARCH_STEP35 && il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        cb(gate_act, "ffn_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        cb(gate_act, "ffn_silu_clamped", il);

                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, tmp);
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        // apply per-expert scale2 to merged gate_up (use up_exps_s since gate and up are fused)
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            gate_up = ggml_mul(ctx0, gate_up, s);
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        // apply per-expert scale2 to up
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            up = ggml_mul(ctx0, up, s);
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }

        // apply per-expert scale2 to gate
        if (gate_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            cur = ggml_mul(ctx0, cur, s);
            cb(cur, "ffn_moe_gate_scaled", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                // Step35: per-layer clamp for routed experts
                if (arch == LLM_ARCH_STEP35 && il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        cb(gate_act, "ffn_moe_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        cb(gate_act, "ffn_moe_silu_clamped", il);

                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, up);
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    // apply per-expert scale2 to down
    if (down_exps_s) {
        ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
        experts = ggml_mul(ctx0, experts, s);
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il);

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, inp->self_k_rot);
        k_cur = ggml_mul_mat_aux(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = ggml_mul_mat_aux(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = ggml_mul_mat_aux(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = ggml_mul_mat_aux(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure F32 mask
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = false;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    int32_t logit_row_idx = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            seq_to_logit_row[seq_id] = logit_row_idx;
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ nullptr,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
