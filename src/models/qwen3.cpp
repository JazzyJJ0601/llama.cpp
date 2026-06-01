#include "models.h"

#include <cstdlib>
#include <cstdio>

namespace {

ggml_tensor * load_helix_tensor_opt(
        llama_model_qwen3 & model,
        llama_model_loader & ml,
        llm_tensor         tensor,
        int                layer) {
    const LLM_TN_IMPL tn(model.model->arch, tensor, "weight", layer, 0);
    const char * name = tn.str().c_str();
    const ggml_tensor * meta = ml.get_tensor_meta(name);
    if (meta == nullptr) {
        if (std::getenv("HELIX_TRACE")) {
            fprintf(stderr, "[HELIX TRACE LOAD] missing tensor meta: %s\n", name);
        }
        return nullptr;
    }

    int rank = GGML_MAX_DIMS;
    while (rank > 1 && meta->ne[rank - 1] == 1) {
        --rank;
    }

    ggml_tensor * out = nullptr;
    switch (rank) {
        case 1:
            out = model.create_tensor_cpu(tn, { meta->ne[0] }, llama_model_loader::TENSOR_NOT_REQUIRED);
            break;
        case 2:
            out = model.create_tensor_cpu(tn, { meta->ne[0], meta->ne[1] }, llama_model_loader::TENSOR_NOT_REQUIRED);
            break;
        case 3:
            out = model.create_tensor_cpu(tn, { meta->ne[0], meta->ne[1], meta->ne[2] }, llama_model_loader::TENSOR_NOT_REQUIRED);
            break;
        default:
            out = model.create_tensor_cpu(tn, { meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3] },
                    llama_model_loader::TENSOR_NOT_REQUIRED);
            break;
    }

    if (out == nullptr) {
        LLAMA_LOG_WARN("%s: failed to bind Helix tensor %s (type=%s ne=[%lld,%lld,%lld,%lld])\n",
                __func__, name, ggml_type_name(meta->type),
                (long long) meta->ne[0], (long long) meta->ne[1],
                (long long) meta->ne[2], (long long) meta->ne[3]);
        if (std::getenv("HELIX_TRACE")) {
            fprintf(stderr, "[HELIX TRACE LOAD] create_tensor failed: %s ne=[%lld,%lld]\n",
                    name, (long long) meta->ne[0], (long long) meta->ne[1]);
        }
    }
    return out;
}

} // namespace

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.helix_router_gate = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_ROUTER_GATE, i);
        layer.helix_cluster_map = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_CLUSTER_MAP, i);
        layer.helix_ffn_gate_exps = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_FFN_GATE_EXPS, i);
        layer.helix_ffn_up_exps   = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_FFN_UP_EXPS, i);
        layer.helix_ffn_down_exps = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_FFN_DOWN_EXPS, i);
        layer.helix_shared_core_gate = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_SHARED_CORE_GATE, i);
        layer.helix_shared_core_up   = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_SHARED_CORE_UP, i);
        layer.helix_shared_core_down = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_SHARED_CORE_DOWN, i);
        layer.helix_magnet_a = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_MAGNET_A, i);
        layer.helix_magnet_b = load_helix_tensor_opt(*this, ml, LLM_TENSOR_HELIX_MAGNET_B, i);
    }

    if (std::getenv("HELIX_DOPPELGANGER") != nullptr) {
        int n_magnet_layers = 0;
        for (int i = 0; i < n_layer; ++i) {
            if (layers[i].helix_magnet_a != nullptr && layers[i].helix_magnet_b != nullptr) {
                ++n_magnet_layers;
            }
        }
        LLAMA_LOG_INFO("%s: HELIX_DOPPELGANGER — magnet tensors on %d / %d layers\n",
                __func__, n_magnet_layers, n_layer);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il,
                model.layers[il].helix_router_gate,
                model.layers[il].helix_cluster_map,
                model.layers[il].helix_ffn_gate_exps,
                model.layers[il].helix_ffn_up_exps,
                model.layers[il].helix_ffn_down_exps,
                model.layers[il].helix_shared_core_gate,
                model.layers[il].helix_shared_core_up,
                model.layers[il].helix_shared_core_down,
                model.layers[il].helix_magnet_a,
                model.layers[il].helix_magnet_b,
                model.layers[il].helix_ffn_gate_live,
                model.layers[il].helix_ffn_up_live,
                model.layers[il].helix_ffn_down_live);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
