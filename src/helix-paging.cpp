#include "helix-paging.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

static bool helix_env_flag_active(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

bool helix_magnet_paging_requested(void) {
    return helix_env_flag_active("HELIX_DOPPELGANGER") && helix_env_flag_active("HELIX_MAGNET_PAGED");
}

static float helix_magnet_max_frac(void) {
    const char * v = std::getenv("HELIX_MAGNET_MAX_FRAC");
    if (v != nullptr && v[0] != '\0') {
        const float f = (float) atof(v);
        if (f > 0.0f && f <= 1.0f) {
            return f;
        }
    }
    const char * w = std::getenv("HELIX_MAGNET_WIDTH_FRAC");
    if (w != nullptr && w[0] != '\0') {
        const float f = (float) atof(w);
        if (f > 0.0f && f <= 1.0f) {
            return f;
        }
    }
    return 0.25f;
}

int64_t helix_magnet_k_max(int64_t n_ff) {
    int64_t k = (int64_t) (helix_magnet_max_frac() * (float) n_ff + 0.5f);
    if (k < 1) {
        k = 1;
    }
    if (k > n_ff) {
        k = n_ff;
    }
    return k;
}

struct ggml_tensor * helix_graph_paged_pack_rows(
        ggml_context * ctx0,
        ggml_tensor * weight_3d,
        ggml_tensor * selected,
        ggml_tensor * live_3d,
        const int64_t  gather_k) {
    GGML_ASSERT(weight_3d != nullptr && selected != nullptr && live_3d != nullptr);
    GGML_ASSERT(gather_k >= 1 && gather_k <= live_3d->ne[2]);

    ggml_tensor * rows = ggml_get_rows(ctx0, weight_3d, selected);

    const int64_t n0 = live_3d->ne[0];
    ggml_tensor * live_view = ggml_view_3d(
            ctx0, live_3d, n0, 1, gather_k, live_3d->nb[0], live_3d->nb[1], 0);

    return ggml_cpy(ctx0, rows, live_view);
}

struct ggml_tensor * helix_graph_seq_indices(
        ggml_context * ctx0,
        const int64_t  k,
        const int64_t  n_tokens) {
    ggml_tensor * id_f32 = ggml_arange(ctx0, 0.f, (float) k, 1.f);
    ggml_tensor * id_i32 = ggml_cast(ctx0, id_f32, GGML_TYPE_I32);
    id_i32 = ggml_reshape_2d(ctx0, id_i32, k, 1);
    if (n_tokens <= 1) {
        return id_i32;
    }
    ggml_tensor * tmpl = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, k, n_tokens);
    return ggml_repeat(ctx0, id_i32, tmpl);
}

struct llama_helix_paging_pool {
    ggml_context_ptr          ctx;
    ggml_backend_buffer_ptr   buf;
    bool                      ready = false;
};

static llama_helix_paging_pool g_helix_paging;

void llama_model_free_helix_paging_buffers(llama_model & model) {
    (void) model;
    for (auto & layer : model.layers) {
        layer.helix_ffn_gate_live  = nullptr;
        layer.helix_ffn_up_live    = nullptr;
        layer.helix_ffn_down_live  = nullptr;
    }
    g_helix_paging.buf.reset();
    g_helix_paging.ctx.reset();
    g_helix_paging.ready = false;
}

bool llama_model_init_helix_paging_buffers(llama_model & model) {
    if (!helix_magnet_paging_requested()) {
        return true;
    }

    if (g_helix_paging.ready) {
        return true;
    }

    const int n_layer = (int) model.hparams.n_layer;

    struct layer_plan {
        int           il;
        int64_t       n_embd;
        int64_t       n_ff;
        int64_t       k_max;
    };

    std::vector<layer_plan> plans;
    plans.reserve(32);

    for (int il = 3; il < n_layer; ++il) {
        auto & layer = model.layers[il];
        if (layer.helix_magnet_a == nullptr || layer.helix_magnet_b == nullptr) {
            continue;
        }
        if (layer.ffn_gate == nullptr || layer.ffn_up == nullptr || layer.ffn_down == nullptr) {
            continue;
        }

        const int64_t n_embd = layer.ffn_gate->ne[0];
        const int64_t n_ff   = layer.ffn_gate->ne[1];
        const int64_t k_max  = helix_magnet_k_max(n_ff);

        plans.push_back({ il, n_embd, n_ff, k_max });
    }

    if (plans.empty()) {
        return true;
    }

    const size_t n_live_tensors = plans.size() * 3;
    const size_t ctx_overhead   = n_live_tensors * ggml_tensor_overhead() + 4096;

    ggml_backend_buffer_type_t buft = model.select_buft(plans[0].il);
    ggml_backend_dev_t dev = nullptr;
    if (ggml_backend_buft_get_device(buft) != nullptr) {
        dev = ggml_backend_buft_get_device(buft);
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_overhead,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    g_helix_paging.ctx.reset(ggml_init(params));
    if (!g_helix_paging.ctx) {
        return false;
    }

    for (const auto & p : plans) {
        auto & layer = model.layers[p.il];

        layer.helix_ffn_gate_live = ggml_new_tensor_2d(
                g_helix_paging.ctx.get(), GGML_TYPE_F32, p.n_embd, p.k_max);
        layer.helix_ffn_up_live = ggml_new_tensor_2d(
                g_helix_paging.ctx.get(), GGML_TYPE_F32, p.n_embd, p.k_max);
        layer.helix_ffn_down_live = ggml_new_tensor_2d(
                g_helix_paging.ctx.get(), GGML_TYPE_F32, p.k_max, p.n_embd);

        for (ggml_tensor * t : { layer.helix_ffn_gate_live, layer.helix_ffn_up_live, layer.helix_ffn_down_live }) {
            ggml_set_name(t, "helix_ffn_live");
        }
    }

    g_helix_paging.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(g_helix_paging.ctx.get(), buft));
    if (!g_helix_paging.buf) {
        g_helix_paging.ctx.reset();
        return false;
    }

    ggml_backend_buffer_set_usage(g_helix_paging.buf.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    g_helix_paging.ready = true;

    const char * dev_name = dev ? ggml_backend_dev_name(dev) : "host";
    const size_t buf_bytes = ggml_backend_buffer_get_size(g_helix_paging.buf.get());
    LLAMA_LOG_INFO(
            "%s: HELIX_MAGNET_PAGED — %zu layer scratchpads, k_max<=25%% n_ff, "
            "%.2f MiB on %s (get_rows+copy per forward)\n",
            __func__, plans.size(),
            (double) (buf_bytes / 1024.0 / 1024.0), dev_name);

    return true;
}
