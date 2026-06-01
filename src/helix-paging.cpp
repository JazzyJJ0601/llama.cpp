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
    // Paged mode via env var (backward compat)
    if (helix_env_flag_active("HELIX_MAGNET_PAGED")) {
        return true;
    }
    // Also activate when FFN weights are offloaded to CPU via -ot/-CpuFfn.
    // Detected by HELIX_MAGNET_CPU_FFN which the launcher sets.
    if (helix_env_flag_active("HELIX_MAGNET_CPU_FFN")) {
        return true;
    }
    return false;
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
        ggml_tensor * weight_2d,
        ggml_tensor * selected_1d,
        ggml_tensor * live_2d,
        const int64_t  gather_k) {
    GGML_ASSERT(weight_2d != nullptr && selected_1d != nullptr && live_2d != nullptr);
    GGML_ASSERT(gather_k >= 1);

    // get_rows on 2D [ne0, n_rows] with 1D indices [gather_k] selects
    // gather_k rows → output is [ne0, gather_k] in F32.
    ggml_tensor * rows = ggml_get_rows(ctx0, weight_2d, selected_1d);

    // Copy gathered rows into the GPU live buffer (may convert F32→F16).
    ggml_tensor * live_view = ggml_view_2d(
            ctx0, live_2d, live_2d->ne[0], gather_k,
            live_2d->nb[1], 0);

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

// --- Delta neuron cache ---------------------------------------------------

static helix_neuron_cache g_neuron_cache;

helix_neuron_cache & helix_get_neuron_cache(void) {
    return g_neuron_cache;
}

void helix_neuron_cache_init(helix_neuron_cache & cache, int n_layer, int64_t k_max) {
    cache.layers.resize((size_t) n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & lc = cache.layers[il];
        lc.k_max = k_max;
        lc.indices.assign((size_t) k_max, -1);
        lc.warm = false;
    }
    helix_neuron_cache_reset_stats(cache);
}

void helix_neuron_cache_reset_stats(helix_neuron_cache & cache) {
    cache.total_fetched = 0;
    cache.total_hits    = 0;
    cache.total_updates = 0;
}

helix_cache_delta helix_neuron_cache_update(
        helix_neuron_cache & cache,
        int                  il,
        const int32_t      * new_indices,
        int64_t              gather_k) {
    helix_cache_delta delta;
    delta.n_hits = 0;
    delta.n_miss = 0;

    if (il < 0 || il >= (int) cache.layers.size()) {
        // Layer not cached — fetch everything
        delta.fetch_indices.resize((size_t) gather_k);
        delta.fetch_slots.resize((size_t) gather_k);
        for (int64_t i = 0; i < gather_k; ++i) {
            delta.fetch_indices[i] = new_indices[i];
            delta.fetch_slots[i]   = (int32_t) i;
        }
        delta.n_miss = gather_k;
        return delta;
    }

    auto & lc = cache.layers[il];

    if (!lc.warm) {
        // First token — cache cold, fetch everything
        delta.fetch_indices.resize((size_t) gather_k);
        delta.fetch_slots.resize((size_t) gather_k);
        for (int64_t i = 0; i < gather_k; ++i) {
            delta.fetch_indices[i] = new_indices[i];
            delta.fetch_slots[i]   = (int32_t) i;
        }
        delta.n_miss = gather_k;

        // Store new state
        for (int64_t i = 0; i < gather_k; ++i) {
            lc.indices[i] = new_indices[i];
        }
        for (int64_t i = gather_k; i < lc.k_max; ++i) {
            lc.indices[i] = -1;
        }
        lc.warm = true;

        cache.total_fetched += (uint64_t) gather_k;
        cache.total_updates++;
        return delta;
    }

    // Build lookup of what's currently cached: index → slot
    // Use a simple linear scan since k_max is small (~1350)
    std::vector<bool> new_needed((size_t) gather_k, true);
    std::vector<bool> old_keep(lc.indices.size(), false);

    for (int64_t i = 0; i < gather_k; ++i) {
        const int32_t want = new_indices[i];
        for (int64_t j = 0; j < lc.k_max; ++j) {
            if (lc.indices[j] == want) {
                // Cache hit — this neuron is already in slot j
                new_needed[i] = false;
                old_keep[j] = true;
                delta.n_hits++;
                break;
            }
        }
    }

    // Collect free slots (evicted neurons)
    std::vector<int32_t> free_slots;
    for (int64_t j = 0; j < lc.k_max; ++j) {
        if (!old_keep[j]) {
            free_slots.push_back((int32_t) j);
        }
    }

    // Assign new neurons to free slots
    size_t free_idx = 0;
    for (int64_t i = 0; i < gather_k; ++i) {
        if (new_needed[i]) {
            int32_t slot = (free_idx < free_slots.size())
                ? free_slots[free_idx++]
                : (int32_t) i;  // fallback
            delta.fetch_indices.push_back(new_indices[i]);
            delta.fetch_slots.push_back(slot);
            lc.indices[slot] = new_indices[i];
            delta.n_miss++;
        }
    }

    cache.total_fetched += (uint64_t) delta.n_miss;
    cache.total_hits    += (uint64_t) delta.n_hits;
    cache.total_updates++;

    return delta;
}

bool helix_neuron_cache_get_stats(helix_cache_stats * out) {
    if (out == nullptr) {
        return false;
    }
    const auto & c = g_neuron_cache;
    out->total_fetched = c.total_fetched;
    out->total_hits    = c.total_hits;
    out->total_updates = c.total_updates;
    const uint64_t total = c.total_fetched + c.total_hits;
    out->hit_rate_pct = total > 0 ? 100.0 * (double) c.total_hits / (double) total : 0.0;
    return c.total_updates > 0;
}

// --- Paging pool ----------------------------------------------------------

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
    if (g_helix_paging.ready) {
        return true;
    }

    // Check if paging is needed: either explicitly requested via env var,
    // or auto-detected when FFN weights landed on CPU (from -ot flag).
    bool needs_paging = helix_magnet_paging_requested();
    if (!needs_paging) {
        // Auto-detect: check if any magnet layer's FFN weights are on a
        // different backend than the first layer (indicating CPU offload).
        for (int il = 3; il < (int) model.hparams.n_layer && !needs_paging; ++il) {
            auto & layer = model.layers[il];
            if (layer.helix_magnet_a != nullptr && layer.ffn_gate != nullptr) {
                if (layer.ffn_gate->buffer != nullptr &&
                    ggml_backend_buffer_is_host(layer.ffn_gate->buffer)) {
                    needs_paging = true;
                }
            }
        }
    }
    if (!needs_paging) {
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
                g_helix_paging.ctx.get(), GGML_TYPE_F16, p.n_embd, p.k_max);
        layer.helix_ffn_up_live = ggml_new_tensor_2d(
                g_helix_paging.ctx.get(), GGML_TYPE_F16, p.n_embd, p.k_max);
        layer.helix_ffn_down_live = ggml_new_tensor_2d(
                g_helix_paging.ctx.get(), GGML_TYPE_F16, p.n_embd, p.k_max);

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

    // Initialize the delta neuron cache
    const int64_t cache_k_max = plans.empty() ? 0 : plans[0].k_max;
    helix_neuron_cache_init(g_neuron_cache, n_layer, cache_k_max);

    const char * dev_name = dev ? ggml_backend_dev_name(dev) : "host";
    const size_t buf_bytes = ggml_backend_buffer_get_size(g_helix_paging.buf.get());
    LLAMA_LOG_INFO(
            "%s: HELIX_MAGNET_PAGED — %zu layer scratchpads, k_max<=25%% n_ff, "
            "%.2f MiB on %s (delta neuron cache enabled)\n",
            __func__, plans.size(),
            (double) (buf_bytes / 1024.0 / 1024.0), dev_name);

    return true;
}
