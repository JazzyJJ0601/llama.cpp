#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

struct llama_model;

// Phase 2: fixed GPU scratchpads + get_rows/cpy from CPU-backed FFN weights.
bool helix_magnet_paging_requested(void);
bool llama_model_init_helix_paging_buffers(struct llama_model & model);
void llama_model_free_helix_paging_buffers(struct llama_model & model);

int64_t helix_magnet_k_max(int64_t n_ff);

// Pack magnet-selected rows from CPU weight [n0,1,n_src] into GPU live [n0,1,k_max].
struct ggml_tensor * helix_graph_paged_pack_rows(
        struct ggml_context * ctx0,
        struct ggml_tensor * weight_3d,
        struct ggml_tensor * selected,
        struct ggml_tensor * live_3d,
        int64_t                gather_k);

struct ggml_tensor * helix_graph_seq_indices(
        struct ggml_context * ctx0,
        int64_t                k,
        int64_t                n_tokens);

// --- Delta neuron cache ---------------------------------------------------
// Tracks which neuron indices are resident in each layer's GPU scratchpad
// between tokens.  Computes set-diff to identify which rows actually need
// PCIe transfer (typically ~5% between consecutive tokens).

struct helix_neuron_cache_layer {
    std::vector<int32_t> indices;   // [k_max] neuron index per slot, -1 = empty
    int64_t              k_max = 0;
    bool                 warm  = false;
};

struct helix_neuron_cache {
    std::vector<helix_neuron_cache_layer> layers; // indexed by layer id
    uint64_t total_fetched  = 0;  // cumulative rows transferred
    uint64_t total_hits     = 0;  // cumulative cache hits (rows NOT transferred)
    uint64_t total_updates  = 0;  // number of cache_update calls
};

void helix_neuron_cache_init(helix_neuron_cache & cache, int n_layer, int64_t k_max);
void helix_neuron_cache_reset_stats(helix_neuron_cache & cache);

struct helix_cache_delta {
    std::vector<int32_t> fetch_indices;  // neuron indices that need transfer
    std::vector<int32_t> fetch_slots;    // scratchpad slots to write them to
    int64_t n_hits;                      // cache hits this update
    int64_t n_miss;                      // cache misses (= fetch_indices.size())
};

// Compare new selected indices against cached state.  Returns the delta:
// which neurons need fetching and into which scratchpad slots.
// After calling, the cache is updated to reflect the new state.
helix_cache_delta helix_neuron_cache_update(
        helix_neuron_cache & cache,
        int                  il,
        const int32_t      * new_indices,
        int64_t              gather_k);

// Global cache instance (one per process, like the paging pool).
helix_neuron_cache & helix_get_neuron_cache(void);

// Stats for display.
struct helix_cache_stats {
    uint64_t total_fetched;
    uint64_t total_hits;
    uint64_t total_updates;
    double   hit_rate_pct;  // total_hits / (total_hits + total_fetched) * 100
};

bool helix_neuron_cache_get_stats(helix_cache_stats * out);
