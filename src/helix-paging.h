#pragma once

#include "ggml.h"

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
