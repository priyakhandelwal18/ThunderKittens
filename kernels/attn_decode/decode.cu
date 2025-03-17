#define TORCH_COMPILE
#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WORKERS = 4; // This kernel uses 4 worker warps per block, and 2 blocks per SM.
template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, H, g.Qg.rows specified at runtime, D=64 known at compile time for this kernel
template<int D> struct globals { global_layout<D> Qg, KCacheg, VCacheg, Og, KNewg, VNewg; };

template<int D, int SEQ_AXIS> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(
    const __grid_constant__ globals<D> g,   // Q, KCache, VCache, O, KNew, VNew
    int* k_seqlens,                         // KCache sequence length
    int k_new_seqlen,                       // KNew sequence length
    bool causal,                            // causal attention flag
    int* cache_batch_idx                    // cache batch indices
) {
    auto ZERO = kittens::base_types::constants<bf16>::zero();
    auto NEG_INF = kittens::base_types::constants<bf16>::neg_infty();
    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int q_batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;
    int kv_batch = q_batch;
    if (cache_batch_idx) {
        kv_batch = cache_batch_idx[q_batch];
    }
    const int q_seq_next = (blockIdx.x + 1) * NUM_WORKERS;
    auto num_q_rows = SEQ_AXIS == 2 ? g.Qg.rows : g.Qg.depth;
    int k_seqlen = k_seqlens[q_batch];

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    
    qkvo_tile<D, bf16> q_reg, k_reg;
    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile<D, float> o_reg;
    attn_tile<D, float> att_block;
    attn_tile<D, bf16> att_block_mma;
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec;

    // Load Q tile with proper boundary handling
    if (q_seq*ROWS<D> < num_q_rows) {
        auto q_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, q_seq, 0} : coord{q_batch, q_seq, head, 0});
        auto n_rows = min(ROWS<D>, num_q_rows - q_seq * ROWS<D>);
        load<shared_tile<D>, global_layout<D>, SEQ_AXIS>(qo_smem[workerid], g.Qg, q_coords, n_rows, ZERO);
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    } else {
        zero(q_reg);
    }
    __syncthreads();
    
    // Temperature adjustment
    if constexpr(D == 64) mul(q_reg, q_reg, __float2bfloat16(0.125f * 1.44269504089));
    else if constexpr(D == 128) mul(q_reg, q_reg, __float2bfloat16(0.08838834764f * 1.44269504089));
    
    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);

    // Calculate total blocks needed without padding assumptions
    int kv_blocks = (k_seqlen + (LOAD_BLOCKS*ROWS<D>) - 1) / (LOAD_BLOCKS*ROWS<D>);
    int kv_blocks_new = (k_new_seqlen + (LOAD_BLOCKS*ROWS<D>) - 1) / (LOAD_BLOCKS*ROWS<D>);
    int kv_blocks_total = kv_blocks + kv_blocks_new;

    int tic = 0;

    // Load first K,V tiles
    if (kv_blocks > 0 && loadid * ROWS<D> < k_seqlen) {
        auto k_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, loadid, 0} : coord{kv_batch, loadid, head, 0});
        auto v_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, loadid, 0} : coord{kv_batch, loadid, head, 0});
        auto n_rows = min(ROWS<D>, k_seqlen - loadid * ROWS<D>);
        
        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][0], g.KCacheg, k_coords, n_rows, ZERO);
        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][0], g.VCacheg, v_coords, n_rows, ZERO);
    } else if (k_new_seqlen > 0 && loadid * ROWS<D> < k_new_seqlen) {
        auto k_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, loadid, 0} : coord{q_batch, loadid, head, 0});
        auto v_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, loadid, 0} : coord{q_batch, loadid, head, 0});
        auto n_rows = min(ROWS<D>, k_new_seqlen - loadid * ROWS<D>);
        
        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][0], g.KNewg, k_coords, n_rows, ZERO);
        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][0], g.VNewg, v_coords, n_rows, ZERO);
    }

    // Main attention loop
    for(auto kv_idx = 0; kv_idx < kv_blocks_total; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        bool load_next = true;
        bool load_next_kv_cache = true;
        
        if (k_new_seqlen == 0) {
            load_next = next_load_idx * ROWS<D> < k_seqlen && (!causal || next_load_idx <= q_seq_next);
        } else {
            int effective_idx = next_load_idx - kv_blocks * LOAD_BLOCKS;
            load_next = effective_idx * ROWS<D> < k_new_seqlen && (!causal || effective_idx <= q_seq_next);
            load_next_kv_cache = next_load_idx * ROWS<D> < k_seqlen;
        }

        // Load next tiles
        if (load_next && load_next_kv_cache) {
            int next_tic = (tic+1)%3;
            auto next_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, next_load_idx, 0} : coord{kv_batch, next_load_idx, head, 0});
            auto n_rows = min(ROWS<D>, k_seqlen - next_load_idx * ROWS<D>);
            
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][next_tic], g.KCacheg, next_coords, n_rows, ZERO);
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][next_tic], g.VCacheg, next_coords, n_rows, ZERO);
            load_async_wait<2>();
        } else if (load_next && !load_next_kv_cache) {
            int next_tic = (tic+1)%3;
            int kv_new_idx = (kv_idx - kv_blocks + 1) * LOAD_BLOCKS + loadid;
            auto next_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, kv_new_idx, 0} : coord{q_batch, kv_new_idx, head, 0});
            auto n_rows = min(ROWS<D>, k_new_seqlen - kv_new_idx * ROWS<D>);
            
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][next_tic], g.KNewg, next_coords, n_rows, ZERO);
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][next_tic], g.VNewg, next_coords, n_rows, ZERO);
            load_async_wait<2>();
        } else {
            load_async_wait();
        }
        
        __syncthreads();

        // Process tiles
        for(int subtile = 0; subtile < LOAD_BLOCKS; subtile++) {
            int kv_cache_tile_idx = kv_idx * LOAD_BLOCKS + subtile;
            int kv_new_tile_idx = (kv_idx >= kv_blocks) ? (kv_idx - kv_blocks) * LOAD_BLOCKS + subtile : -1;
            
            bool process_tile = true;
            if (causal) {
                if (k_new_seqlen == 0 && kv_cache_tile_idx > q_seq) process_tile = false;
                if (k_new_seqlen > 0 && kv_new_tile_idx > -1 && kv_new_tile_idx > q_seq) process_tile = false;
            }
            if (kv_new_tile_idx == -1 && kv_cache_tile_idx * ROWS<D> >= k_seqlen) process_tile = false;
            if (kv_new_tile_idx > -1 && kv_new_tile_idx * ROWS<D> >= k_new_seqlen) process_tile = false;
            
            if (!process_tile) continue;

            load(k_reg, k_smem[subtile][tic]);
            zero(att_block);
            mma_ABt(att_block, q_reg, k_reg, att_block);

            if (causal && ((k_new_seqlen == 0 && kv_cache_tile_idx == q_seq) || 
                          (k_new_seqlen > 0 && kv_new_tile_idx > -1 && kv_new_tile_idx == q_seq))) {
                make_causal(att_block, att_block, kittens::base_types::constants<float>::neg_infty());
            }

            copy(max_vec_last, max_vec);
            row_max(max_vec, att_block, max_vec);
            sub_row(att_block, att_block, max_vec);
            exp2(att_block, att_block);
            sub(max_vec_last, max_vec_last, max_vec);
            exp2(max_vec_last, max_vec_last);
            mul(norm_vec, norm_vec, max_vec_last);
            row_sum(norm_vec, att_block, norm_vec);
            
            copy(att_block_mma, att_block);
            load(v_reg, v_smem[subtile][tic]);
            mul_row(o_reg, o_reg, max_vec_last);
            mma_AB(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    div_row(o_reg, o_reg, norm_vec);
    __syncthreads();

    // Write output and update KV cache if needed
    if (q_seq*ROWS<D> < num_q_rows) {
        store(qo_smem[workerid], o_reg);
        __syncwarp();
        
        auto q_out_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, q_seq, 0} : coord{q_batch, q_seq, head, 0});
        auto n_rows = min(ROWS<D>, num_q_rows - q_seq * ROWS<D>);
        store<shared_tile<D>, global_layout<D>, SEQ_AXIS>(g.Og, qo_smem[workerid], q_out_coords, n_rows);

        if (k_new_seqlen > 0) {
            int kv_update_idx = (k_seqlen + ROWS<D> - 1) / ROWS<D> + q_seq;
            auto kv_cache_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, kv_update_idx, 0} : coord{kv_batch, kv_update_idx, head, 0});
            
            __syncwarp();
            load<shared_tile<D>, global_layout<D>, SEQ_AXIS>(qo_smem[workerid], g.KNewg, q_out_coords, n_rows, ZERO);
            __syncwarp();
            store<shared_tile<D>, global_layout<D>, SEQ_AXIS>(g.KCacheg, qo_smem[workerid], kv_cache_coords, n_rows);
            
            __syncwarp();
            load<shared_tile<D>, global_layout<D>, SEQ_AXIS>(qo_smem[workerid], g.VNewg, q_out_coords, n_rows, ZERO);
            __syncwarp();
            store<shared_tile<D>, global_layout<D>, SEQ_AXIS>(g.VCacheg, qo_smem[workerid], kv_cache_coords, n_rows);
        }
    }
}