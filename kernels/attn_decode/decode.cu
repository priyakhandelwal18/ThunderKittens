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
    const int q_batch = blockIdx.z, head  = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;
    int kv_batch = q_batch;
    if (cache_batch_idx) {
        kv_batch = cache_batch_idx[q_batch];
    }
    const int q_seq_next = (blockIdx.x + 1) * NUM_WORKERS;
    auto num_q_rows = SEQ_AXIS == 2 ? g.Qg.rows : g.Qg.depth;
    int k_seqlen = k_seqlens[q_batch];

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    // K and V live in shared memory. Here, we instantiate three tiles for a 3-stage pipeline.
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    // We also reuse this memory to improve coalescing of DRAM reads and writes.
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile of 16x64
    if (q_seq*ROWS<D> < num_q_rows) {
        auto q_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, q_seq, 0} : coord{q_batch, q_seq, head, 0});
        auto n_rows = (q_seq + 1) * ROWS<D> > num_q_rows ? num_q_rows - q_seq * ROWS<D> : ROWS<D>;

        // replace the rest of the tile with NEG_INF for attention
        load<shared_tile<D>, global_layout<D>, SEQ_AXIS>(qo_smem[workerid], g.Qg, q_coords, n_rows, NEG_INF);  // going through shared memory improves coalescing of dram reads.
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();
    // temperature adjustment. Pre-multiplying by lg2(e), too, so we can use exp2 later.
    if constexpr(D == 64) mul(q_reg, q_reg, __float2bfloat16(0.125f * 1.44269504089));
    else if constexpr(D == 128) mul(q_reg, q_reg, __float2bfloat16(0.08838834764f * 1.44269504089));
    // initialize flash attention L, M, and O registers.
    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_reg);
    // launch the load of the first k, v tiles
    // total number of blocks we want to load
    int kv_blocks = (k_seqlen + (LOAD_BLOCKS*ROWS<D>) - 1) / (LOAD_BLOCKS*ROWS<D>);
    
    // we want to load all the blocks from the cache, plus the new blocks
    int kv_blocks_total = kv_blocks + (k_new_seqlen + (LOAD_BLOCKS*ROWS<D>) - 1) / (LOAD_BLOCKS*ROWS<D>);

    // total number of tiles in KV cache
    int kv_tiles = (k_seqlen + ROWS<D> - 1) / ROWS<D>;
    // total number of tiles in KV cache and KNew/VNew
    int kv_tiles_total = kv_tiles + (k_new_seqlen + ROWS<D> - 1) / ROWS<D>;

    int tic = 0;

    auto k_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, loadid, 0} : coord{kv_batch, loadid, head, 0});
    auto v_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, loadid, 0} : coord{kv_batch, loadid, head, 0});

    if (kv_blocks > 0 && loadid * ROWS<D> < k_seqlen) {
        auto n_rows = loadid * ROWS<D> < k_seqlen ? ROWS<D> : k_seqlen - loadid * ROWS<D>;

        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][0], g.KCacheg, k_coords, n_rows, NEG_INF);
        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][0], g.VCacheg, v_coords, n_rows, ZERO);
    } else if (k_new_seqlen > 0 && loadid * ROWS<D> < k_new_seqlen) {
        auto next_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, loadid, 0} : coord{q_batch, loadid, head, 0});
        auto n_rows = loadid * ROWS<D> < k_new_seqlen ? ROWS<D> : k_new_seqlen - loadid * ROWS<D>;

        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][0], g.KNewg, k_coords, n_rows, NEG_INF);
        load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][0], g.VNewg, v_coords, n_rows, ZERO);
    }
    
    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks_total; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        bool load_next = true;
        bool load_next_kv_cache = true;
        if (k_new_seqlen == 0) {
            // skip if we're out of K's or we're past the causal mask
            load_next = next_load_idx < kv_tiles && (!causal || next_load_idx <= q_seq_next);
        } else {
            // skip if we're out of KNew's or we're past the causal mask for KNew
            load_next = next_load_idx < kv_tiles_total && (!causal || (kv_idx + 1 - kv_blocks) * LOAD_BLOCKS + loadid <= q_seq_next);
            load_next_kv_cache = kv_idx + 1 < kv_blocks;
        }
        if(load_next && load_next_kv_cache && next_load_idx < kv_tiles) {
            // every two workers are working together to load the next tiles, then broadcast to all workers
            // we need to load the next times for all workers, and then skip selectively in the individual worker
            int next_tic = (tic+1)%3;
            auto next_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, next_load_idx, 0} : coord{kv_batch, next_load_idx, head, 0});

            auto n_rows = (next_load_idx + 1) * ROWS<D> > k_seqlen ? k_seqlen - (next_load_idx + 1) * ROWS<D> : ROWS<D>;

            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][next_tic], g.KCacheg, next_coords, n_rows, NEG_INF);
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][next_tic], g.VCacheg, next_coords, n_rows, ZERO);
            load_async_wait<2>(); // next k, v can stay in flight.
        }
        else if (load_next && !load_next_kv_cache && next_load_idx < kv_tiles_total) {
            // load the next tiles from KNew and VNew
            int next_tic = (tic+1)%3;
            int kv_new_idx = (kv_idx - kv_blocks + 1) * LOAD_BLOCKS + loadid;

            auto next_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, kv_new_idx, 0} : coord{q_batch, kv_new_idx, head, 0});

            auto n_rows = (kv_new_idx + 1) * ROWS<D> > k_new_seqlen ? k_new_seqlen - kv_new_idx * ROWS<D> : ROWS<D>;
            
            // replace the rest of the tile with NEG_INF for attention
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(k_smem[loadid][next_tic], g.KNewg, next_coords, n_rows, NEG_INF);
            load_group::load_async<shared_tile<D>, global_layout<D>, SEQ_AXIS>(v_smem[loadid][next_tic], g.VNewg, next_coords, n_rows, ZERO);
            load_async_wait<2>(); // next k, v can stay in flight.
        }
        else load_async_wait(); // all must arrive
        __syncthreads(); // Everyone's memory must be ready for the next stage.
        // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0;
            subtile < LOAD_BLOCKS &&
            kv_idx * LOAD_BLOCKS + subtile < kv_tiles_total;
            subtile++) {

            auto kv_cache_tile_idx = kv_idx * LOAD_BLOCKS + subtile;
            auto kv_new_tile_idx = (kv_idx >= kv_blocks) ? (kv_idx - kv_blocks) * LOAD_BLOCKS + subtile : -1;

            if (causal && (
                (
                    k_new_seqlen == 0 &&
                    kv_cache_tile_idx > q_seq // we've passed the diagonal for this subtile and not using KNew
                ) ||
                (
                    k_new_seqlen > 0 &&
                    kv_new_tile_idx > -1 &&
                    kv_new_tile_idx > q_seq // we are using KNew, and we've passed the diagonal for this subtile
                ))
            ) {
                break;
            }
            if (kv_new_tile_idx == -1 &&
                kv_cache_tile_idx >= kv_tiles
            ) {
                // we are not using KNew, and we've passed the valid KV cache range 
                break;
            }

            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            zero(att_block); // zero 16x16 attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T
            if (
                causal && (
                    (
                        k_new_seqlen == 0 && 
                        kv_cache_tile_idx == q_seq
                    ) // we are not using KNew, and Q_tile == K_tile on the diagonal
                    || (
                        k_new_seqlen > 0 &&
                        kv_new_tile_idx > -1 &&
                        kv_new_tile_idx == q_seq
                    ) // we are using KNew, and Q_tile == K_new_tile, on the diagonal
                )
            ) {
                // mask out the upper triangle
                make_causal(att_block, att_block, kittens::base_types::constants<float>::neg_infty());
            }
            copy(max_vec_last,  max_vec);
            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
            exp2(att_block, att_block); // exponentiate the block in-place.
            sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
            exp2(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
            mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.
            row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
            copy(att_block_mma, att_block); // convert to bf16 for mma_AB
            load(v_reg, v_smem[subtile][tic]); // load v from shared into registers.
            mul_row(o_reg, o_reg, max_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
            mma_AB(o_reg, att_block_mma, v_reg, o_reg); // mfma onto o_reg with the local attention@V matmul.
        }
    }
    div_row(o_reg, o_reg, norm_vec);
    __syncthreads();
    if (q_seq*ROWS<D> < num_q_rows) { // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();
        auto q_out_coords = (SEQ_AXIS == 2 ? coord{q_batch, head, q_seq, 0} : coord{q_batch, q_seq, head, 0});
        auto n_rows = (q_seq + 1) * ROWS<D> > num_q_rows ? num_q_rows - q_seq * ROWS<D> : ROWS<D>;
        store<shared_tile<D>, global_layout<D>, SEQ_AXIS>(g.Og, qo_smem[workerid], q_out_coords, n_rows);

        if (k_new_seqlen > 0) {
            // TODO(danfu): this needs to be fixed for when k_seqlen % 32 != 0
            int kv_blocks_orig = (k_seqlen + ROWS<D> - 1) / ROWS<D>;
            auto kv_cache_coords = (SEQ_AXIS == 2 ? coord{kv_batch, head, kv_blocks_orig + q_seq, 0} : coord{kv_batch, kv_blocks_orig + q_seq, head, 0});

            __syncwarp();
            // in-place update KCache with KNew, replace the rest of the tile with ZERO for reading
            load<shared_tile<D>, global_layout<D>, SEQ_AXIS>(qo_smem[workerid], g.KNewg, q_out_coords, n_rows, ZERO);  // going through shared memory improves coalescing of dram reads.
            __syncwarp();
            // TODO(danfu): this needs to be fixed for when k_seqlen % 32 != 0
            store<shared_tile<D>, global_layout<D>, SEQ_AXIS>(g.KCacheg, qo_smem[workerid], kv_cache_coords, n_rows);

            __syncwarp();
            // in-place update VCache with VNew, replace the rest of the tile with ZERO for reading
            load<shared_tile<D>, global_layout<D>, SEQ_AXIS>(qo_smem[workerid], g.VNewg, q_out_coords, n_rows, ZERO);  // going through shared memory improves coalescing of dram reads.
            __syncwarp();
            // TODO(danfu): this needs to be fixed for when k_seqlen % 32 != 0
            store<shared_tile<D>, global_layout<D>, SEQ_AXIS>(g.VCacheg, qo_smem[workerid], kv_cache_coords, n_rows);
        }
    }
}

#ifdef TORCH_COMPILE

#include "common/pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

/**
 * @brief Decode attention forward pass with a KV cache and optional in-place update to the KV cache.
 * 
 * @param q The new query. (batch, num_heads, seqlen_q, head_dim)
 * @param k_cache The existing key cache. (batch_c, num_heads, seqlen_k_max, head_dim)
 * @param v_cache The existing value cache. (batch_c, num_heads, seqlen_v, head_dim)
 * @param k_new_ The new keys to update k_cache in-place. (batch, num_heads, seqlen_q, head_dim)
 * @param v_new_ The new values to update v_cache in-place. (batch, num_heads, seqlen_q, head_dim)
 * @param causal Whether to use causal attention. If k_new_ and v_new_ are provided, causal mask is only applied against the new queries and keys. If they are not provided, k_seqlen must match seqlen_q, and the causal mask is applied against all the queries and keys.
 * @param k_seqlens (batch). The sequence lengths of the key cache.
 * @param cache_batch_idx (batch). The indices of the k_cache and v_cache to index into. Rows may be non-contiguous and repeated. If repeated, the in-place update is undefined.
 * @param blhd_format If true, use the (batch, seq, head, dim) format. Otherwise, use (batch, head, seq, dim).
 * @return The output of the attention operation. (batch, num_heads, seqlen_q, head_dim)
 */
torch::Tensor
attention_decode_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    c10::optional<torch::Tensor> k_new_,
    c10::optional<torch::Tensor> v_new_,
    bool causal,
    torch::Tensor k_seqlens,
    c10::optional<torch::Tensor> cache_batch_idx,
    bool blhd_format
)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(k_seqlens);

    int BATCH_AXIS = 0;
    int SEQ_AXIS   = blhd_format ? 1 : 2;
    int HEAD_AXIS  = blhd_format ? 2 : 1;
    int DIM_AXIS   = 3;

    auto batch_q   = q.size(BATCH_AXIS);
    auto batch_c   = k_cache.size(BATCH_AXIS);
    auto head_dim  = q.size(DIM_AXIS); 
    auto q_seq_len = q.size(SEQ_AXIS); 
    auto k_max_len = k_cache.size(SEQ_AXIS); 
    auto qo_heads  = q.size(HEAD_AXIS);
    auto kv_heads  = k_cache.size(HEAD_AXIS);

    TORCH_CHECK(k_seqlens.size(BATCH_AXIS) == batch_q, "K sequence lengths batch dimension - idx 0 - must match Q batch size");

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(BATCH_AXIS) == batch_q, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k_cache.size(BATCH_AXIS) == batch_c, "K cache batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v_cache.size(BATCH_AXIS) == batch_c, "V cache batch dimension - idx 0 - must match for all inputs");

    // TORCH_CHECK(q_seq_len % 32 == 0, "Q sequence length must be divisible by 32");
    // TORCH_CHECK(k_max_len % 32 == 0, "K cache sequence length must be divisible by 32");

    TORCH_CHECK(v_cache.size(SEQ_AXIS) == k_max_len, "V cache sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(DIM_AXIS) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k_cache.size(DIM_AXIS) == head_dim, "K cache head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v_cache.size(DIM_AXIS) == head_dim, "V cache head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(HEAD_AXIS) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k_cache.size(HEAD_AXIS) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v_cache.size(HEAD_AXIS) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");

    torch::Tensor k_new, v_new;
    auto k_new_seqlen = k_new_.has_value() ? k_new_.value().size(SEQ_AXIS) : 0;
    if (k_new_.has_value()) {
        assert(v_new_.has_value());
        k_new = k_new_.value();
        v_new = v_new_.value();
        CHECK_INPUT(k_new);
        CHECK_INPUT(v_new);
        TORCH_CHECK(k_new.size(BATCH_AXIS) == batch_q, "K new batch dimension - idx 0 - must match for Q batch size");
        TORCH_CHECK(v_new.size(BATCH_AXIS) == batch_q, "V new batch dimension - idx 0 - must match for Q batch size");
        TORCH_CHECK(k_new.size(HEAD_AXIS) == kv_heads, "K new heads - idx 1 - must match for all inputs");
        TORCH_CHECK(v_new.size(HEAD_AXIS) == kv_heads, "V new heads - idx 1 - must match for all inputs");
        TORCH_CHECK(k_new.size(SEQ_AXIS) == q_seq_len, "K new sequence length - idx 2 - must match for all inputs");
        TORCH_CHECK(v_new.size(SEQ_AXIS) == q_seq_len, "V new sequence length - idx 2 - must match for all inputs");
        TORCH_CHECK(k_new.size(DIM_AXIS) == head_dim, "K new head dimension - idx 3 - must match for all inputs");
        TORCH_CHECK(v_new.size(DIM_AXIS) == head_dim, "V new head dimension - idx 3 - must match for all inputs");
    }
    if (cache_batch_idx.has_value()) {
        CHECK_INPUT(cache_batch_idx.value());
        TORCH_CHECK(cache_batch_idx.value().size(BATCH_AXIS) == batch_q, "Cache batch indices batch dimension - idx 0 - must match Q batch size");
    }
    
    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_cache_ptr = k_cache.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_cache_ptr = v_cache.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_new_ptr = k_new_.has_value() ? k_new.data_ptr<c10::BFloat16>() : nullptr;
    c10::BFloat16* v_new_ptr = v_new_.has_value() ? v_new.data_ptr<c10::BFloat16>() : nullptr;
    int* k_seqlens_ptr = k_seqlens.data_ptr<int>();
    int* cache_batch_idx_ptr = cache_batch_idx.has_value() ? cache_batch_idx.value().data_ptr<int>() : nullptr;
    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k_cache = reinterpret_cast<bf16*>(k_cache_ptr);
    bf16*  d_v_cache = reinterpret_cast<bf16*>(v_cache_ptr);
    bf16*  d_k_new = k_new_ptr ? reinterpret_cast<bf16*>(k_new_ptr) : nullptr;
    bf16*  d_v_new = v_new_ptr ? reinterpret_cast<bf16*>(v_new_ptr) : nullptr;
    
    // for the returned outputs
    torch::Tensor o     = blhd_format ? torch::empty({static_cast<const uint>(batch_q), 
                                        static_cast<const uint>(q_seq_len), 
                                        static_cast<const uint>(qo_heads), 
                                        static_cast<const uint>(head_dim)}, q.options())
                                      : torch::empty({static_cast<const uint>(batch_q), 
                                        static_cast<const uint>(qo_heads), 
                                        static_cast<const uint>(q_seq_len), 
                                        static_cast<const uint>(head_dim)}, q.options());
    
    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    cudaDeviceSynchronize();

    unsigned long mem_size = (kittens::MAX_SHARED_MEMORY-1000) / 2; // have the flag tell us

    BOOL_SWITCH(blhd_format, IS_BLHD, [&] {
        const int SEQ_AXIS_CONST = IS_BLHD ? 1 : 2;
        if (head_dim == 64) {
            auto q_dim_1 = IS_BLHD ? q_seq_len : qo_heads;
            auto q_dim_2 = IS_BLHD ? qo_heads : q_seq_len;
            auto kv_dim_1 = IS_BLHD ? k_max_len : kv_heads;
            auto kv_dim_2 = IS_BLHD ? kv_heads : k_max_len;

            global_layout<64> qg = global_layout<64>(d_q, batch_q, q_dim_1, q_dim_2, nullptr);
            global_layout<64> kg = global_layout<64>(d_k_cache, batch_c, kv_dim_1, kv_dim_2, nullptr);
            global_layout<64> vg = global_layout<64>(d_v_cache, batch_c, kv_dim_1, kv_dim_2, nullptr);
            global_layout<64> og = global_layout<64>(d_o, batch_q, q_dim_1, q_dim_2, nullptr);
            global_layout<64> kg_new = global_layout<64>(d_k_new, batch_q, q_dim_1, q_dim_2, nullptr);
            global_layout<64> vg_new = global_layout<64>(d_v_new, batch_q, q_dim_1, q_dim_2, nullptr);

            globals<64> g(qg, kg, vg, og, kg_new, vg_new);

            cudaFuncSetAttribute(
                attend_ker<64, SEQ_AXIS_CONST>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            dim3 grid((q_seq_len + qkvo_tile<64>::rows*NUM_WORKERS - 1) / (qkvo_tile<64>::rows*NUM_WORKERS), qo_heads, batch_q);
            attend_ker<64, SEQ_AXIS_CONST><<<grid, (32*NUM_WORKERS), mem_size>>>(
                g,
                k_seqlens_ptr,
                k_new_seqlen,
                causal,
                cache_batch_idx_ptr
            );
        }
        else if (head_dim == 128) {

            auto q_dim_1 = IS_BLHD ? q_seq_len : qo_heads;
            auto q_dim_2 = IS_BLHD ? qo_heads : q_seq_len;
            auto kv_dim_1 = IS_BLHD ? k_max_len : kv_heads;
            auto kv_dim_2 = IS_BLHD ? kv_heads : k_max_len;

            global_layout<128> qg = global_layout<128>(d_q, batch_q, q_dim_1, q_dim_2, nullptr);
            global_layout<128> kg = global_layout<128>(d_k_cache, batch_c, kv_dim_1, kv_dim_2, nullptr);
            global_layout<128> vg = global_layout<128>(d_v_cache, batch_c, kv_dim_1, kv_dim_2, nullptr);
            global_layout<128> og = global_layout<128>(d_o, batch_q, q_dim_1, q_dim_2, nullptr);
            global_layout<128> kg_new = global_layout<128>(d_k_new, batch_q, q_dim_1, q_dim_2, nullptr);
            global_layout<128> vg_new = global_layout<128>(d_v_new, batch_q, q_dim_1, q_dim_2, nullptr);

            globals<128> g(qg, kg, vg, og, kg_new, vg_new);

            cudaFuncSetAttribute(
                attend_ker<128, SEQ_AXIS_CONST>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            dim3 grid((q_seq_len + qkvo_tile<128>::rows*NUM_WORKERS - 1) / (qkvo_tile<128>::rows*NUM_WORKERS), qo_heads, batch_q);
            attend_ker<128, SEQ_AXIS_CONST><<<grid, (32*NUM_WORKERS), mem_size>>>(
                g,
                k_seqlens_ptr,
                k_new_seqlen,
                causal,
                cache_batch_idx_ptr
            );
        }
        else {
            TORCH_CHECK(false, "head_dim must be 64 or 128");
        }
    });

    CHECK_CUDA_ERROR(cudaGetLastError());

    return o;
    cudaDeviceSynchronize();
}

#endif