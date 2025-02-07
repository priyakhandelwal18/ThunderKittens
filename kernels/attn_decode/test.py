import torch
import thunderkittens as tk
import random
from tqdm import tqdm
from flash_attn_2_cuda import fwd_kvcache
from flash_attn import flash_attn_with_kvcache
from einops import rearrange
from typing import Optional, Tuple, Union

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def mha_kvcache_tmp(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    seqlens_k: Optional[torch.Tensor] = None,
    is_causal: bool = False,
):
    return flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=seqlens_k, causal=is_causal)


# the reference FlashAttention2 implementation, FA 2.6.3
def mha_fwd_kvcache_fa2(
    q: torch.Tensor,         # batch_size x seqlen_q x num_heads x head_size
    k_cache: torch.Tensor,   # batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    v_cache: torch.Tensor,   # batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    k: Optional[torch.Tensor] = None, # batch_size x seqlen_knew x num_heads_k x head_size
    v: Optional[torch.Tensor] = None, # batch_size x seqlen_knew x num_heads_k x head_size
    seqlens_k: Optional[torch.Tensor] = None, # batch_size
    rotary_cos: Optional[torch.Tensor] = None, # seqlen_ro x (rotary_dim / 2)
    rotary_sin: Optional[torch.Tensor] = None, # seqlen_ro x (rotary_dim / 2)     
    cache_batch_idx: Optional[torch.Tensor] = None, # indices to index into the KV cache
    leftpad_k: Optional[torch.Tensor] = None, # batch_size
    block_table: Optional[torch.Tensor] = None, # batch_size x max_num_blocks_per_seq
    alibi_slopes: Optional[torch.Tensor] = None, # num_heads or batch_size x num_heads
    out: Optional[torch.Tensor] = None, # batch_size x seqlen_q x num_heads x head_size
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    is_rotary_interleaved: bool = True,
    num_splits: int = 0,
) -> torch.Tensor:
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        leftpad_k: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if seqlens_k is not None and isinstance(seqlens_k, int):
        seqlens_k = torch.full(
            (k_cache.shape[0],), seqlens_k, dtype=torch.int32, device=k_cache.device
        )
        seqlens_k = maybe_contiguous(seqlens_k)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = fwd_kvcache(
        q, 
        k_cache, 
        v_cache, 
        k, 
        v, 
        seqlens_k, 
        rotary_cos, 
        rotary_sin, 
        cache_batch_idx, 
        leftpad_k, 
        block_table, 
        alibi_slopes, 
        out, 
        softmax_scale, 
        is_causal, 
        window_size_left, 
        window_size_right, 
        softcap, 
        is_rotary_interleaved, 
        num_splits
    )
    return out


# the reference PyTorch implementation, written out by hand
def mha_fwd_kvcache_torch(
    q: torch.Tensor,         # batch_size x seqlen_q x num_heads x head_size
    k_cache: torch.Tensor,   # batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    v_cache: torch.Tensor,   # batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    k: Optional[torch.Tensor] = None, # batch_size x seqlen_knew x num_heads_k x head_size
    v: Optional[torch.Tensor] = None, # batch_size x seqlen_knew x num_heads_k x head_size
    seqlens_k: Optional[torch.Tensor] = None, # batch_size
    rotary_cos: Optional[torch.Tensor] = None, # seqlen_ro x (rotary_dim / 2)
    rotary_sin: Optional[torch.Tensor] = None, # seqlen_ro x (rotary_dim / 2)     
    cache_batch_idx: Optional[torch.Tensor] = None, # indices to index into the KV cache
    leftpad_k: Optional[torch.Tensor] = None, # batch_size
    block_table: Optional[torch.Tensor] = None, # batch_size x max_num_blocks_per_seq
    alibi_slopes: Optional[torch.Tensor] = None, # num_heads or batch_size x num_heads
    out: Optional[torch.Tensor] = None, # batch_size x seqlen_q x num_heads x head_size
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    is_rotary_interleaved: bool = True,
    num_splits: int = 0,
) -> torch.Tensor:
    assert alibi_slopes is None, "alibi_slopes not supported"
    if out is None:
        out = torch.empty_like(q)
    
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if seqlens_k is not None and isinstance(seqlens_k, int):
        seqlens_k = torch.full(
            (k_cache.shape[0],), seqlens_k, dtype=torch.int32, device=k_cache.device
        )

    seqlen_knew = k.shape[1] if k is not None else 0

    if block_table is not None:
        assert False, "paged KV cache not supported"
        # TODO: paged KV cache
        pass
    else:
        if cache_batch_idx is None:
            cache_batch_idx = torch.arange(k_cache.shape[0], device=k_cache.device).long()

        # concatenate k and v with k_cache and v_cache using cache_batch_idx
        if k is not None:
            assert v is not None
            assert seqlens_k is not None
            
            if leftpad_k is not None:
                assert False, "left pad k not supported"
                # Create offset indices for the cache
                offset = leftpad_k + seqlens_k  # [batch_size]
                # Create indices for the new sequence length
                seq_idx = torch.arange(k.shape[1], device=k.device)  # [seqlen_knew]
                # Broadcast to create all indices: [batch_size, seqlen_knew]
                indices = (offset.unsqueeze(1) + seq_idx.unsqueeze(0)).long()
                # Update cache using advanced indexing
                k_cache[cache_batch_idx[:, None], indices] = k
                v_cache[cache_batch_idx[:, None], indices] = v
            else:
                # Similar to above but without leftpad_k
                offset = seqlens_k  # [batch_size]
                seq_idx = torch.arange(k.shape[1], device=k.device)  # [seqlen_knew]
                indices = (offset.unsqueeze(1) + seq_idx.unsqueeze(0)).long()  # [batch_size, seqlen_knew]
                k_cache[cache_batch_idx[:, None], indices] = k
                v_cache[cache_batch_idx[:, None], indices] = v
        else:
            offset = seqlens_k

    # apply rotary embedding if rotary_cos and rotary_sin are passed in
    if rotary_cos is not None and rotary_sin is not None:
        assert False, "rotary embedding not supported"
        if is_rotary_interleaved:
            pass
        else:
            pass

    if block_table is not None:
        pass
    else:

        # For each batch item, we need to only attend up to its sequence length
        seqlen_total = seqlens_k + seqlen_knew if k is not None else seqlens_k
        # Create a mask for valid positions based on sequence lengths
        batch_size = q.shape[0]
        valid_mask = torch.arange(k_cache[cache_batch_idx].shape[1], device=q.device)[None, :] < seqlen_total[:, None]  # [batch_size, seqlen_k]
        valid_mask = valid_mask.view(batch_size, 1, 1, -1)  # [batch_size, 1, 1, seqlen_k]

        # compute attention scores
        att = (
            # b, h, l_q, d @ b, h, d, l_k -> b, h, l_q, l_k
            q.transpose(1, 2) @ k_cache[cache_batch_idx].permute(0, 2, 3, 1)
        ) * softmax_scale

        if softcap > 0:
            att = att / softcap
            att = att.tanh()
            att = att * softcap

        # Mask out padding tokens
        if seqlens_k is not None:
            att = att.masked_fill(~valid_mask, float('-inf'))

        # apply causal mask if needed
        if is_causal:
            q_idx = torch.arange(q.size(1), device=q.device)
            k_idx = torch.arange(k_cache.shape[1], device=q.device)
            mask = k_idx[None, None, :] <= (q_idx[None, :, None] + (seqlen_total.view(-1, 1, 1) - q.size(1)))  # [batch_size, seqlen_q, seqlen_k]
            mask = mask.view(batch_size, 1, q.size(1), k_cache.shape[1])
            att = att.masked_fill(~mask, float('-inf'))

        # apply sliding window if specified
        if window_size_left != -1 or window_size_right != -1:
            assert False, "sliding window not supported"
            # This is Claude, NOT TESTED!
            q_idx = torch.arange(q.size(1), device=q.device)
            k_idx = torch.arange(seqlen_total, device=q.device)
            
            # Calculate relative positions
            relative_pos = k_idx[None, :] - (q_idx[:, None] + (seqlen_total - q.size(1)))
            
            # Create window mask
            window_mask = (relative_pos >= -window_size_left) & (relative_pos <= window_size_right)
            window_mask = window_mask.view(1, 1, q.size(1), seqlen_total)
            att = att.masked_fill(~window_mask, float('-inf'))

        # apply softmax
        att = torch.softmax(att, dim=-1)

        # compute output
        out = (
            # b, h, l_q, l_k @ b, h, l_k, d -> b, h, l_q, d
            att @ v_cache[cache_batch_idx].permute(0, 2, 1, 3)
        ).transpose(1, 2).contiguous()
    
    return out


def mha_fwd_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1) ** 0.5)
    if causal:
        # Create position indices for queries and keys
        q_idx = torch.arange(q.size(-2), device=q.device)
        k_idx = torch.arange(k.size(-2), device=k.device)
        # A query at position i can attend to keys up to position i + (k_len - q_len)
        # This works for both prefill (k_len == q_len) and decode (k_len > q_len)
        mask = k_idx[None, :] > (q_idx[:, None] + (k.size(-2) - q.size(-2)))
        QK.masked_fill_(mask, float('-inf'))
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)

    return output


if __name__ == "__main__":
    B = 4
    H = 32
    d = 128
    L_max = 1024
    Q_seqlen = 10
    L_new = 3
    dtype = torch.bfloat16
    is_causal = True

    q = torch.randn(B, Q_seqlen, H, d, device="cuda", dtype=dtype)
    k_cache = torch.randn(B, L_max, H, d, device="cuda", dtype=dtype)
    v_cache = torch.randn(B, L_max, H, d, device="cuda", dtype=dtype)

    k_seqlens = torch.randint(1, L_max-20, (B,), device="cuda").int()

    for i in range(B):
        k_cache[i, k_seqlens[i]:] = 0
        v_cache[i, k_seqlens[i]:] = 0

    k_new = torch.randn(B, L_new, H, d, device="cuda", dtype=dtype)
    v_new = torch.randn(B, L_new, H, d, device="cuda", dtype=dtype)

    print("Comparing fa2 and torch:")
    out_fa2 = mha_kvcache_tmp(
        q, k_cache, v_cache, 
        k = k_new, v = v_new, 
        seqlens_k = k_seqlens, 
        is_causal=is_causal
    )
    out_torch = mha_fwd_kvcache_torch(
        q, k_cache, v_cache, 
        k = k_new, v = v_new, 
        seqlens_k = k_seqlens, 
        is_causal=is_causal
    )
    out_fa2_redo = mha_fwd_kvcache_fa2(
        q, k_cache, v_cache, 
        k_new, v_new, k_seqlens,
        is_causal=is_causal
    )
    breakpoint()

    print((out_fa2 - out_torch).abs().max())
    print((out_fa2_redo - out_torch).abs().max())
    print()
