import torch
import random
from einops import rearrange
from typing import Optional, Tuple, Union
import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange, repeat
import math


# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 4
H = 32
N = 1024
D = 128 

Q_LEN = 10
L_NEW = 3


TESTNAME = sys.argv[1]
dtype=torch.bfloat16
if TESTNAME == 'ones':

    q = torch.ones(B, Q_LEN, H, D, device="cuda", dtype=dtype)
    k_cache = torch.ones(B, N, H, D, device="cuda", dtype=dtype)
    v_cache = torch.ones(B, N, H, D, device="cuda", dtype=dtype)
    k_seqlens = torch.randint(1, N-20, (B,), device="cuda").int()

    for i in range(B):
        k_cache[i, k_seqlens[i]:] = 0
        v_cache[i, k_seqlens[i]:] = 0

    k_new = torch.ones(B, L_NEW, H, D, device="cuda", dtype=dtype)
    v_new = torch.ones(B, L_NEW, H, D, device="cuda", dtype=dtype)

else:
    print('Invalid test name')
    sys.exit(0)


#########

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

#########

out_torch = mha_fwd_kvcache_torch(
        q, 
        k_cache, v_cache, 
        k = k_new, v = v_new, 
        seqlens_k = k_seqlens, 
        is_causal=True
    )

breakpoint()

fn = f'{TESTNAME}_{N}_{D}.txt'
with open(fn, 'w') as f:

    # inputs
    qf = q.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy() 
    k_cachef = k_cache.to(torch.float32).flatten().detach().cpu().numpy()
    v_cachef = v_cache.to(torch.float32).flatten().detach().cpu().numpy()
    k_newf = k_new.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    v_newf = v_new.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    k_seqlensf = k_seqlens.to(torch.float32).detach().cpu().numpy()
    outf = out_torch.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(qf.shape[0]):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(k_cachef.shape[0]):
        f.write(repr(k_cache[i]))
        f.write(' ')
    for i in trange(v_cachef.shape[0]):
        f.write(repr(v_cache[i]))
        f.write(' ')
    for i in trange(k_newf.shape[0]):
        f.write(repr(k_new[i]))
        f.write(' ')
    for i in trange(v_newf.shape[0]):
        f.write(repr(v_new[i]))
        f.write(' ')
    for i in trange(k_seqlensf.shape[0]):
        f.write(repr(k_seqlens[i]))
        f.write(' ')

    # output
    for i in trange(outf.shape[0]):
        f.write(repr(outf[i]))
        f.write(' ')

    print('Done!')


    

    
