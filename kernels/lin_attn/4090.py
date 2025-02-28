import torch
from einops import rearrange
from tqdm import trange

def prepare_lens(offsets):
    """Calculate sequence lengths from offsets."""
    return offsets[1:] - offsets[:-1]

def prepare_token_indices(offsets):
    """Convert sequence offsets to token indices."""
    batch_sizes = prepare_lens(offsets)
    indices = []
    for batch_idx, size in enumerate(batch_sizes):
        for pos in range(size):
            indices.append([batch_idx, pos])
    return torch.tensor(indices, device=offsets.device)

def prepare_chunk_indices(offsets, block_size):
    """Convert sequence offsets to chunk indices."""
    batch_sizes = prepare_lens(offsets)
    indices = []
    for batch_idx, size in enumerate(batch_sizes):
        for chunk_idx in range(0, (size + block_size - 1) // block_size):
            indices.append([batch_idx, chunk_idx])
    return torch.tensor(indices, device=offsets.device)

def parallel_nsa_fwd(q, k, v, block_indices, block_size, scale, offsets=None):
    """Forward pass for Neighborhood Search Attention.
    
    Args:
        q: Query tensor of shape [B, T, HQ, K]
        k: Key tensor of shape [B, T, H, K]
        v: Value tensor of shape [B, T, H, V]
        block_indices: Block indices of shape [B, T, H, S]
        block_size: Size of each block
        scale: Scaling factor for attention scores
        offsets: Optional tensor for variable-length sequences
    
    Returns:
        o: Output tensor of shape [B, T, HQ, V]
        lse: Log-sum-exp values
    """
    B, T, HQ, K_dim = q.shape
    _, _, H, _ = k.shape
    V_dim = v.shape[-1]
    S = block_indices.shape[-1]
    G = HQ // H  # Group size (for GQA)
    BS = block_size
    
    # Handle variable-length sequences
    if offsets is not None:
        token_indices = prepare_token_indices(offsets)
        o = torch.zeros(B, T, HQ, V_dim, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)
        
        for idx in range(len(token_indices)):
            i_n, i_t = token_indices[idx]
            i_n = i_n.item() if isinstance(i_n, torch.Tensor) else i_n
            i_t = i_t.item() if isinstance(i_t, torch.Tensor) else i_t
            
            bos = offsets[i_n].item()
            eos = offsets[i_n + 1].item()
            seq_len = eos - bos
            
            for i_h in range(H):
                for g in range(G):
                    # Initialize accumulators
                    m_i = float('-inf')  # Max value tracker
                    acc_i = 0.0  # Normalization accumulator
                    o_i = torch.zeros(V_dim, dtype=torch.float32, device=q.device)
                    
                    # Get query for this position
                    q_i = q[i_n, i_t, i_h * G + g] * scale
                    
                    # Process each block
                    for i in range(S):
                        block_idx = block_indices[i_n, i_t, i_h, i].item()
                        start_idx = block_idx * BS
                        end_idx = min(start_idx + BS, seq_len)
                        
                        if start_idx >= end_idx:
                            continue
                        
                        # Get keys and values for this block
                        k_block = k[i_n, bos + start_idx:bos + end_idx, i_h]
                        v_block = v[i_n, bos + start_idx:bos + end_idx, i_h]
                        
                        # Calculate attention scores
                        s_block = torch.matmul(q_i, k_block.transpose(-2, -1))
                        
                        # Apply causal mask
                        pos_indices = torch.arange(start_idx, end_idx, device=q.device)
                        causal_mask = i_t >= pos_indices
                        s_block = torch.where(causal_mask, s_block, torch.tensor(float('-inf')))
                        
                        # Update max value
                        m_new = torch.max(s_block)
                        if m_new > m_i:
                            # Re-scale previous contributions
                            scaling_factor = torch.exp(m_i - m_new)
                            o_i *= scaling_factor
                            acc_i *= scaling_factor
                            m_i = m_new
                        
                        # Calculate probabilities
                        p_block = torch.exp(s_block - m_i)
                        
                        # Update output and accumulator
                        o_i += torch.matmul(p_block, v_block)
                        acc_i += p_block.sum()
                    
                    # Normalize output
                    if acc_i > 0:
                        o_i /= acc_i
                    
                    # Store output and lse
                    o[i_n, i_t, i_h * G + g] = o_i
                    lse[i_n, i_t, i_h * G + g] = m_i + torch.log(acc_i) if acc_i > 0 else m_i
    else:
        # Non-variable length version
        o = torch.zeros(B, T, HQ, V_dim, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, T, HQ, dtype=torch.float32, device=q.device)
        
        for i_b in range(B):
            for i_t in range(T):
                for i_h in range(H):
                    for g in range(G):
                        # Initialize accumulators
                        m_i = float('-inf')  # Max value tracker
                        acc_i = 0.0  # Normalization accumulator
                        o_i = torch.zeros(V_dim, dtype=torch.float32, device=q.device)
                        
                        # Get query for this position
                        q_i = q[i_b, i_t, i_h * G + g] * scale
                        
                        # Process each block
                        for i in range(S):
                            block_idx = block_indices[i_b, i_t, i_h, i].item()
                            start_idx = block_idx * BS
                            end_idx = min(start_idx + BS, T)
                            
                            if start_idx >= end_idx:
                                continue
                            
                            # Get keys and values for this block
                            k_block = k[i_b, start_idx:end_idx, i_h]
                            v_block = v[i_b, start_idx:end_idx, i_h]
                            
                            # Calculate attention scores
                            s_block = torch.matmul(q_i, k_block.transpose(-2, -1))
                            
                            # Apply causal mask
                            pos_indices = torch.arange(start_idx, end_idx, device=q.device)
                            causal_mask = i_t >= pos_indices
                            s_block = torch.where(causal_mask, s_block, torch.tensor(float('-inf')))
                            
                            # Update max value
                            m_new = torch.max(s_block)
                            if m_new > m_i:
                                # Re-scale previous contributions
                                scaling_factor = torch.exp(m_i - m_new)
                                o_i *= scaling_factor
                                acc_i *= scaling_factor
                                m_i = m_new
                            
                            # Calculate probabilities
                            p_block = torch.exp(s_block - m_i)
                            
                            # Update output and accumulator
                            o_i += torch.matmul(p_block, v_block)
                            acc_i += p_block.sum()
                        
                        # Normalize output
                        if acc_i > 0:
                            o_i /= acc_i
                        
                        # Store output and lse
                        o[i_b, i_t, i_h * G + g] = o_i
                        lse[i_b, i_t, i_h * G + g] = m_i + torch.log(acc_i) if acc_i > 0 else m_i
    
    return o, lse

def parallel_nsa_block_mask(block_indices, offsets, block_size):
    """Create a mask indicating which blocks are valid for each token."""
    B, T, H, S = block_indices.shape
    BS = block_size
    
    if offsets is not None:
        lens = prepare_lens(offsets)
        NS = (lens.max().item() + BS - 1) // BS
    else:
        NS = (T + BS - 1) // BS
    
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)
    
    for i_b in range(B):
        for i_t in range(T):
            for i_h in range(H):
                for i_s in range(S):
                    b_i = block_indices[i_b, i_t, i_h, i_s].item()
                    if b_i < NS and i_t >= b_i * BS:
                        block_mask[i_b, i_t, i_h, b_i] = True
    
    return block_mask

def parallel_nsa_bwd_preprocess(o, do):
    """Preprocess for backward pass to compute delta values."""
    delta = torch.sum(o * do, dim=-1)
    return delta

def parallel_nsa_bwd(q, k, v, o, lse, do, block_indices, block_size, scale, offsets=None):
    """Backward pass for Neighborhood Search Attention."""
    B, T, HQ, K_dim = q.shape
    _, _, H, _ = k.shape
    V_dim = v.shape[-1]
    S = block_indices.shape[-1]
    G = HQ // H
    BS = block_size
    
    # Compute delta (used for gradient stability)
    delta = parallel_nsa_bwd_preprocess(o, do)
    
    # Initialize gradients
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    
    # Create block mask
    block_mask = parallel_nsa_block_mask(block_indices, offsets, block_size)
    
    if offsets is not None:
        token_indices = prepare_token_indices(offsets)
        chunk_indices = prepare_chunk_indices(offsets, block_size)
        
        # Compute dq
        for idx in range(len(token_indices)):
            i_n, i_t = token_indices[idx]
            i_n = i_n.item() if isinstance(i_n, torch.Tensor) else i_n
            i_t = i_t.item() if isinstance(i_t, torch.Tensor) else i_t
            
            bos = offsets[i_n].item()
            eos = offsets[i_n + 1].item()
            seq_len = eos - bos
            
            for i_h in range(H):
                for g in range(G):
                    hq_idx = i_h * G + g
                    
                    # Get query and its gradient for this position
                    q_i = q[i_n, i_t, hq_idx] * scale
                    do_i = do[i_n, i_t, hq_idx]
                    lse_i = lse[i_n, i_t, hq_idx]
                    delta_i = delta[i_n, i_t, hq_idx]
                    
                    # Initialize gradient for this query
                    dq_i = torch.zeros_like(q_i)
                    
                    # Process each block for dq
                    for i in range(S):
                        block_idx = block_indices[i_n, i_t, i_h, i].item()
                        start_idx = block_idx * BS
                        end_idx = min(start_idx + BS, seq_len)
                        
                        if start_idx >= end_idx:
                            continue
                        
                        # Get keys and values for this block
                        k_block = k[i_n, bos + start_idx:bos + end_idx, i_h]
                        v_block = v[i_n, bos + start_idx:bos + end_idx, i_h]
                        
                        # Calculate attention scores
                        s_block = torch.matmul(q_i, k_block.transpose(-2, -1))
                        
                        # Apply causal mask
                        pos_indices = torch.arange(start_idx, end_idx, device=q.device)
                        causal_mask = i_t >= pos_indices
                        s_block = torch.where(causal_mask, s_block, torch.tensor(float('-inf')))
                        
                        # Calculate probabilities
                        p_block = torch.exp(s_block - lse_i)
                        
                        # Calculate gradient for dq
                        dp_block = torch.matmul(do_i, v_block.transpose(-2, -1))
                        ds_block = p_block * (dp_block - delta_i)
                        dq_i += torch.matmul(ds_block, k_block)
                    
                    # Scale and store dq
                    dq[i_n, i_t, hq_idx] = dq_i * scale
        
        # Compute dk and dv
        for c_idx in range(len(chunk_indices)):
            i_n, i_s = chunk_indices[c_idx]
            i_n = i_n.item() if isinstance(i_n, torch.Tensor) else i_n
            i_s = i_s.item() if isinstance(i_s, torch.Tensor) else i_s
            
            bos = offsets[i_n].item()
            eos = offsets[i_n + 1].item()
            seq_len = eos - bos
            
            start_idx = i_s * BS
            end_idx = min(start_idx + BS, seq_len)
            
            if start_idx >= end_idx:
                continue
            
            for i_h in range(H):
                # Initialize gradients for this chunk
                dk_chunk = torch.zeros((end_idx - start_idx, K_dim), dtype=k.dtype, device=k.device)
                dv_chunk = torch.zeros((end_idx - start_idx, V_dim), dtype=v.dtype, device=v.device)
                
                # Find all queries that attend to this chunk
                for i_t in range(bos, bos + seq_len):
                    local_t = i_t - bos
                    if block_mask[i_n, local_t, i_h, i_s]:
                        for g in range(G):
                            hq_idx = i_h * G + g
                            
                            # Get query and gradients
                            q_i = q[i_n, local_t, hq_idx] * scale
                            do_i = do[i_n, local_t, hq_idx]
                            lse_i = lse[i_n, local_t, hq_idx]
                            delta_i = delta[i_n, local_t, hq_idx]
                            
                            # Get chunk data
                            k_chunk = k[i_n, bos + start_idx:bos + end_idx, i_h]
                            v_chunk = v[i_n, bos + start_idx:bos + end_idx, i_h]
                            
                            # Calculate attention scores
                            s_chunk = torch.matmul(q_i, k_chunk.transpose(-2, -1))
                            
                            # Apply causal mask
                            pos_indices = torch.arange(start_idx, end_idx, device=q.device)
                            causal_mask = local_t >= pos_indices
                            s_chunk = torch.where(causal_mask, s_chunk, torch.tensor(float('-inf')))
                            
                            # Calculate probabilities
                            p_chunk = torch.exp(s_chunk - lse_i)
                            
                            # Calculate gradients for dk and dv
                            # For dk: dL/dk = dL/do * dP * q
                            dp_k = torch.outer(v_chunk, do_i)
                            ds_k = p_chunk.unsqueeze(-1) * (dp_k - delta_i)
                            dk_chunk += torch.matmul(ds_k, q_i.unsqueeze(0))
                            
                            # For dv: dL/dv = dL/do * P
                            dv_chunk += torch.outer(p_chunk, do_i)
                
                # Store gradients
                dk[i_n, bos + start_idx:bos + end_idx, i_h] = dk_chunk
                dv[i_n, bos + start_idx:bos + end_idx, i_h] = dv_chunk
    else:
        # Non-variable length version
        # Compute dq
        for i_b in range(B):
            for i_t in range(T):
                for i_h in range(H):
                    for g in range(G):
                        hq_idx = i_h * G + g
                        
                        # Get query and its gradient for this position
                        q_i = q[i_b, i_t, hq_idx] * scale
                        do_i = do[i_b, i_t, hq_idx]
                        lse_i = lse[i_b, i_t, hq_idx]
                        delta_i = delta[i_b, i_t, hq_idx]
                        
                        # Initialize gradient for this query
                        dq_i = torch.zeros_like(q_i)
                        
                        # Process each block for dq
                        for i in range(S):
                            block_idx = block_indices[i_b, i_t, i_h, i].item()
                            start_idx = block_idx * BS
                            end_idx = min(start_idx + BS, T)
                            
                            if start_idx >= end_idx:
                                continue
                            
                            # Get keys and values for this block
                            k_block = k[i_b, start_idx:end_idx, i_h]
                            v_block = v[i_b, start_idx:end_idx, i_h]
                            
                            # Calculate attention scores
                            s_block = torch.matmul(q_i, k_block.transpose(-2, -1))
                            
                            # Apply causal mask
                            pos_indices = torch.arange(start_idx, end_idx, device=q.device)
                            causal_mask = i_t >= pos_indices
                            s_block = torch.where(causal_mask, s_block, torch.tensor(float('-inf')))
                            
                            # Calculate probabilities
                            p_block = torch.exp(s_block - lse_i)
                            
                            # Calculate gradient for dq
                            dp_block = torch.matmul(do_i, v_block.transpose(-2, -1))
                            ds_block = p_block * (dp_block - delta_i)
                            dq_i += torch.matmul(ds_block, k_block)
                        
                        # Scale and store dq
                        dq[i_b, i_t, hq_idx] = dq_i * scale
        
        # Compute dk and dv by iterating through each chunk
        for i_b in range(B):
            for i_s in range((T + BS - 1) // BS):
                start_idx = i_s * BS
                end_idx = min(start_idx + BS, T)
                
                for i_h in range(H):
                    # Initialize gradients for this chunk
                    dk_chunk = torch.zeros((end_idx - start_idx, K_dim), dtype=k.dtype, device=k.device)
                    dv_chunk = torch.zeros((end_idx - start_idx, V_dim), dtype=v.dtype, device=v.device)
                    
                    # Find all queries that attend to this chunk
                    for i_t in range(T):
                        if block_mask[i_b, i_t, i_h, i_s]:
                            for g in range(G):
                                hq_idx = i_h * G + g
                                
                                # Get query and gradients
                                q_i = q[i_b, i_t, hq_idx] * scale
                                do_i = do[i_b, i_t, hq_idx]
                                lse_i = lse[i_b, i_t, hq_idx]
                                delta_i = delta[i_b, i_t, hq_idx]
                                
                                # Get chunk data
                                k_chunk = k[i_b, start_idx:end_idx, i_h]
                                v_chunk = v[i_b, start_idx:end_idx, i_h]
                                
                                # Calculate attention scores
                                s_chunk = torch.matmul(q_i, k_chunk.transpose(-2, -1))
                                
                                # Apply causal mask
                                pos_indices = torch.arange(start_idx, end_idx, device=q.device)
                                causal_mask = i_t >= pos_indices
                                s_chunk = torch.where(causal_mask, s_chunk, torch.tensor(float('-inf')))
                                
                                # Calculate probabilities
                                p_chunk = torch.exp(s_chunk - lse_i)
                                
                                # Calculate gradients for dk and dv
                                # For dk
                                dp_k = torch.outer(v_chunk, do_i)
                                ds_k = p_chunk.unsqueeze(-1) * (dp_k - delta_i)
                                dk_chunk += torch.matmul(ds_k, q_i.unsqueeze(0))
                                
                                # For dv
                                dv_chunk += torch.outer(p_chunk, do_i)
                    
                    # Store gradients
                    dk[i_b, start_idx:end_idx, i_h] = dk_chunk
                    dv[i_b, start_idx:end_idx, i_h] = dv_chunk
    
    return dq, dk, dv

class ParallelNSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, block_size, scale, offsets=None):
        """Forward pass for Neighborhood Search Attention."""
        # Make inputs contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_indices = block_indices.contiguous()
        if offsets is not None:
            offsets = offsets.contiguous()
        
        # Save context for backward pass
        ctx.save_for_backward(q, k, v, block_indices, offsets)
        ctx.block_size = block_size
        ctx.scale = scale if scale is not None else 1.0 / (k.shape[-1] ** 0.5)
        
        # Compute forward pass
        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_size=block_size,
            scale=ctx.scale,
            offsets=offsets
        )
        
        # Save output and lse for backward pass
        ctx.save_for_backward(q, k, v, o, lse, block_indices, offsets)
        
        return o

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for Neighborhood Search Attention."""
        q, k, v, o, lse, block_indices, offsets = ctx.saved_tensors
        block_size = ctx.block_size
        scale = ctx.scale
        
        # Make gradients contiguous
        grad_output = grad_output.contiguous()
        
        # Compute gradients
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=grad_output,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            offsets=offsets
        )
        
        # Return gradients for inputs (None for non-tensor inputs)
        return dq, dk, dv, None, None, None, None

def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    block_size: int,
    scale: float = None,
    cu_seqlens: torch.LongTensor = None,
    head_first: bool = False
) -> torch.Tensor:
    """Neighborhood Search Attention.
    
    Args:
        q: Query tensor of shape [B, T, HQ, K] if head_first=False else [B, HQ, T, K]
        k: Key tensor of shape [B, T, H, K] if head_first=False else [B, H, T, K]
        v: Value tensor of shape [B, T, H, V] if head_first=False else [B, H, T, V]
        indices: Block indices of shape [B, T, H, S]
        block_size: Size of each block
        scale: Scaling factor for attention scores
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        head_first: Whether inputs are in head-first format
        
    Returns:
        Output tensor of shape [B, T, HQ, V] if head_first=False else [B, HQ, T, V]
    """
    if scale is None:
        scale = 1.0 / (k.shape[-1] ** 0.5)
    
    # Handle head_first format
    if head_first:
        q = rearrange(q, 'b h t d -> b t h d')
        k = rearrange(k, 'b h t d -> b t h d')
        v = rearrange(v, 'b h t d -> b t h d')
        indices = rearrange(indices, 'b h t s -> b t h s')
    
    # Apply NSA function
    o = ParallelNSAFunction.apply(q, k, v, indices, block_size, scale, cu_seqlens)
    
    # Convert back to head_first format if needed
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    
    return o

def test_parallel_nsa():
    """Test the NSA implementation with a simple example."""
    torch.manual_seed(42)
    
    # Test parameters
    B = 2      # Batch size
    T = 16     # Sequence length
    H = 2      # Number of heads
    HQ = 4     # Number of query heads (for GQA)
    K = 32     # Key dimension
    V = 32     # Value dimension
    S = 4      # Number of blocks per query
    BS = 4     # Block size
    
    # Create inputs
    q = torch.randn(B, T, HQ, K, device="cuda")
    k = torch.randn(B, T, H, K, device="cuda")
    v = torch.randn(B, T, H, V, device="cuda")
    
    # Create block indices - each query attends to S blocks
    block_indices = torch.zeros(B, T, H, S, dtype=torch.long, device="cuda")
    for i_b in range(B):
        for i_t in range(T):
            for i_h in range(H):
                # For demonstration, select blocks based on position
                # In practice, this would be based on a meaningful selection strategy
                for i_s in range(S):
                    block_idx = max(0, min((i_t // BS) - i_s, T // BS - 1))
                    block_indices[i_b, i_t, i_h, i_s] = block_idx
    
    # Run forward pass
    output = parallel_nsa(q, k, v, block_indices, BS)
    
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {output.norm()}")
    
    # Test backward pass with random gradient
    output.retain_grad()
    loss = output.sum()
    loss.backward()
    
    print(f"q.grad norm: {q.grad.norm()}")
    print(f"k.grad norm: {k.grad.norm()}")
    print(f"v.grad norm: {v.grad.norm()}")
    
    return output

if __name__ == "__main__":
    test_parallel_nsa()


# REGULAR LINEAR ATTENTION

# import sys
# from tqdm import trange

# import torch
# from einops import rearrange

# ROWS = 16
# ATTN_D = 64
# ACTIVE_TILES = 4
# NUM_WORKERS = 8

# B = 1 # keep 1
# H = 1 # keep 1
# N = 1024
# D = ATTN_D
# scale = False

# def pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=False):
#     if scale:
#         q = q * (q.shape[-1] ** -0.5)
#     attn = q @ k.transpose(-2, -1)
#     attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
#     o = attn @ v
#     return o

# def zero(x):
#     x.zero_()

# def make_causal(x):
#     mask = torch.triu(torch.ones_like(x), diagonal=1)
#     x.masked_fill_(mask.bool(), 0)

# def make_causal_t(x):
#     mask = torch.triu(torch.ones_like(x), diagonal=1)
#     x.masked_fill_(mask.T.bool(), 0)

# def cumsum_inplace(sds_s, start_idx):
#     accum = torch.zeros_like(sds_s[0])
#     for i in range(ACTIVE_TILES + 1):
#         accum += sds_s[(start_idx + i) % (ACTIVE_TILES + 1)]
#         sds_s[(start_idx + i) % (ACTIVE_TILES + 1)] = accum.clone()

# def revcumsum_inplace(sds_s, start_idx):
#     accum = torch.zeros_like(sds_s[0])
#     for i in range(ACTIVE_TILES + 1):
#         accum += sds_s[(start_idx - i) % (ACTIVE_TILES + 1)]
#         sds_s[(start_idx - i) % (ACTIVE_TILES + 1)] = accum.clone()

# def linear_attention_bwd(q_g, k_g, v_g, d_o_g, b, h):
#     # q,k,v,d_o: (B,H,N,D)
#     B,H,N,D = q.shape

#     dq_g = torch.zeros_like(q, requires_grad=False)
#     dk_g = torch.zeros_like(k, requires_grad=False)
#     dv_g = torch.zeros_like(v, requires_grad=False)
#     dq_g.requires_grad = False
#     dk_g.requires_grad = False
#     dv_g.requires_grad = False

#     n_blocks = N // (ACTIVE_TILES * ROWS)

#     dodqqdk_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
#     k_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
#     v_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
#     sds_s = torch.zeros((ACTIVE_TILES+1, D, D), dtype=q.dtype)
#     dodv_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)

#     dq_r = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)

#     # ---- First loop: compute dq ----
#     total_block_idx = 0
#     for block in range(n_blocks):
#         # Load tiles of d_o and k,v
#         # We'll emulate tiled loading by slicing
#         # cur_idx range: block*BLOCK to block*BLOCK + BLOCK

#         for warpid in range(ACTIVE_TILES):
#             cur_idx = block*ACTIVE_TILES + warpid
#             dodqqdk_s[warpid] = d_o_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
#             k_s[warpid] = k_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
#         for warpid in range(ACTIVE_TILES, NUM_WORKERS):
#             cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES
#             v_s[warpid-ACTIVE_TILES] = v_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
        
#         for warpid in range(ACTIVE_TILES):
#             d_o_tile = dodqqdk_s[warpid]
#             v_tile = v_s[warpid]

#             local_attn = d_o_tile @ v_tile.T
#             make_causal(local_attn)

#             k_tile = k_s[warpid]

#             dq_r[warpid] = local_attn @ k_tile

#             accum = v_tile.T @ k_tile
#             sds_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)] = accum

#         cumsum_inplace(sds_s, total_block_idx)

#         for warpid in range(ACTIVE_TILES):
#             d_o_tile = dodqqdk_s[warpid]
#             s = sds_s[(total_block_idx+warpid)%(ACTIVE_TILES+1)]
#             dq_r[warpid] += d_o_tile @ s
#             dodqqdk_s[warpid] = dq_r[warpid]

#         total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1)

#         for warpidx in range(ACTIVE_TILES):
#             cur_idx = block*ACTIVE_TILES + warpidx
#             dq_tile = dodqqdk_s[warpidx]
#             dq_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:] = dq_tile

#     # ---- Second loop: compute dk, dv ----

#     dk_r = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
#     dv_r = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)

#     total_block_idx = 0
#     zero(sds_s)
#     for block in range(n_blocks-1, -1, -1):

#         for warpid in range(ACTIVE_TILES):
#             cur_idx = block*ACTIVE_TILES + warpid
#             dodqqdk_s[warpid] = q_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
#             k_s[warpid] = k_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]

#         for warpid in range(ACTIVE_TILES, NUM_WORKERS):
#             cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES
#             v_s[warpid-ACTIVE_TILES] = v_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
#             dodv_s[warpid-ACTIVE_TILES] = d_o_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]

#         for warpid in range(ACTIVE_TILES):
#             d_o_tile = dodv_s[warpid]

#             # first part of dk
#             v_tile = v_s[warpid]

#             local_attn = v_tile @ d_o_tile.T
#             make_causal_t(local_attn) # TODO : make_causal_t

#             q_tile = dodqqdk_s[warpid]
#             dk_r[warpid] = local_attn @ q_tile

#             # first part of dv
#             k_tile = k_s[warpid]

#             local_attn = k_tile @ q_tile.T
#             make_causal_t(local_attn) # TODO : make_causal_t

#             dv_r[warpid] = local_attn @ d_o_tile

#             # ds
#             accum = q_tile.T @ d_o_tile
#             sds_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)] = accum
        
#         revcumsum_inplace(sds_s, total_block_idx)

#         for warpid in range(ACTIVE_TILES):
#             # second part of dk
#             v_tile = v_s[warpid]
#             ds = sds_s[(total_block_idx+warpid+2)%(ACTIVE_TILES+1)]
#             dk_r[warpid] += v_tile @ ds.T
#             dodqqdk_s[warpid] = dk_r[warpid]

#         for warpid in range(ACTIVE_TILES):
#             # second part of dv
#             k_tile = k_s[warpid]
#             ds = sds_s[(total_block_idx+warpid+2)%(ACTIVE_TILES+1)]
#             dv_r[warpid] += k_tile @ ds
#             dodv_s[warpid] = dv_r[warpid]

#         total_block_idx = ((total_block_idx - ACTIVE_TILES) % (ACTIVE_TILES + 1) + (ACTIVE_TILES + 1)) % (ACTIVE_TILES + 1)

#         for warpidx in range(ACTIVE_TILES):
#             cur_idx = block*ACTIVE_TILES + warpidx
#             dk_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:] = dodqqdk_s[warpidx]
#             dv_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:] = dodv_s[warpidx]

#     return dq_g, dk_g, dv_g

# torch.random.manual_seed(42)
# q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
# k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
# v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
# A = torch.rand_like(v, dtype=torch.bfloat16, device='cuda').to(torch.float32)

# q.requires_grad = True
# k.requires_grad = True
# v.requires_grad = True

# o = pytorch_ref(q, k, v)
# J = (o*A).sum()
# J.backward(retain_graph=True)
# grad_q, grad_k, grad_v = q.grad, k.grad, v.grad
# grad_o = torch.autograd.grad(J, o, retain_graph=True)[0]

# dq, dk, dv = linear_attention_bwd(q, k, v, grad_o, 0, 0)

# print(torch.norm(dq))
# print(torch.norm(dk))
# print(torch.norm(dv))

# # checks
# print(torch.allclose(dq, grad_q, atol=1e-4))
# print(torch.allclose(dk, grad_k, atol=1e-4))
# print(torch.allclose(dv, grad_v, atol=1e-4))

# # save to file
# with open(f'bwd_{B}x{H}x{N}x{D}.txt', 'w') as f:
#     qf = q.to(torch.float32).flatten().cpu().detach().numpy().tolist()
#     kf = k.to(torch.float32).flatten().cpu().detach().numpy().tolist()
#     vf = v.to(torch.float32).flatten().cpu().detach().numpy().tolist()
#     dof = grad_o.to(torch.float32).flatten().cpu().numpy().tolist()
#     dqf_ref = dq.to(torch.float32).flatten().cpu().detach().numpy().tolist()
#     dkf_ref = dk.to(torch.float32).flatten().cpu().detach().numpy().tolist()
#     dvf_ref = dv.to(torch.float32).flatten().cpu().detach().numpy().tolist()

#     for i in trange(B*H*N*D):
#         f.write(repr(qf[i]))
#         f.write(' ')
#     for i in trange(B*H*N*D):
#         f.write(repr(kf[i]))
#         f.write(' ')
#     for i in trange(B*H*N*D):
#         f.write(repr(vf[i]))
#         f.write(' ')
#     for i in trange(B*H*N*D):
#         f.write(repr(dof[i]))
#         f.write(' ')
#     for i in trange(B*H*N*D):
#         f.write(repr(dqf_ref[i]))
#         f.write(' ')
#     for i in trange(B*H*N*D):
#         f.write(repr(dkf_ref[i]))
#         f.write(' ')
#     for i in trange(B*H*N*D):
#         f.write(repr(dvf_ref[i]))
#         f.write(' ')
