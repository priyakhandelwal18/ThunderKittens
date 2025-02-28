import sys
from tqdm import trange

import torch
from einops import rearrange, repeat

def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    block_size: int = 64,
    scale: bool = False
) -> torch.Tensor:
    """
    Naive implementation of neighborhood search attention (NSA).
    
    Args:
        q: Query tensor of shape [B, H, N, D]
        k: Key tensor of shape [B, H, N, D]
        v: Value tensor of shape [B, H, N, D]
        indices: Block indices of shape [B, H, N, S] where S is number of selected blocks per query
        block_size: Size of each block
        scale: Whether to scale query by 1/sqrt(D)
        
    Returns:
        Output tensor of shape [B, H, N, D]
    """
    B, H, N, D = q.shape
    S = indices.shape[-1]  # Number of selected blocks per query
    
    # Apply scaling if requested
    if scale:
        q = q * (D ** -0.5)
    
    # Initialize output tensor
    o = torch.zeros_like(v)
    
    # Rearrange tensors for processing
    q = rearrange(q, 'b h n d -> b n h d')
    k = rearrange(k, 'b h n d -> b n h d')
    v = rearrange(v, 'b h n d -> b n h d')
    indices = rearrange(indices, 'b h n s -> b n h s')
    
    # Process each batch
    for b in range(B):
        q_b, k_b, v_b, indices_b = q[b], k[b], v[b], indices[b]
        
        # Expand indices to get token-level indices
        # [N, H, S, block_size]
        token_indices = indices_b.unsqueeze(-1) * block_size + torch.arange(block_size, device=indices_b.device)
        # Reshape to [N, H, S*block_size]
        token_indices = token_indices.view(N, H, -1)
        
        # Process each query position
        for i_q in range(N):
            q_i = q_b[i_q]  # [H, D]
            indices_i = token_indices[i_q]  # [H, S*block_size]
            
            # Gather keys and values based on indices
            for h in range(H):
                idx = indices_i[h].clamp(0, N-1)  # Ensure indices are in bounds
                k_i = k_b.index_select(0, idx)  # [S*block_size, H, D]
                v_i = v_b.index_select(0, idx)  # [S*block_size, H, D]
                
                # Create attention mask to prevent attending to future tokens
                mask = idx > i_q
                
                # Compute attention scores
                attn = torch.einsum('d, n h d -> n h', q_i[h], k_i)  # [S*block_size, H]
                attn.masked_fill_(mask.unsqueeze(1), float('-inf'))
                attn = torch.softmax(attn, dim=0)
                
                # Compute weighted sum of values
                o_i = torch.einsum('n h, n h d -> h d', attn, v_i)
                o[b, h, i_q] = o_i[h]
    
    return o

# Main test generation code
B = 1  # batch size (keep 1)
H = 1  # number of heads (keep 1)
N = 1024  # sequence length
D = 64  # embedding dimension
S = 16  # number of selected blocks per query
block_size = 64  # block size
scale = False  # whether to scale queries

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn_all'

# Generate test data based on test name
if TESTNAME in ['ones_all', 'ones_t0', 'ones_t1', 'ones_t0t1', 'ones_t2']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    v = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    # Generate indices - for 'ones' test, use sequential blocks
    indices = torch.zeros((B, H, N, S), dtype=torch.long, device='cuda')
    for i in range(N):
        for s in range(S):
            block_idx = max(0, min(N//block_size - 1, i//block_size - S//2 + s))
            indices[:, :, i, s] = block_idx
elif TESTNAME in ['randn_all', 'randn_t0', 'randn_t1', 'randn_t0t1', 'randn_t2']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    # Generate random indices for the 'randn' test
    # Select S blocks per query with neighboring blocks preferred
    indices = torch.zeros((B, H, N, S), dtype=torch.long, device='cuda')
    num_blocks = N // block_size
    for i in range(N):
        current_block = i // block_size
        # Generate a distribution centered around the current block
        probs = torch.exp(-0.5 * torch.square(torch.arange(num_blocks, device='cuda') - current_block) / (num_blocks/10)**2)
        # Sample S blocks based on this distribution
        sampled_blocks = torch.multinomial(probs, S, replacement=False)
        indices[:, :, i, :] = sampled_blocks
else:
    print('Invalid test name')
    sys.exit(0)

# Run the NSA forward pass
o = naive_nsa(q, k, v, indices, block_size, scale)

# Save inputs and outputs to a file
with open(f'nsa_fwd_{B}x{H}x{N}x{D}_S{S}_B{block_size}.txt', 'w') as f:
    # Flatten and convert tensors to Python lists
    qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
    kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
    vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
    indices_f = indices.flatten().cpu().numpy().tolist()
    of_ref = o.to(torch.float32).flatten().cpu().numpy().tolist()
    
    # Write query values
    print(f"Writing {B*H*N*D} query values...")
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    
    # Write key values
    print(f"Writing {B*H*N*D} key values...")
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    
    # Write value values
    print(f"Writing {B*H*N*D} value values...")
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    
    # Write indices
    print(f"Writing {B*H*N*S} indices...")
    for i in trange(B*H*N*S):
        f.write(repr(indices_f[i]))
        f.write(' ')
    
    # Write output values
    print(f"Writing {B*H*N*D} output values...")
    for i in trange(B*H*N*D):
        f.write(repr(of_ref[i]))
        f.write(' ')

print(f"Test data saved to nsa_fwd_{B}x{H}x{N}x{D}_S{S}_B{block_size}.txt")


# REGULAR LINEAR ATTENTION

# import sys
# from tqdm import trange

# import torch
# from einops import rearrange

# # TODO : q scale!

# def pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=False):
#     if scale:
#         q = q * (q.shape[-1] ** -0.5)
#     attn = q @ k.transpose(-2, -1)
#     attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
#     o = attn @ v
#     return o

# B = 1 # keep 1
# H = 1 # keep 1
# N = 1024
# D = 64
# scale = False

# TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn_all'

# if TESTNAME in ['ones_all', 'ones_t0', 'ones_t1', 'ones_t0t1', 'ones_t2']:
#     q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
#     k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
#     v = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
# elif TESTNAME in ['randn_all', 'randn_t0', 'randn_t1', 'randn_t0t1', 'randn_t2']:
#     torch.random.manual_seed(42)
#     q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
#     k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
#     v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
# else:
#     print('Invalid test name')
#     sys.exit(0)

# o = pytorch_ref(q, k, v)

# with open(f'fwd_{B}x{H}x{N}x{D}.txt', 'w') as f:
#     qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
#     kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
#     vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
#     of_ref = o.to(torch.float32).flatten().cpu().numpy().tolist()

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
#         f.write(repr(of_ref[i]))
#         f.write(' ')
