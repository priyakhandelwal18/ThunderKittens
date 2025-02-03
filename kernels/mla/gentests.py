import torch
from tqdm import trange
import sys
import math

# # Import from ohara modules
# from ohara.modules.norm import RMSNorm
# from ohara.embedings_pos.rotatry import precompute_freqs_cis, apply_rope

# Only generate a single batch/head of data to simplify testing
B = 1
N = int(sys.argv[1])  # Sequence length
D = int(sys.argv[2])  # Model dimension
H_QO = int(sys.argv[3])  # Number of query heads
H_KV = int(sys.argv[4])  # Number of key-value heads

causal = False  # Flag for causal attention

torch.random.manual_seed(42)
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

torch.random.manual_seed(42)
q = (torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
k = (torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
v = (torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
grad_output = (torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda'))

# --- Apply RoPE for MLA Attention ---
def precompute_freqs_cis(seq_len, dim, base=10000.0):
    """
    Precompute complex exponentials for RoPE.
    
    Args:
    - seq_len: Sequence length
    - dim: Dimension of the embedding (typically head dimension)
    - base: Base for frequency computation (default 10000 as in original paper)
    
    Returns:
    - freqs: Complex tensor with precomputed rotation frequencies
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    emb = torch.cat([freqs, freqs], dim=-1)
    return torch.polar(torch.ones_like(emb), emb)


def apply_rope(x, freqs_cis):
    """
    Apply Rotary Positional Embedding (RoPE) to input tensor.
    
    Args:
    - x: Input tensor of shape (..., seq_len, dim)
    - freqs_cis: Precomputed complex frequencies
    
    Returns:
    - Rotated tensor
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[-2]].to(x.device)
    rotated_x_complex = x_complex * freqs_cis
    return torch.view_as_real(rotated_x_complex).reshape(x.shape).to(x.dtype)


freqs_cis = precompute_freqs_cis(N, D // 2).to(q.device).to(q.dtype)

q_rot = apply_rope(q, freqs_cis)
k_rot = apply_rope(k, freqs_cis)
softmax_scale = 1 / math.sqrt(D)
q_rot = q_rot * softmax_scale

scores = torch.matmul(q_rot, k_rot.transpose(2, 3))

if causal:
    mask = torch.full((N, N), float('-inf'), device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask

scores = torch.nn.functional.softmax(scores, dim=-1).type_as(q)

output = torch.matmul(scores, v)

# --- Backward Pass ---
output.backward(grad_output)

# Get gradients for Q, K, and V
q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

# Debug info for output and gradients
print("--------------------------------------")
print("Q shape: ",      q.shape)
print("K shape: ",      k.shape)
print("V shape: ",      v.shape)
print("O shape: ",      output.shape)
print("Q grad shape: ", q_grad.shape)
print("K grad shape: ", k_grad.shape)
print("V grad shape: ", v_grad.shape)
print("--------------------------------------")

# Print average magnitude of tensors for stability checks
print(f'Average magnitude of OUTPUT tensor: {output.abs().mean()}')
print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean()}')
print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean()}')
print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean()}')
print("--------------------------------------")

# --- Save output and gradients to file ---
filename = f'randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV'
if causal:
    filename += '_causal'
filename += '.txt'

with open(filename, 'w') as f:
    # Save tensors to file
    tensors_to_save = {
        "q": q, "k": k, "v": v, "output": output,
        "grad_output": grad_output, "q_grad": q_grad,
        "k_grad": k_grad, "v_grad": v_grad
    }

    # Write tensor values to the file in a flattened format
    for name, tensor in tensors_to_save.items():
        flat_tensor = tensor.to(torch.float32).flatten().detach().cpu().numpy()
        for value in trange(flat_tensor.size):
            f.write(repr(float(flat_tensor[value])) + ' ')
