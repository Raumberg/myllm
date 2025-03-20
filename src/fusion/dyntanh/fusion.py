import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HIDDEN_SIZE']
)
@triton.jit
def DynTanhKerr(
    x_ptr, alpha_ptr, weight_ptr, bias_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # indexes and offsets
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HIDDEN_SIZE

    # mem alloc optim
    tl.advice(x_ptr, tl.constexpr.ALLOCATION_POLICY.STATIC)
    
    # load params for the whole layer
    alpha = tl.load(alpha_ptr)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0)
    
    # load inputs
    x = tl.load(x_ptr + row_idx * HIDDEN_SIZE + col_offsets, mask=mask, other=0)
    x = x.to(tl.float32)
    
    # compute, compute and compute once again
    scaled = alpha * tl.minimum(tl.maximum(x, -10.0), 10.0) # mem buffer overflow sheild
    activated = tl.tanh(scaled)
    weighted = activated * weight + bias
    
    # store
    tl.store(output_ptr + row_idx * HIDDEN_SIZE + col_offsets, 
            weighted.to(tl.__getattribute__(DTYPE)), mask=mask)