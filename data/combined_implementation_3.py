import torch
import triton
import triton.language as tl

# --- Python Implementation ---
def python_where_cos_sin(start: int, end: int, threshold: int) -> torch.Tensor:
    # PYTHON_BODY_START
    x = torch.arange(start, end, dtype=torch.float32)
    condition = x > threshold
    y = torch.where(condition, torch.cos(x), torch.sin(x))
    # PYTHON_BODY_END
    return y

# --- Triton Implementation ---
@triton.jit
def where_cos_sin_kernel(output_ptr, start_val, n_elements, threshold, BLOCK_SIZE: tl.constexpr):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = start_val + offsets
    
    condition = x > threshold
    
    result = tl.where(condition, tl.cos(x.to(tl.float32)), tl.sin(x.to(tl.float32)))
    
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_where_cos_sin(start: int, end: int, threshold: int) -> torch.Tensor:
    n_elements = end - start
    output = torch.empty(n_elements, device='cuda', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    where_cos_sin_kernel[grid](
        output,
        start,
        n_elements,
        threshold,
        BLOCK_SIZE=1024
    )
    return output

# --- Test ---
if __name__ == '__main__':
    start, end, threshold = 0, 128, 64
    
    python_result = python_where_cos_sin(start, end, threshold)
    
    if torch.cuda.is_available():
        triton_result = triton_where_cos_sin(start, end, threshold)
        print("Python vs Triton results are close: ", torch.allclose(python_result.cuda(), triton_result))
