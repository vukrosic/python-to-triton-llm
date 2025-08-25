import torch
import triton
import triton.language as tl

# --- Python Implementation ---
def python_scaled_multiply(start: int, end: int, scale: float) -> torch.Tensor:
    # PYTHON_BODY_START
    x = torch.arange(start, end, dtype=torch.float32)
    y = x * scale
    z = x * y
    # PYTHON_BODY_END
    return z

# --- Triton Implementation ---
@triton.jit
def scaled_multiply_kernel(output_ptr, start_val, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = start_val + offsets
    y = x * scale
    z = x * y
    
    tl.store(output_ptr + offsets, z, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_scaled_multiply(start: int, end: int, scale: float) -> torch.Tensor:
    n_elements = end - start
    output = torch.empty(n_elements, device='cuda', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scaled_multiply_kernel[grid](
        output,
        start,
        n_elements,
        scale,
        BLOCK_SIZE=1024
    )
    return output

# --- Test ---
if __name__ == '__main__':
    start, end, scale = 0, 128, 2.5
    
    python_result = python_scaled_multiply(start, end, scale)
    
    if torch.cuda.is_available():
        triton_result = triton_scaled_multiply(start, end, scale)
        print("Python vs Triton results are close: ", torch.allclose(python_result.cuda(), triton_result))