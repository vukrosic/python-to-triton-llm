import torch
import triton
import triton.language as tl

def python_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.clamp(x, min_val, max_val)
    # PYTHON_BODY_END

@triton.jit
def clamp_kernel(
    x_ptr,
    output_ptr,
    num_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.maximum(min_val, tl.minimum(x, max_val))
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    clamp_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        num_elements=num_elements,
        min_val=min_val,
        max_val=max_val,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: clamp ---")
    
    input_tensor = torch.randn((16, 16), device='cuda')
    min_val = -0.5
    max_val = 0.5

    python_result = python_clamp(input_tensor.cpu(), min_val, max_val)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_clamp(input_tensor, min_val, max_val)

    are_close = torch.allclose(python_result.cuda(), triton_result)
    
    if are_close:
        print("✅ PASSED")
        sys.exit(0)
    else:
        print("❌ FAILED")
        abs_diff = torch.abs(python_result.cuda() - triton_result)
        max_abs_diff = torch.max(abs_diff)
        print(f"  - Max Absolute Difference: {max_abs_diff.item()}")
        sys.exit(1)
