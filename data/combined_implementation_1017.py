import torch
import triton
import triton.language as tl

def python_zeros_like(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.zeros_like(x)
    # PYTHON_BODY_END

@triton.jit
def zeros_like_kernel(
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    tl.store(output_ptr + offsets, 0, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_zeros_like(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    zeros_like_kernel[grid](
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: zeros_like ---")
    
    input_tensor = torch.randn((16, 16), device='cuda')

    python_result = python_zeros_like(input_tensor.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_zeros_like(input_tensor)

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
