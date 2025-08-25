import torch
import triton
import triton.language as tl

def python_add_rsqrt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.rsqrt(x + y + 1e-8)
    # PYTHON_BODY_END

@triton.jit
def add_rsqrt_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = tl.rsqrt(x + y + 1e-8)
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_add_rsqrt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    add_rsqrt_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: add_rsqrt ---")
    
    input_tensor1 = torch.rand((16, 16), device='cuda')
    input_tensor2 = torch.rand((16, 16), device='cuda')

    python_result = python_add_rsqrt(input_tensor1.cpu(), input_tensor2.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_add_rsqrt(input_tensor1, input_tensor2)

    are_close = torch.allclose(python_result.cuda(), triton_result, atol=1e-6)
    
    if are_close:
        print("✅ PASSED")
        sys.exit(0)
    else:
        print("❌ FAILED")
        abs_diff = torch.abs(python_result.cuda() - triton_result)
        max_abs_diff = torch.max(abs_diff)
        print(f"  - Max Absolute Difference: {max_abs_diff.item()}")
        sys.exit(1)
