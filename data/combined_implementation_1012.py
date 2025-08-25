import torch
import triton
import triton.language as tl

def python_sum(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.sum(x)
    # PYTHON_BODY_END

@triton.jit
def sum_kernel(
    x_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    tl.store(output_ptr + pid, sum_val)
    # TRITON_KERNEL_BODY_END

def triton_sum(x: torch.Tensor) -> torch.Tensor:
    num_elements = x.numel()
    num_blocks = triton.cdiv(num_elements, 1024)
    output = torch.empty(num_blocks, device=x.device, dtype=x.dtype)
    grid = lambda meta: (num_blocks,)
    sum_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return torch.sum(output)

if __name__ == '__main__':
    import sys

    print("--- Running Test: sum ---")
    
    input_tensor = torch.randn((10240,), device='cuda')

    python_result = python_sum(input_tensor.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_sum(input_tensor)

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
