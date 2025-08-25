import torch
import triton
import triton.language as tl

def python_full(shape: tuple[int, ...], fill_value: float) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.full(shape, fill_value)
    # PYTHON_BODY_END

@triton.jit
def full_kernel(
    output_ptr,
    num_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    tl.store(output_ptr + offsets, fill_value, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_full(shape: tuple[int, ...], fill_value: float) -> torch.Tensor:
    output = torch.empty(shape, device='cuda')
    num_elements = output.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    full_kernel[grid](
        output_ptr=output,
        num_elements=num_elements,
        fill_value=fill_value,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: full ---")
    
    shape = (16, 16)
    fill_value = 3.14

    python_result = python_full(shape, fill_value)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_full(shape, fill_value)

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
