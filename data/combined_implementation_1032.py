import torch
import triton
import triton.language as tl

def python_arange_reshape_sum(start: int, end: int, shape: tuple[int, ...]) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.sum(torch.arange(start, end, dtype=torch.float32).reshape(shape))
    # PYTHON_BODY_END

@triton.jit
def arange_reshape_sum_kernel(
    output_ptr,
    start,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = start + offsets
    sum_val = tl.sum(values, axis=0)
    tl.store(output_ptr + pid, sum_val)
    # TRITON_KERNEL_BODY_END

def triton_arange_reshape_sum(start: int, end: int, shape: tuple[int, ...]) -> torch.Tensor:
    num_elements = end - start
    num_blocks = triton.cdiv(num_elements, 1024)
    output = torch.empty(num_blocks, device='cuda', dtype=torch.float32)
    grid = lambda meta: (num_blocks,)
    arange_reshape_sum_kernel[grid](
        output_ptr=output,
        start=start,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return torch.sum(output)

if __name__ == '__main__':
    import sys

    print("--- Running Test: arange_reshape_sum ---")
    
    start = 0
    end = 256
    shape = (16, 16)

    python_result = python_arange_reshape_sum(start, end, shape)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_arange_reshape_sum(start, end, shape)

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
