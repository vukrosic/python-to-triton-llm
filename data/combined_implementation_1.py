import torch
import triton
import triton.language as tl

def python_arange_reshape_exp(start: int, end: int, shape: tuple[int, ...]) -> torch.Tensor:
    # PYTHON_BODY_START
    arange_tensor = torch.arange(start, end, dtype=torch.float32)
    reshaped_tensor = arange_tensor.reshape(shape)
    result_tensor = torch.exp(reshaped_tensor)
    # PYTHON_BODY_END
    return result_tensor

@triton.jit
def arange_reshape_exp_kernel(
    output_ptr,
    start_val,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    arange_vals = start_val + offsets
    exp_vals = tl.exp(arange_vals.to(tl.float32))
    tl.store(output_ptr + offsets, exp_vals, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_arange_reshape_exp(start: int, end: int, shape: tuple[int, ...]) -> torch.Tensor:
    num_elements = end - start
    output_tensor = torch.empty(num_elements, device='cuda', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    arange_reshape_exp_kernel[grid](
        output_ptr=output_tensor,
        start_val=start,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output_tensor.reshape(shape)

if __name__ == '__main__':
    import sys

    print("--- Running Test: arange_reshape_exp ---")
    
    start = 0
    end = 256
    shape = (16, 16)

    python_result = python_arange_reshape_exp(start, end, shape)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_arange_reshape_exp(start, end, shape)

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