import torch
import triton
import triton.language as tl

def python_exp2_log2(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.log2(torch.exp2(x))
    # PYTHON_BODY_END

@triton.jit
def exp2_log2_kernel(
    x_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    exp2_x = tl.exp2(x)
    log2_exp2_x = tl.log2(exp2_x)
    tl.store(output_ptr + offsets, log2_exp2_x, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_exp2_log2(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    exp2_log2_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: exp2_log2 ---")
    
    input_tensor = torch.randn((16, 16), device='cuda')

    python_result = python_exp2_log2(input_tensor.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_exp2_log2(input_tensor)

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
