import torch
import triton
import triton.language as tl

def python_cdiv(x: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return (x + div - 1) // div
    # PYTHON_BODY_END

@triton.jit
def cdiv_kernel(
    x_ptr,
    div_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    div = tl.load(div_ptr + offsets, mask=mask)
    result = tl.cdiv(x, div)
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_cdiv(x: torch.Tensor, div: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x, dtype=torch.int32)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    cdiv_kernel[grid](
        x_ptr=x,
        div_ptr=div,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: cdiv ---")
    
    input_tensor1 = torch.randint(1, 100, (16, 16), device='cuda', dtype=torch.int32)
    input_tensor2 = torch.randint(1, 20, (16, 16), device='cuda', dtype=torch.int32)

    python_result = python_cdiv(input_tensor1.cpu(), input_tensor2.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_cdiv(input_tensor1, input_tensor2)

    are_close = torch.allclose(python_result.cuda().float(), triton_result.float())
    
    if are_close:
        print("✅ PASSED")
        sys.exit(0)
    else:
        print("❌ FAILED")
        abs_diff = torch.abs(python_result.cuda() - triton_result)
        max_abs_diff = torch.max(abs_diff)
        print(f"  - Max Absolute Difference: {max_abs_diff.item()}")
        sys.exit(1)
