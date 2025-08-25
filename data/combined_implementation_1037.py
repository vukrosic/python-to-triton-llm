import torch
import triton
import triton.language as tl

def python_cat_cos(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.cos(torch.cat((x, y), dim=dim))
    # PYTHON_BODY_END

@triton.jit
def cat_cos_kernel(
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
    result = tl.cos(x)
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_cat_cos(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    cat_tensor = torch.cat((x, y), dim=dim)
    output = torch.empty_like(cat_tensor)
    num_elements = cat_tensor.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    cat_cos_kernel[grid](
        x_ptr=cat_tensor,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: cat_cos ---")
    
    input_tensor1 = torch.randn((16, 16), device='cuda')
    input_tensor2 = torch.randn((16, 16), device='cuda')
    dim = 1

    python_result = python_cat_cos(input_tensor1.cpu(), input_tensor2.cpu(), dim)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_cat_cos(input_tensor1, input_tensor2, dim)

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
