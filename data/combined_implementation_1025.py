import torch
import triton
import triton.language as tl

def python_where_abs(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.abs(torch.where(condition, x, y))
    # PYTHON_BODY_END

@triton.jit
def where_abs_kernel(
    condition_ptr,
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
    condition = tl.load(condition_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = tl.where(condition, x, y)
    result_abs = tl.abs(result)
    tl.store(output_ptr + offsets, result_abs, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_where_abs(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    where_abs_kernel[grid](
        condition_ptr=condition,
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: where_abs ---")
    
    condition = torch.rand((16, 16), device='cuda') > 0.5
    input_tensor1 = torch.randn((16, 16), device='cuda')
    input_tensor2 = torch.randn((16, 16), device='cuda')

    python_result = python_where_abs(condition.cpu(), input_tensor1.cpu(), input_tensor2.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_where_abs(condition, input_tensor1, input_tensor2)

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
