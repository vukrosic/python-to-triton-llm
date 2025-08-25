import torch
import triton
import triton.language as tl

def python_sigmoid_sqrt(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.sqrt(torch.sigmoid(x))
    # PYTHON_BODY_END

@triton.jit
def sigmoid_sqrt_kernel(
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
    sigmoid_x = tl.sigmoid(x)
    sqrt_sigmoid_x = tl.sqrt(sigmoid_x)
    tl.store(output_ptr + offsets, sqrt_sigmoid_x, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_sigmoid_sqrt(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    sigmoid_sqrt_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: sigmoid_sqrt ---")
    
    input_tensor = torch.randn((16, 16), device='cuda')

    python_result = python_sigmoid_sqrt(input_tensor.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_sigmoid_sqrt(input_tensor)

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
