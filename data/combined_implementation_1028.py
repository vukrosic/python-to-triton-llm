import torch
import triton
import triton.language as tl

def python_min_max(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.maximum(x, torch.minimum(y, z))
    # PYTHON_BODY_END

@triton.jit
def min_max_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
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
    z = tl.load(z_ptr + offsets, mask=mask)
    min_yz = tl.minimum(y, z)
    max_x_min_yz = tl.maximum(x, min_yz)
    tl.store(output_ptr + offsets, max_x_min_yz, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_min_max(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    min_max_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: min_max ---")
    
    input_tensor1 = torch.randn((16, 16), device='cuda')
    input_tensor2 = torch.randn((16, 16), device='cuda')
    input_tensor3 = torch.randn((16, 16), device='cuda')

    python_result = python_min_max(input_tensor1.cpu(), input_tensor2.cpu(), input_tensor3.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_min_max(input_tensor1, input_tensor2, input_tensor3)

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
