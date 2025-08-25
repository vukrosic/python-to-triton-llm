import torch
import triton
import triton.language as tl

def python_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # PYTHON_BODY_START
    return x.to(dtype)
    # PYTHON_BODY_END

@triton.jit
def cast_kernel(
    x_ptr,
    output_ptr,
    num_elements,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x.to(output_dtype)
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    output = torch.empty_like(x, dtype=dtype)
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    triton_dtype = getattr(tl, str(dtype).split('.')[-1])

    cast_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        num_elements=num_elements,
        output_dtype=triton_dtype,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: cast ---")
    
    input_tensor = torch.randn((16, 16), device='cuda', dtype=torch.float32)
    target_dtype = torch.float16

    python_result = python_cast(input_tensor.cpu(), target_dtype)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_cast(input_tensor, target_dtype)

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
