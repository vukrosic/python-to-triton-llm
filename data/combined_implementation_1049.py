import torch
import triton
import triton.language as tl

def python_atomic_max(x: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    for i, v in zip(indices, values):
        x[i] = max(x[i], v)
    return x
    # PYTHON_BODY_END

@triton.jit
def atomic_max_kernel(
    x_ptr,
    indices_ptr,
    values_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(indices_ptr + offsets, mask=mask)
    values = tl.load(values_ptr + offsets, mask=mask)
    tl.atomic_max(x_ptr + indices, values, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_atomic_max(x: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    num_elements = indices.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    atomic_max_kernel[grid](
        x_ptr=x,
        indices_ptr=indices,
        values_ptr=values,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return x

if __name__ == '__main__':
    import sys

    print("--- Running Test: atomic_max ---")
    
    input_tensor = torch.randn((1024,), device='cuda')
    indices = torch.randint(0, 1024, (2048,), device='cuda')
    values = torch.randn((2048,), device='cuda')

    python_result = input_tensor.cpu().clone()
    # PyTorch doesn't have a direct equivalent of atomic_max that works on tensors in the same way.
    # We can simulate it for testing by iterating and taking the max.
    for i, v in zip(indices.cpu().tolist(), values.cpu().tolist()):
        python_result[i] = max(python_result[i], v)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_atomic_max(input_tensor, indices, values)

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
