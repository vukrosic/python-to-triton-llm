import torch
import triton
import triton.language as tl

def python_atomic_add(x: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    for i, v in zip(indices, values):
        x[i] += v
    return x
    # PYTHON_BODY_END

@triton.jit
def atomic_add_kernel(
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
    tl.atomic_add(x_ptr + indices, values, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_atomic_add(x: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    num_elements = indices.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    atomic_add_kernel[grid](
        x_ptr=x,
        indices_ptr=indices,
        values_ptr=values,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return x

if __name__ == '__main__':
    import sys

    print("--- Running Test: atomic_add ---")
    
    input_tensor = torch.zeros((1024,), device='cuda')
    indices = torch.randint(0, 1024, (2048,), device='cuda')
    values = torch.randn((2048,), device='cuda')

    python_result = torch.zeros_like(input_tensor)
    # PyTorch doesn't have a direct equivalent of atomic_add that works on tensors in the same way.
    # We can simulate it for testing by iterating and adding.
    # A better way is to use torch.Tensor.put_ with accumulate=True
    python_result.index_add_(0, indices.cpu(), values.cpu())


    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_atomic_add(input_tensor, indices, values)

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
