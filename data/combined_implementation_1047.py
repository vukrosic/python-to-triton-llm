import torch
import triton
import triton.language as tl

def python_randn(shape) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.randn(shape)
    # PYTHON_BODY_END

@triton.jit
def randn_kernel(
    output_ptr,
    seed,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    random_vals = tl.randn(seed, offsets)
    tl.store(output_ptr + offsets, random_vals, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_randn(seed, shape) -> torch.Tensor:
    output = torch.empty(shape, device='cuda', dtype=torch.float32)
    num_elements = output.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    randn_kernel[grid](
        output_ptr=output,
        seed=seed,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: randn ---")
    
    seed = 12345
    shape = (1024, 1024)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_randn(seed, shape)

    # We can't directly compare to torch.randn because the underlying
    # random number generators are different. Instead, we'll check that
    # the mean and std are close to 0 and 1, respectively.
    mean = torch.mean(triton_result)
    std = torch.std(triton_result)

    mean_close = torch.allclose(mean, torch.tensor(0.0, device=mean.device), atol=1e-2)
    std_close = torch.allclose(std, torch.tensor(1.0, device=std.device), atol=1e-2)

    if mean_close and std_close:
        print("✅ PASSED")
        sys.exit(0)
    else:
        print("❌ FAILED")
        print(f"  - Mean: {mean.item()}")
        print(f"  - Std: {std.item()}")
        sys.exit(1)
