import torch
import triton
import triton.language as tl

def python_trans(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return x.T
    # PYTHON_BODY_END

@triton.jit
def trans_kernel(
    x_ptr,
    output_ptr,
    N, M,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    # TRITON_KERNEL_BODY_START
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    x_ptrs = x_ptr + offs_n[:, None] * M + offs_m[None, :]
    mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    x = tl.load(x_ptrs, mask=mask)

    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, tl.trans(x), mask=tl.trans(mask))
    # TRITON_KERNEL_BODY_END

def triton_trans(x: torch.Tensor) -> torch.Tensor:
    N, M = x.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(M, META['BLOCK_SIZE_M']))
    
    trans_kernel[grid](
        x, output,
        N, M,
        BLOCK_SIZE_N=16, BLOCK_SIZE_M=16
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: trans ---")
    
    input_tensor = torch.randn((32, 64), device='cuda')

    python_result = python_trans(input_tensor.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_trans(input_tensor)

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
