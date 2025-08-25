import torch
import triton
import triton.language as tl

def python_broadcast_add(start: int, end: int) -> torch.Tensor:
    # PYTHON_BODY_START
    v = torch.arange(start, end, dtype=torch.float32)
    r = v.unsqueeze(0)
    c = v.unsqueeze(1)
    # PYTHON_BODY_END
    return r + c

@triton.jit
def broadcast_add_kernel(
    output_ptr,
    start_val,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # TRITON_KERNEL_BODY_START
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_m = offs_m < N
    mask_n = offs_n < N

    vals_r = start_val + offs_n
    vals_c = start_val + offs_m

    vals_r = tl.expand_dims(vals_r, 0)
    vals_c = tl.expand_dims(vals_c, 1)
    
    result_block = vals_r + vals_c

    output_offsets = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_offsets, result_block, mask=mask_m[:, None] & mask_n[None, :])
    # TRITON_KERNEL_BODY_END

def triton_broadcast_add(start: int, end: int) -> torch.Tensor:
    N = end - start
    output = torch.empty((N, N), device='cuda', dtype=torch.float32)
    
    BLOCK_SIZE = 16
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    
    broadcast_add_kernel[grid](
        output_ptr=output,
        start_val=start,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

if __name__ == '__main__':
    start = 0
    end = 64

    print("--- Testing Broadcast Add ---")
    
    python_result = python_broadcast_add(start, end)
    print("Python implementation executed.")

    if torch.cuda.is_available():
        triton_result = triton_broadcast_add(start, end)
        print("Triton implementation executed.")
        
        print("\n--- Comparison ---")
        are_close = torch.allclose(python_result.cuda(), triton_result)
        print(f"Are the results close? {are_close}")

        if are_close:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
            if end - start <= 16:
                print("Python result:")
                print(python_result)
                print("Triton result:")
                print(triton_result)
            else:
                print("Matrices are too large to print.")
                diff = torch.abs(python_result.cuda() - triton_result)
                print(f"Max difference: {torch.max(diff)}")

    else:
        print("\nCUDA not available, skipping Triton execution and comparison.")
