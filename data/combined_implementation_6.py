import torch
import triton
import triton.language as tl

# --- Python Implementation ---
def python_matrix_transpose(matrix: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    transposed = matrix.t()
    # PYTHON_BODY_END
    return transposed

# --- Triton Implementation ---
@triton.jit
def matrix_transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # TRITON_KERNEL_BODY_START
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load input in row-major order
    input_offsets = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    input_data = tl.load(input_offsets, mask=mask_m[:, None] & mask_n[None, :])
    
    # Store output in column-major order (transposed)
    output_offsets = output_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(output_offsets, input_data, mask=mask_n[:, None] & mask_m[None, :])
    # TRITON_KERNEL_BODY_END

def triton_matrix_transpose(matrix: torch.Tensor) -> torch.Tensor:
    M, N = matrix.shape
    output = torch.empty((N, M), device='cuda', dtype=matrix.dtype)
    
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matrix_transpose_kernel[grid](
        matrix,
        output,
        M,
        N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    return output

# --- Test ---
if __name__ == '__main__':
    print("--- Running Rigorous Tests for Matrix Transpose ---")
    
    test_configs = [
        {'M': 64, 'N': 32},
        {'M': 128, 'N': 128},
        {'M': 256, 'N': 64},
    ]
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test Case {i+1}: M={config['M']}, N={config['N']} ---")
        
        torch.manual_seed(i)
        input_matrix = torch.randn(config['M'], config['N'], dtype=torch.float32)
        
        python_result = python_matrix_transpose(input_matrix)
        
        if torch.any(torch.isnan(python_result)) or torch.any(torch.isinf(python_result)):
            print("❌ FAILED: Python implementation produced NaN/Inf values.")
            all_passed = False
            continue

        if torch.cuda.is_available():
            input_matrix_cuda = input_matrix.cuda()
            triton_result = triton_matrix_transpose(input_matrix_cuda)
            
            if torch.any(torch.isnan(triton_result)) or torch.any(torch.isinf(triton_result)):
                print("❌ FAILED: Triton implementation produced NaN/Inf values.")
                all_passed = False
                continue

            are_close = torch.allclose(python_result.cuda(), triton_result)
            
            if are_close:
                print("✅ PASSED: Results are close.")
            else:
                print("❌ FAILED: Results are NOT close.")
                all_passed = False
                abs_diff = torch.abs(python_result.cuda() - triton_result)
                max_abs_diff = torch.max(abs_diff)
                print(f"  - Max Absolute Difference: {max_abs_diff.item()}")
        else:
            print("SKIPPED: CUDA not available.")

    print("\n--- Overall Test Summary ---")
    if all_passed:
        print("✅ All test cases passed!")
    else:
        print("❌ Some test cases failed.")