import torch
import triton
import triton.language as tl

# --- Python Implementation ---
def python_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    y = torch.sigmoid(x)
    # PYTHON_BODY_END
    return y

# --- Triton Implementation ---
@triton.jit
def sigmoid_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.sigmoid(x)
    tl.store(output_ptr + offsets, result, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output

# --- Test ---
if __name__ == '__main__':
    print("--- Running Rigorous Tests for Sigmoid ---")
    
    test_configs = [
        {'size': 128},
        {'size': 1024},
        {'size': 2048},
        {'size': 4096},
    ]
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test Case {i+1}: size={config['size']} ---")
        
        torch.manual_seed(i)
        input_tensor = torch.randn(config['size'], dtype=torch.float32)
        
        python_result = python_sigmoid(input_tensor)
        
        if torch.any(torch.isnan(python_result)) or torch.any(torch.isinf(python_result)):
            print("❌ FAILED: Python implementation produced NaN/Inf values.")
            all_passed = False
            continue

        if torch.cuda.is_available():
            input_tensor_cuda = input_tensor.cuda()
            triton_result = triton_sigmoid(input_tensor_cuda)
            
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
                rel_diff = abs_diff / torch.abs(python_result.cuda())
                max_rel_diff = torch.max(rel_diff)
                print(f"  - Max Absolute Difference: {max_abs_diff.item()}")
                print(f"  - Max Relative Difference: {max_rel_diff.item()}")
        else:
            print("SKIPPED: CUDA not available.")

    print("\n--- Overall Test Summary ---")
    if all_passed:
        print("✅ All test cases passed!")
    else:
        print("❌ Some test cases failed.")
