import torch
import triton
import triton.language as tl

# --- Python Implementation ---
def python_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    result = x + y
    # PYTHON_BODY_END
    return result

# --- Triton Implementation ---
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TRITON_KERNEL_BODY_START
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# --- Test ---
if __name__ == '__main__':
    print("--- Running Tests for Element-wise Addition ---")
    
    test_configs = [
        {'size': 256},
        {'size': 1024},
    ]
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test Case {i+1}: size={config['size']} ---")
        
        torch.manual_seed(i)
        x = torch.randn(config['size'], dtype=torch.float32)
        y = torch.randn(config['size'], dtype=torch.float32)
        
        python_result = python_add(x, y)
        
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            triton_result = triton_add(x_cuda, y_cuda)
            
            are_close = torch.allclose(python_result.cuda(), triton_result)
            
            if are_close:
                print("✅ PASSED: Results are close.")
            else:
                print("❌ FAILED: Results are NOT close.")
                all_passed = False
        else:
            print("SKIPPED: CUDA not available.")

    print("\n--- Overall Test Summary ---")
    if all_passed:
        print("✅ All test cases passed!")
    else:
        print("❌ Some test cases failed.")