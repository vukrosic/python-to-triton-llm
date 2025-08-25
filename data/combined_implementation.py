import torch
import triton
import triton.language as tl

def python_arange_reshape_exp(start: int, end: int, shape: tuple[int, ...]) -> torch.Tensor:
    arange_tensor = torch.arange(start, end, dtype=torch.float32)
    reshaped_tensor = arange_tensor.reshape(shape)
    result_tensor = torch.exp(reshaped_tensor)
    return result_tensor

@triton.jit
def arange_reshape_exp_kernel(
    output_ptr,
    start_val,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    arange_vals = start_val + offsets
    exp_vals = tl.exp(arange_vals.to(tl.float32))
    tl.store(output_ptr + offsets, exp_vals, mask=mask)

def triton_arange_reshape_exp(start: int, end: int, shape: tuple[int, ...]) -> torch.Tensor:
    num_elements = end - start
    output_tensor = torch.empty(num_elements, device='cuda', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    arange_reshape_exp_kernel[grid](
        output_ptr=output_tensor,
        start_val=start,
        num_elements=num_elements,
        BLOCK_SIZE=1024,
    )
    return output_tensor.reshape(shape)

if __name__ == '__main__':
    start = 0
    end = 256
    shape = (16, 16)

    print("--- Testing Implementations ---")
    
    python_result = python_arange_reshape_exp(start, end, shape)
    print("Python implementation executed.")

    if torch.cuda.is_available():
        triton_result = triton_arange_reshape_exp(start, end, shape)
        print("Triton implementation executed.")
        
        print("\n--- Comparison ---")
        are_close = torch.allclose(python_result.cuda(), triton_result)
        print(f"Are the results close? {are_close}")

        if are_close:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
            print("Python result:")
            print(python_result)
            print("Triton result:")
            print(triton_result)
            
    else:
        print("\nCUDA not available, skipping Triton execution and comparison.")