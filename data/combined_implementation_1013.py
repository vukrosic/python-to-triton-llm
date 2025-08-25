import torch
import triton
import triton.language as tl

def python_softmax(x: torch.Tensor) -> torch.Tensor:
    # PYTHON_BODY_START
    return torch.softmax(x, dim=0)
    # PYTHON_BODY_END

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # TRITON_KERNEL_BODY_START
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
    # TRITON_KERNEL_BODY_END

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols)
    )
    return output

if __name__ == '__main__':
    import sys

    print("--- Running Test: softmax ---")
    
    input_tensor = torch.randn((16, 16), device='cuda')

    python_result = python_softmax(input_tensor.cpu())

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available.")
        sys.exit(0)
        
    triton_result = triton_softmax(input_tensor)

    are_close = torch.allclose(python_result.cuda(), triton_result, atol=1e-6)
    
    if are_close:
        print("✅ PASSED")
        sys.exit(0)
    else:
        print("❌ FAILED")
        abs_diff = torch.abs(python_result.cuda() - triton_result)
        max_abs_diff = torch.max(abs_diff)
        print(f"  - Max Absolute Difference: {max_abs_diff.item()}")
        sys.exit(1)
