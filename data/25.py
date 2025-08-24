# triton_softmax_rms_norm.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_softmax_rms_norm(x, weight, eps):
    """Standard PyTorch implementation: rms_norm(softmax(x))"""
    x_softmax = torch.softmax(x, dim=-1)
    variance = x_softmax.pow(2).mean(-1, keepdim=True)
    x_norm = x_softmax * torch.rsqrt(variance + eps)
    return weight * x_norm

# Triton kernel - fused softmax + RMS normalization
@triton.jit
def softmax_rms_norm_kernel(
    output_ptr, input_ptr, weight_ptr,
    input_row_stride, output_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get row index
    row_idx = tl.program_id(0)
    
    # Calculate pointers for input and output
    input_row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Load data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Compute RMS normalization
    x_squared = softmax_output * softmax_output
    mean_square = tl.sum(x_squared, axis=0) / n_cols
    rms = tl.sqrt(mean_square + eps)
    x_norm = softmax_output / rms
    
    # Load weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # Scale by weight
    result = x_norm * weight
    
    # Store result
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, result, mask=mask)

def triton_softmax_rms_norm(x, weight, eps=1e-6):
    """Triton wrapper for softmax with RMS normalization"""
    # Ensure input is 2D
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    n_rows, n_cols = x.shape
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Calculate block size (next power of 2 >= n_cols, up to 1024)
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 2)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (n_rows, )
    softmax_rms_norm_kernel[grid](
        output, x, weight,
        x.stride(0), output.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test functions
def test_correctness():
    """Test that Triton implementation matches PyTorch"""
    print("Testing correctness...")
    torch.manual_seed(42)
    
    # Test various sizes
    test_sizes = [(8, 64), (16, 128), (32, 256)]
    
    for rows, cols in test_sizes:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
        weight = torch.randn(cols, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        torch_result = torch_softmax_rms_norm(x, weight, eps)
        triton_result = triton_softmax_rms_norm(x, weight, eps)
        
        diff = (torch_result - triton_result).abs()
        max_diff = diff.max().item()
        
        rtol, atol = 1e-3, 1e-3
        if torch.allclose(torch_result, triton_result, rtol=rtol, atol=atol):
            print(f"Size {rows}x{cols}: PASSED (max diff: {max_diff:.2e})")
        else:
            print(f"Size {rows}x{cols}: FAILED (max diff: {max_diff:.2e})")
            print("Sample values:")
            print("PyTorch:", torch_result[0, :5])
            print("Triton: ", triton_result[0, :5])
            raise AssertionError("Results don't match!")
    
    print("All correctness tests passed!\n")

# Benchmark functions
def benchmark():
    """Benchmark both implementations"""
    print("Running benchmarks...")
    sizes = [
        (16, 128),
        (32, 256),
        (64, 512),
        (128, 1024)
    ]
    
    for rows, cols in sizes:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
        weight = torch.randn(cols, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        # Warmup
        for _ in range(5):
            _ = torch_softmax_rms_norm(x, weight, eps)
            _ = triton_softmax_rms_norm(x, weight, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_softmax_rms_norm(x, weight, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_softmax_rms_norm(x, weight, eps)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 20
        
        speedup = torch_time / triton_time if triton_time > 0 else 0
        print(f"Size {rows:4d}x{cols:4d}: PyTorch {torch_time*1000:.3f} ms | "
              f"Triton {triton_time*1000:.3f} ms | Speedup {speedup:.2f}x")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    
    # Run tests and benchmarks
    test_correctness()
    print()
    benchmark()