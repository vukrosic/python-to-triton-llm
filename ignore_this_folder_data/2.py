import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_matmul_relu(x, w, bias):
    """Standard PyTorch implementation of matrix multiplication with bias and ReLU"""
    return torch.relu(torch.matmul(x, w) + bias)

# Triton kernel implementation
@triton.jit
def matmul_relu_kernel(
    x_ptr, w_ptr, bias_ptr, output_ptr,
    x_height, x_width, w_width,
    stride_xh, stride_xw,
    stride_wh, stride_ww,
    stride_oh, stride_ow,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(x_width, BLOCK_SIZE_K)):
        offs_k_inner = k * BLOCK_SIZE_K + offs_k
        
        # Load x tile
        x_mask = (offs_m < x_height) & (offs_k_inner[None, :] < x_width)
        x_ptrs = x_ptr + offs_m * stride_xh + offs_k_inner[None, :] * stride_xw
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load w tile
        w_mask = (offs_k_inner[:, None] < x_width) & (offs_n < w_width)
        w_ptrs = w_ptr + offs_k_inner[:, None] * stride_wh + offs_n * stride_ww
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Accumulate
        accumulator += tl.dot(x, w, allow_tf32=False)
    
    # Add bias
    bias_mask = offs_n < w_width
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    accumulator += bias
    
    # Apply ReLU
    accumulator = tl.maximum(accumulator, 0.0)
    
    # Store result
    output_mask = (offs_m < x_height) & (offs_n < w_width)
    output_ptrs = output_ptr + offs_m * stride_oh + offs_n * stride_ow
    tl.store(output_ptrs, accumulator, mask=output_mask)

def triton_matmul_relu(x, w, bias):
    """Triton wrapper for matrix multiplication with bias and ReLU"""
    x_height, x_width = x.shape
    w_width = w.shape[1]
    
    output = torch.empty((x_height, w_width), device=x.device, dtype=x.dtype)
    
    # Choose block sizes based on output dimensions
    BLOCK_SIZE_M = min(128, triton.next_power_of_2(x_height))
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(w_width))
    BLOCK_SIZE_K = 32
    
    grid = (
        triton.cdiv(x_height, BLOCK_SIZE_M),
        triton.cdiv(w_width, BLOCK_SIZE_N)
    )
    
    matmul_relu_kernel[grid](
        x, w, bias, output,
        x_height, x_width, w_width,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

# Test functions
def test_correctness():
    """Test that Triton implementation matches PyTorch"""
    torch.manual_seed(42)
    x = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    w = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    bias = torch.randn(32, device='cuda', dtype=torch.float32)
    
    torch_result = torch_matmul_relu(x, w, bias)
    triton_result = triton_matmul_relu(x, w, bias)
    
    print("Testing correctness...")
    diff = (torch_result - triton_result).abs()
    max_diff = diff.max().item()
    print(f"Max difference: {max_diff}")
    
    # Use appropriate tolerance for floating point comparisons
    rtol, atol = 1e-2, 1e-2
    if torch.allclose(torch_result, triton_result, rtol=rtol, atol=atol):
        print("Correctness test passed!\n")
    else:
        print(f"Test failed! Max difference: {max_diff} > {atol}")
        print("First 5x5 values comparison:")
        print("PyTorch:")
        print(torch_result[:5, :5])
        print("Triton:")
        print(triton_result[:5, :5])
        print("Differences:")
        print(diff[:5, :5])
        raise AssertionError("Results don't match!")

# Benchmark functions
def benchmark():
    """Benchmark both implementations"""
    sizes = [
        (512, 256, 128),
        (1024, 512, 256),
        (2048, 1024, 512),
    ]
    
    for M, N, K in sizes:
        print(f"Benchmarking M={M}, N={N}, K={K}")
        x = torch.randn(M, K, device='cuda', dtype=torch.float32)
        w = torch.randn(K, N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = torch_matmul_relu(x, w, bias)
            _ = triton_matmul_relu(x, w, bias)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_matmul_relu(x, w, bias)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_matmul_relu(x, w, bias)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 20
        
        print(f"PyTorch: {torch_time*1000:.3f} ms")
        print(f"Triton:  {triton_time*1000:.3f} ms")
        print(f"Speedup: {torch_time/triton_time:.2f}x\n")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    
    # Run tests and benchmarks
    test_correctness()
    benchmark()