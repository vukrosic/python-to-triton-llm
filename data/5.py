import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_relu_multiply(x, scale):
    """PyTorch implementation: relu(x) * scale"""
    return torch.relu(x) * scale

# Triton kernel implementation
@triton.jit
def relu_multiply_kernel(
    x_ptr, scale_ptr, output_ptr,
    N,  # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load data (assuming contiguous memory layout)
    x_vals = tl.load(x_ptr + offs, mask=mask)
    scale_vals = tl.load(scale_ptr + offs, mask=mask)
    
    # Apply ReLU: max(0, x)
    relu_result = tl.maximum(x_vals, 0.0)
    
    # Multiply by scale
    result = relu_result * scale_vals
    
    # Store result
    tl.store(output_ptr + offs, result, mask=mask)

def triton_relu_multiply(x, scale):
    """Triton implementation: relu(x) * scale"""
    # Get total number of elements
    N = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Define grid
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    relu_multiply_kernel[grid](
        x, scale, output,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Test functions
def test_correctness():
    """Test that Triton implementation matches PyTorch"""
    print("Testing correctness...")
    torch.manual_seed(42)
    
    # Create tensors
    shape = (2048, 1024)
    x = torch.randn(shape, device='cuda', dtype=torch.float32)
    scale = torch.randn(shape, device='cuda', dtype=torch.float32)
    
    # Compute results
    torch_result = torch_relu_multiply(x, scale)
    triton_result = triton_relu_multiply(x, scale)
    
    # Check correctness
    diff = (torch_result - triton_result).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Use appropriate tolerance
    rtol, atol = 1e-5, 1e-5
    if torch.allclose(torch_result, triton_result, rtol=rtol, atol=atol):
        print("Correctness test passed!\n")
    else:
        print("Test failed! Values differ significantly")
        print("First 5 values (PyTorch):", torch_result.flatten()[:5])
        print("First 5 values (Triton): ", triton_result.flatten()[:5])
        raise AssertionError("Results don't match!")

# Benchmark functions
def benchmark():
    """Benchmark both implementations"""
    print("Running benchmarks...")
    configs = [
        (1024, 1024),   # Small
        (4096, 2048),   # Medium
    ]
    
    for h, w in configs:
        print(f"Benchmarking shape=({h}, {w})")
        
        # Create tensors
        shape = (h, w)
        x = torch.randn(shape, device='cuda', dtype=torch.float32)
        scale = torch.randn(shape, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = torch_relu_multiply(x, scale)
            _ = triton_relu_multiply(x, scale)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = torch_relu_multiply(x, scale)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 10
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = triton_relu_multiply(x, scale)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 10
        
        speedup = torch_time / triton_time if triton_time > 0 else 0
        print(f"  PyTorch: {torch_time*1000:.3f} ms")
        print(f"  Triton:  {triton_time*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x\n")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    
    # Run tests and benchmarks
    test_correctness()
    benchmark()