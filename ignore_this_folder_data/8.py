# triton_elementwise_operations.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_fused_ops(x, y, alpha, beta):
    """Standard PyTorch implementation: alpha * x^2 + beta * y^3 + sin(x * y)"""
    return alpha * x.pow(2) + beta * y.pow(3) + torch.sin(x * y)

# Triton kernel - fused elementwise operations
@triton.jit
def fused_ops_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    alpha,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to guard memory operations
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Fused operations: alpha * x^2 + beta * y^3 + sin(x * y)
    x_squared = x * x
    y_cubed = y * y * y
    xy = x * y
    sin_xy = tl.sin(xy)
    
    result = alpha * x_squared + beta * y_cubed + sin_xy
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_fused_ops(x, y, alpha, beta):
    """Triton wrapper function"""
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_ops_kernel[grid](
        x, y, output,
        n_elements,
        alpha,
        beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    alpha = 2.5
    beta = 1.3
    
    torch_result = torch_fused_ops(x, y, alpha, beta)
    triton_result = triton_fused_ops(x, y, alpha, beta)
    
    diff = (torch_result - triton_result).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-2:
        print("Correctness test PASSED!\n")
    else:
        print("Correctness test FAILED!")
        raise AssertionError("Results don't match!")

# Benchmark performance
def benchmark():
    print("Benchmarking performance...")
    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        alpha = 1.5
        beta = 2.0
        
        # Warmup
        for _ in range(5):
            _ = torch_fused_ops(x, y, alpha, beta)
            _ = triton_fused_ops(x, y, alpha, beta)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_fused_ops(x, y, alpha, beta)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_fused_ops(x, y, alpha, beta)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 20
        
        print(f"  PyTorch: {torch_time*1000:.2f} ms")
        print(f"  Triton:  {triton_time*1000:.2f} ms")
        print(f"  Speedup: {torch_time/triton_time:.2f}x\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    test_correctness()
    benchmark()