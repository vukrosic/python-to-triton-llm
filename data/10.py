# triton_vector_operations.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_vector_ops(a, b, c, alpha, beta):
    """Standard PyTorch implementation: sqrt(alpha * a^2 + beta * b * c + 1)"""
    return torch.sqrt(alpha * a.square() + beta * b * c + 1)

# Triton kernel - fused vector operations
@triton.jit
def vector_ops_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
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
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Fused operations: sqrt(alpha * a^2 + beta * b * c + 1)
    a_squared = a * a
    bc = b * c
    inner = alpha * a_squared + beta * bc + 1
    result = tl.sqrt(inner)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_vector_ops(a, b, c, alpha, beta):
    """Triton wrapper function"""
    n_elements = a.numel()
    output = torch.empty_like(a)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vector_ops_kernel[grid](
        a, b, c, output,
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
    size = (1024, 1024)
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.randn(size, device='cuda', dtype=torch.float32)
    alpha = 2.5
    beta = 1.3
    
    torch_result = torch_vector_ops(a, b, c, alpha, beta)
    triton_result = triton_vector_ops(a, b, c, alpha, beta)
    
    diff = (torch_result - triton_result).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-2:
        print("Correctness test PASSED!
")
    else:
        print("Correctness test FAILED!")
        print("Sample values:")
        print("PyTorch:", torch_result[0, :5])
        print("Triton: ", triton_result[0, :5])
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
        a = torch.randn(M, N, device='cuda', dtype=torch.float32)
        b = torch.randn(M, N, device='cuda', dtype=torch.float32)
        c = torch.randn(M, N, device='cuda', dtype=torch.float32)
        alpha = 1.5
        beta = 2.0
        
        # Warmup
        for _ in range(5):
            _ = torch_vector_ops(a, b, c, alpha, beta)
            _ = triton_vector_ops(a, b, c, alpha, beta)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_vector_ops(a, b, c, alpha, beta)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_vector_ops(a, b, c, alpha, beta)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 20
        
        print(f"  PyTorch: {torch_time*1000:.2f} ms")
        print(f"  Triton:  {triton_time*1000:.2f} ms")
        print(f"  Speedup: {torch_time/triton_time:.2f}x
")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    test_correctness()
    benchmark()