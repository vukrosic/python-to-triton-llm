# triton_abs_log.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_abs_log(x):
    """Standard PyTorch implementation"""
    return torch.log(torch.abs(x))

# Triton kernel - fused element-wise Absolute Value + Logarithm
@triton.jit
def abs_log_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation
    result = tl.log(tl.abs(x))
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_abs_log(x):
    """Triton wrapper function"""
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    abs_log_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32).abs() + 1e-3 # Ensure positive for log
    
    torch_result = torch_abs_log(x)
    triton_result = triton_abs_log(x)
    
    diff = (torch_result - triton_result).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-4:
        print("Correctness test PASSED!\n")
    else:
        print("Correctness test FAILED!")
        print("Sample values:")
        print("PyTorch:", torch_result.flatten()[:5])
        print("Triton: ", triton_result.flatten()[:5])
        raise AssertionError("Results don't match!")

# Benchmark performance
def benchmark():
    print("Benchmarking performance...")
    sizes = [
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32).abs() + 1e-3
        
        # Warmup
        for _ in range(10):
            _ = torch_abs_log(x)
            _ = triton_abs_log(x)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = torch_abs_log(x)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 50
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = triton_abs_log(x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 50
        
        print(f"  PyTorch: {torch_time*1000:.2f} ms")
        print(f"  Triton:  {triton_time*1000:.2f} ms")
        print(f"  Speedup: {torch_time/triton_time:.2f}x\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    test_correctness()
    benchmark()
