# triton_div_hardshrink.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_div_hardshrink(x, y, lambd):
    """Standard PyTorch implementation"""
    return torch.nn.functional.hardshrink(x / y, lambd=lambd)

# Triton kernel - fused element-wise Division + Hardshrink
@triton.jit
def div_hardshrink_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements, lambd,
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
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0) # Avoid division by zero
    
    # Compute fused operation
    val = x / y
    result = tl.where((val > -lambd) & (val < lambd), 0.0, val)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_div_hardshrink(x, y, lambd):
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
    div_hardshrink_kernel[grid](
        x, y, output,
        n_elements,
        lambd,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024, 1024, device='cuda', dtype=torch.float32).abs() + 1e-3 # Ensure non-zero for division
    lambd = 0.5
    
    torch_result = torch_div_hardshrink(x, y, lambd)
    triton_result = triton_div_hardshrink(x, y, lambd)
    
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
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32).abs() + 1e-3
        lambd = 0.5
        
        # Warmup
        for _ in range(10):
            _ = torch_div_hardshrink(x, y, lambd)
            _ = triton_div_hardshrink(x, y, lambd)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = torch_div_hardshrink(x, y, lambd)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 50
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = triton_div_hardshrink(x, y, lambd)
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
