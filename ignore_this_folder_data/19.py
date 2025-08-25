# triton_silu_dropout.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_silu_dropout(x, p):
    """Standard PyTorch implementation: dropout(SiLU(x))"""
    x_silu = torch.nn.functional.silu(x)
    return torch.nn.functional.dropout(x_silu, p)

# Triton kernel - fused SiLU + dropout
@triton.jit
def silu_dropout_kernel(
    x_ptr, output_ptr,
    n_elements, p,
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
    
    # Compute SiLU: x * sigmoid(x)
    x_sigmoid = tl.sigmoid(x)
    silu_x = x * x_sigmoid
    
    # Apply dropout
    random = tl.rand(pid, offsets)
    dropout_mask = random > p
    result = tl.where(dropout_mask, silu_x / (1 - p), 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_silu_dropout(x, p):
    """Triton wrapper function"""
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    silu_dropout_kernel[grid](
        x, output,
        n_elements, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(256, 512, device='cuda', dtype=torch.float32)
    p = 0.15
    
    # Set seed for consistent dropout
    torch.manual_seed(123)
    torch_result = torch_silu_dropout(x, p)
    
    torch.manual_seed(123)
    triton_result = triton_silu_dropout(x, p)
    
    diff = (torch_result - triton_result).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-2:
        print("Correctness test PASSED!\n")
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
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 2048),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        p = 0.25
        
        # Warmup
        for _ in range(5):
            _ = torch_silu_dropout(x, p)
            _ = triton_silu_dropout(x, p)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_silu_dropout(x, p)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_silu_dropout(x, p)
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