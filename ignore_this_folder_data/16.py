# triton_gelu_dropout.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_gelu_dropout(x, p):
    """Standard PyTorch implementation: dropout(GELU(x))"""
    x_gelu = torch.nn.functional.gelu(x)
    return torch.nn.functional.dropout(x_gelu, p)

# Triton kernel - fused GELU + dropout
@triton.jit
def gelu_dropout_kernel(
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
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = x * 0.7071067811865476  # 1/sqrt(2)
    x_erf = tl.libdevice.erf(x_scaled)
    gelu_x = 0.5 * x * (1 + x_erf)
    
    # Apply dropout
    random = tl.rand(pid, offsets)
    dropout_mask = random > p
    result = tl.where(dropout_mask, gelu_x / (1 - p), 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_gelu_dropout(x, p):
    """Triton wrapper function"""
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    gelu_dropout_kernel[grid](
        x, output,
        n_elements, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(512, 256, device='cuda', dtype=torch.float32)
    p = 0.1
    
    # Set seed for consistent dropout
    torch.manual_seed(123)
    torch_result = torch_gelu_dropout(x, p)
    
    torch.manual_seed(123)
    triton_result = triton_gelu_dropout(x, p)
    
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
        (256, 128),
        (512, 256),
        (1024, 512),
        (2048, 1024),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        p = 0.2
        
        # Warmup
        for _ in range(5):
            _ = torch_gelu_dropout(x, p)
            _ = triton_gelu_dropout(x, p)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_gelu_dropout(x, p)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_gelu_dropout(x, p)
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