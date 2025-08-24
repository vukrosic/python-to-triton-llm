# triton_swiglu_activation.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_swiglu(x):
    """Standard PyTorch implementation of SwiGLU activation"""
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.nn.functional.silu(x2)

# Triton kernel - SwiGLU activation
@triton.jit
def swiglu_kernel(
    x_ptr, output_ptr,
    n_elements,
    half_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks
    mask = offsets < n_elements
    half_mask = offsets < half_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Split into two halves
    # For simplicity, we assume even division
    x1_offsets = offsets
    x2_offsets = offsets + half_elements
    
    x1 = tl.load(x_ptr + x1_offsets, mask=half_mask, other=0.0)
    x2 = tl.load(x_ptr + x2_offsets, mask=half_mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    x2_sigmoid = tl.sigmoid(x2)
    silu_x2 = x2 * x2_sigmoid
    
    # Compute SwiGLU: x1 * SiLU(x2)
    result = x1 * silu_x2
    
    # Store result
    tl.store(output_ptr + x1_offsets, result, mask=half_mask)

def triton_swiglu(x):
    """Triton wrapper function"""
    n_elements = x.numel()
    half_elements = n_elements // 2
    output = torch.empty(x.shape[0], x.shape[1] // 2, device=x.device, dtype=x.dtype)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(half_elements, BLOCK_SIZE),)
    
    # Launch kernel
    swiglu_kernel[grid](
        x, output,
        n_elements,
        half_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(32, 128, device='cuda', dtype=torch.float32)  # Even number of features
    
    torch_result = torch_swiglu(x)
    triton_result = triton_swiglu(x)
    
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
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N*2, device='cuda', dtype=torch.float32)  # Need even features
        
        # Warmup
        for _ in range(5):
            _ = torch_swiglu(x)
            _ = triton_swiglu(x)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_swiglu(x)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_swiglu(x)
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