# triton_round_clip.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_round_clip(x, min_val, max_val):
    """Standard PyTorch implementation"""
    return torch.clip(torch.round(x), min_val, max_val)

# Triton kernel - fused element-wise round + Clip
@triton.jit
def round_clip_kernel(
    x_ptr, output_ptr,
    n_elements, min_val, max_val,
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
    rounded_x = tl.round(x)
    result = tl.maximum(tl.minimum(rounded_x, max_val), min_val)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_round_clip(x, min_val, max_val):
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
    round_clip_kernel[grid](
        x, output,
        n_elements, min_val, max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32) * 10 # Make values larger
    min_val = -5.0
    max_val = 5.0
    
    torch_result = torch_round_clip(x, min_val, max_val)
    triton_result = triton_round_clip(x, min_val, max_val)
    
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
        x = torch.randn(M, N, device='cuda', dtype=torch.float32) * 10
        min_val = -5.0
        max_val = 5.0
        
        # Warmup
        for _ in range(10):
            _ = torch_round_clip(x, min_val, max_val)
            _ = triton_round_clip(x, min_val, max_val)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = torch_round_clip(x, min_val, max_val)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 50
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = triton_round_clip(x, min_val, max_val)
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
