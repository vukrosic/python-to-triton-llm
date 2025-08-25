# triton_add_mul_sum.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_add_mul_sum(x, y, z, scale):
    """Standard PyTorch implementation: sum((x + y) * z * scale, dim=-1)"""
    intermediate = (x + y) * z * scale
    return torch.sum(intermediate, dim=-1)

# Triton kernel - fused add + mul + scale + sum
@triton.jit
def add_mul_sum_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID - each program processes one row
    pid = tl.program_id(axis=0)
    
    # Calculate the row offset
    row_start = pid * n_cols
    
    # Accumulator for the sum
    accumulator = tl.zeros((), dtype=tl.float32)
    
    # Process the row in blocks
    for col_off in range(0, n_cols, BLOCK_SIZE):
        # Calculate the offsets for this block
        block_start = col_off
        block_end = min(col_off + BLOCK_SIZE, n_cols)
        block_size = block_end - col_off
        
        # Create offsets for loading
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < block_size
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        
        # Fused operations: (x + y) * z * scale
        intermediate = (x + y) * z * scale
        
        # Accumulate the sum
        accumulator += tl.sum(intermediate, axis=0)
    
    # Store the final result
    tl.store(output_ptr + pid, accumulator)

def triton_add_mul_sum(x, y, z, scale):
    """Triton wrapper function"""
    n_rows, n_cols = x.shape
    
    # Output tensor - one value per row
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    # Block size
    BLOCK_SIZE = 128
    
    # Grid configuration - one program per row
    grid = (n_rows,)
    
    # Launch kernel
    add_mul_sum_kernel[grid](
        x, y, z, output,
        n_cols,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
    y = torch.randn(32, 128, device='cuda', dtype=torch.float32)
    z = torch.randn(32, 128, device='cuda', dtype=torch.float32)
    scale = 2.5
    
    torch_result = torch_add_mul_sum(x, y, z, scale)
    triton_result = triton_add_mul_sum(x, y, z, scale)
    
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
        (64, 256),
        (128, 512),
        (256, 1024),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        z = torch.randn(M, N, device='cuda', dtype=torch.float32)
        scale = 1.5
        
        # Warmup
        for _ in range(5):
            _ = torch_add_mul_sum(x, y, z, scale)
            _ = triton_add_mul_sum(x, y, z, scale)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_add_mul_sum(x, y, z, scale)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_add_mul_sum(x, y, z, scale)
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