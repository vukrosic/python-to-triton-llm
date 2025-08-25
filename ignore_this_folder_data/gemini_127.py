# triton_add_logsoftmax.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_add_logsoftmax(x, y):
    """Standard PyTorch implementation"""
    return torch.nn.functional.log_softmax(x + y, dim=-1)

# Triton kernel - fused element-wise Addition + LogSoftmax
@triton.jit
def add_logsoftmax_kernel(
    x_ptr, y_ptr, output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Offsets
    row_offset = pid * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load data
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    
    # Compute fused operation
    val = x + y
    # LogSoftmax: val - log(sum(exp(val)))
    numerator = tl.exp(val - tl.max(val, axis=0))
    denominator = tl.sum(numerator, axis=0)
    result = val - tl.log(denominator)
    
    # Store result
    tl.store(output_ptr + row_offset + col_offsets, result, mask=mask)

def triton_add_logsoftmax(x, y):
    """Triton wrapper function"""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Grid configuration
    grid = (n_rows,)
    
    # Launch kernel
    add_logsoftmax_kernel[grid](
        x, y, output,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(4, 256, device='cuda', dtype=torch.float32)
    y = torch.randn(4, 256, device='cuda', dtype=torch.float32)
    
    torch_result = torch_add_logsoftmax(x, y)
    triton_result = triton_add_logsoftmax(x, y)
    
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
        (1024, 256),
        (2048, 512),
        (4096, 1024),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = torch_add_logsoftmax(x, y)
            _ = triton_add_logsoftmax(x, y)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = torch_add_logsoftmax(x, y)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 50
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            _ = triton_add_logsoftmax(x, y)
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
