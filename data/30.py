# triton_rms_norm_silu.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_rms_norm_silu(x, weight, eps):
    """Standard PyTorch implementation: silu(rms_norm(x))"""
    # RMS normalization
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    x_rms = weight * x_norm
    # SiLU activation
    return torch.nn.functional.silu(x_rms)

# Triton kernel - fused RMS normalization + SiLU
@triton.jit
def rms_norm_silu_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    row_idx = tl.program_id(axis=0)
    
    # Offsets
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute RMS normalization
    x_squared = x * x
    mean_square = tl.sum(x_squared, axis=0) / n_cols
    rms = tl.sqrt(mean_square + eps)
    x_norm = x / rms
    
    # Scale by weight
    x_rms = x_norm * weight
    
    # Compute SiLU: x * sigmoid(x)
    x_sigmoid = tl.sigmoid(x_rms)
    result = x_rms * x_sigmoid
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

def triton_rms_norm_silu(x, weight, eps=1e-6):
    """Triton wrapper function"""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (n_rows,)
    
    # Launch kernel
    rms_norm_silu_kernel[grid](
        x, weight, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    weight = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-6
    
    torch_result = torch_rms_norm_silu(x, weight, eps)
    triton_result = triton_rms_norm_silu(x, weight, eps)
    
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
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
    ]
    
    for M, N in sizes:
        print(f"Size: ({M}, {N})")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        # Warmup
        for _ in range(5):
            _ = torch_rms_norm_silu(x, weight, eps)
            _ = triton_rms_norm_silu(x, weight, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_rms_norm_silu(x, weight, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_rms_norm_silu(x, weight, eps)
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