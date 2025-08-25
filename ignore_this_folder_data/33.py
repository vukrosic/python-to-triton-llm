# triton_gelu_rms_norm_residual.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_gelu_rms_norm_residual(x, residual, weight, eps):
    """Standard PyTorch implementation: rms_norm(gelu(x)) + residual"""
    x_res = x + residual
    x_gelu = torch.nn.functional.gelu(x_res)
    variance = x_gelu.pow(2).mean(-1, keepdim=True)
    x_norm = x_gelu * torch.rsqrt(variance + eps)
    return weight * x_norm

# Triton kernel - fused add + GELU + RMS normalization
@triton.jit
def gelu_rms_norm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr,
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
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # Add residual
    x_res = x + residual
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = x_res * 0.7071067811865476  # 1/sqrt(2)
    x_erf = tl.libdevice.erf(x_scaled)
    gelu_x = 0.5 * x_res * (1 + x_erf)
    
    # Compute RMS normalization
    x_squared = gelu_x * gelu_x
    mean_square = tl.sum(x_squared, axis=0) / n_cols
    rms = tl.sqrt(mean_square + eps)
    x_norm = gelu_x / rms
    
    # Scale by weight
    result = x_norm * weight
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

def triton_gelu_rms_norm_residual(x, residual, weight, eps=1e-6):
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
    gelu_rms_norm_residual_kernel[grid](
        x, residual, weight, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    residual = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    weight = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-6
    
    torch_result = torch_gelu_rms_norm_residual(x, residual, weight, eps)
    triton_result = triton_gelu_rms_norm_residual(x, residual, weight, eps)
    
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
        residual = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        # Warmup
        for _ in range(5):
            _ = torch_gelu_rms_norm_residual(x, residual, weight, eps)
            _ = triton_gelu_rms_norm_residual(x, residual, weight, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_gelu_rms_norm_residual(x, residual, weight, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_gelu_rms_norm_residual(x, residual, weight, eps)
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