# triton_mul_add_sub_div_gelu_rms_norm.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps):
    """Standard PyTorch implementation: rms_norm(gelu(x * y + z - w / v))"""
    intermediate = x * y + z - w / v
    x_gelu = torch.nn.functional.gelu(intermediate)
    variance = x_gelu.pow(2).mean(-1, keepdim=True)
    x_norm = x_gelu * torch.rsqrt(variance + eps)
    return weight * x_norm

# Triton kernel - fused mul + add + sub + div + GELU + RMS normalization
@triton.jit
def mul_add_sub_div_gelu_rms_norm_kernel(
    x_ptr, y_ptr, z_ptr, w_ptr, v_ptr, weight_ptr, output_ptr,
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
    y = tl.load(y_ptr + row_start + col_offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + row_start + col_offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + row_start + col_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    
    # Fused operations: x * y + z - w / v
    intermediate = x * y + z - w / v
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = intermediate * 0.7071067811865476  # 1/sqrt(2)
    x_erf = tl.libdevice.erf(x_scaled)
    x_gelu = 0.5 * intermediate * (1 + x_erf)
    
    # Compute RMS normalization
    x_squared = x_gelu * x_gelu
    mean_square = tl.sum(x_squared, axis=0) / n_cols
    rms = tl.sqrt(mean_square + eps)
    x_norm = x_gelu / rms
    
    # Scale by weight
    result = x_norm * weight
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

def triton_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps=1e-6):
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
    mul_add_sub_div_gelu_rms_norm_kernel[grid](
        x, y, z, w, v, weight, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    y = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    z = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    w = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    v = torch.randn(16, 128, device='cuda', dtype=torch.float32) + 1.0  # Ensure no division by zero
    weight = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-6
    
    torch_result = torch_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps)
    triton_result = triton_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps)
    
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
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        z = torch.randn(M, N, device='cuda', dtype=torch.float32)
        w = torch.randn(M, N, device='cuda', dtype=torch.float32)
        v = torch.randn(M, N, device='cuda', dtype=torch.float32) + 1.0  # Ensure no division by zero
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        # Warmup
        for _ in range(5):
            _ = torch_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps)
            _ = triton_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_mul_add_sub_div_gelu_rms_norm(x, y, z, w, v, weight, eps)
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