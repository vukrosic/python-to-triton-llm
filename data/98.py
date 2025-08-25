# triton_mul_add_div_sub_gelu_layer_norm.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps):
    """Standard PyTorch implementation: layer_norm(gelu(x * y + z / w - v))"""
    intermediate = x * y + z / w - v
    x_gelu = torch.nn.functional.gelu(intermediate)
    return torch.nn.functional.layer_norm(x_gelu, normalized_shape, weight, bias, eps)

# Triton kernel - fused mul + add + div + sub + GELU + layer normalization
@triton.jit
def mul_add_div_sub_gelu_layer_norm_kernel(
    x_ptr, y_ptr, z_ptr, w_ptr, v_ptr, output_ptr, weight_ptr, bias_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Offsets
    row_offset = pid * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load row
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    
    # Fused operations: x * y + z / w - v
    intermediate = x * y + z / w - v
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = intermediate * 0.7071067811865476  # 1/sqrt(2)
    x_erf = tl.libdevice.erf(x_scaled)
    x_gelu = 0.5 * intermediate * (1 + x_erf)
    
    # Layer norm
    mean = tl.sum(x_gelu, axis=0) / n_cols
    var = tl.sum((x_gelu - mean) * (x_gelu - mean), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    result = (x_gelu - mean) * rstd * weight + bias
    
    # Store result
    tl.store(output_ptr + row_offset + col_offsets, result, mask=mask)

def triton_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps=1e-5):
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
    mul_add_div_sub_gelu_layer_norm_kernel[grid](
        x, y, z, w, v, output, weight, bias,
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
    w = torch.randn(16, 128, device='cuda', dtype=torch.float32) + 1.0  # Ensure no division by zero
    v = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    normalized_shape = (128,)
    weight = torch.randn(128, device='cuda', dtype=torch.float32)
    bias = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-5
    
    torch_result = torch_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps)
    triton_result = triton_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps)
    
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
        w = torch.randn(M, N, device='cuda', dtype=torch.float32) + 1.0  # Ensure no division by zero
        v = torch.randn(M, N, device='cuda', dtype=torch.float32)
        normalized_shape = (N,)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-5
        
        # Warmup
        for _ in range(5):
            _ = torch_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps)
            _ = triton_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_mul_add_div_sub_gelu_layer_norm(x, y, z, w, v, normalized_shape, weight, bias, eps)
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