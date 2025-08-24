# triton_layer_norm_gelu_silu_layer_norm_softmax.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps):
    """Standard PyTorch implementation: softmax(layer_norm(gelu(layer_norm(x)) * silu(layer_norm(x))))"""
    # First layer normalization
    x_norm1 = torch.nn.functional.layer_norm(x, normalized_shape, weight1, bias1, eps)
    
    # GELU and SiLU activations
    x_gelu = torch.nn.functional.gelu(x_norm1)
    x_silu = torch.nn.functional.silu(x_norm1)
    
    # Elementwise multiplication
    x_combined = x_gelu * x_silu
    
    # Second layer normalization
    x_norm2 = torch.nn.functional.layer_norm(x_combined, normalized_shape, weight2, bias2, eps)
    
    # Softmax
    return torch.softmax(x_norm2, dim=-1)

# Triton kernel - fused layer norm + GELU * SiLU + layer norm + softmax
@triton.jit
def layer_norm_gelu_silu_layer_norm_softmax_kernel(
    x_ptr, output_ptr, weight1_ptr, bias1_ptr, weight2_ptr, bias2_ptr,
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
    
    # First layer norm
    mean1 = tl.sum(x, axis=0) / n_cols
    var1 = tl.sum((x - mean1) * (x - mean1), axis=0) / n_cols
    rstd1 = tl.rsqrt(var1 + eps)
    
    weight1 = tl.load(weight1_ptr + col_offsets, mask=mask, other=0.0)
    bias1 = tl.load(bias1_ptr + col_offsets, mask=mask, other=0.0)
    
    x_norm1 = (x - mean1) * rstd1 * weight1 + bias1
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled_gelu = x_norm1 * 0.7071067811865476  # 1/sqrt(2)
    x_erf_gelu = tl.libdevice.erf(x_scaled_gelu)
    x_gelu = 0.5 * x_norm1 * (1 + x_erf_gelu)
    
    # Compute SiLU: x * sigmoid(x)
    x_sigmoid_silu = tl.sigmoid(x_norm1)
    x_silu = x_norm1 * x_sigmoid_silu
    
    # Elementwise multiplication
    combined = x_gelu * x_silu
    
    # Second layer norm
    mean2 = tl.sum(combined, axis=0) / n_cols
    var2 = tl.sum((combined - mean2) * (combined - mean2), axis=0) / n_cols
    rstd2 = tl.rsqrt(var2 + eps)
    
    weight2 = tl.load(weight2_ptr + col_offsets, mask=mask, other=0.0)
    bias2 = tl.load(bias2_ptr + col_offsets, mask=mask, other=0.0)
    
    x_norm2 = (combined - mean2) * rstd2 * weight2 + bias2
    
    # Softmax activation
    x_minus_max = x_norm2 - tl.max(x_norm2, axis=0)
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    
    # Store result
    tl.store(output_ptr + row_offset + col_offsets, result, mask=mask)

def triton_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps=1e-5):
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
    layer_norm_gelu_silu_layer_norm_softmax_kernel[grid](
        x, output, weight1, bias1, weight2, bias2,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    normalized_shape = (128,)
    weight1 = torch.randn(128, device='cuda', dtype=torch.float32)
    bias1 = torch.randn(128, device='cuda', dtype=torch.float32)
    weight2 = torch.randn(128, device='cuda', dtype=torch.float32)
    bias2 = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-5
    
    torch_result = torch_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps)
    triton_result = triton_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps)
    
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
        normalized_shape = (N,)
        weight1 = torch.randn(N, device='cuda', dtype=torch.float32)
        bias1 = torch.randn(N, device='cuda', dtype=torch.float32)
        weight2 = torch.randn(N, device='cuda', dtype=torch.float32)
        bias2 = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-5
        
        # Warmup
        for _ in range(5):
            _ = torch_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps)
            _ = triton_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_layer_norm_gelu_silu_layer_norm_softmax(x, normalized_shape, weight1, bias1, weight2, bias2, eps)
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