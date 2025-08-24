# triton_gelu_silu_layer_norm_gelu_silu_softmax.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps):
    """Standard PyTorch implementation: softmax(gelu(layer_norm(gelu(x) * silu(x))) * silu(layer_norm(gelu(x) * silu(x))))"""
    x_gelu = torch.nn.functional.gelu(x)
    x_silu = torch.nn.functional.silu(x)
    x_combined = x_gelu * x_silu
    x_norm = torch.nn.functional.layer_norm(x_combined, normalized_shape, weight, bias, eps)
    x_gelu2 = torch.nn.functional.gelu(x_norm)
    x_silu2 = torch.nn.functional.silu(x_norm)
    x_combined2 = x_gelu2 * x_silu2
    return torch.softmax(x_combined2, dim=-1)

# Triton kernel - fused GELU * SiLU + layer norm + GELU * SiLU + softmax
@triton.jit
def gelu_silu_layer_norm_gelu_silu_softmax_kernel(
    x_ptr, output_ptr, weight_ptr, bias_ptr,
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
    
    # Compute first GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled1 = x * 0.7071067811865476  # 1/sqrt(2)
    x_erf1 = tl.libdevice.erf(x_scaled1)
    x_gelu1 = 0.5 * x * (1 + x_erf1)
    
    # Compute first SiLU: x * sigmoid(x)
    x_sigmoid1 = tl.sigmoid(x)
    x_silu1 = x * x_sigmoid1
    
    # Elementwise multiplication
    combined1 = x_gelu1 * x_silu1
    
    # Layer norm
    mean = tl.sum(combined1, axis=0) / n_cols
    var = tl.sum((combined1 - mean) * (combined1 - mean), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    x_norm = (combined1 - mean) * rstd * weight + bias
    
    # Compute second GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled2 = x_norm * 0.7071067811865476  # 1/sqrt(2)
    x_erf2 = tl.libdevice.erf(x_scaled2)
    x_gelu2 = 0.5 * x_norm * (1 + x_erf2)
    
    # Compute second SiLU: x * sigmoid(x)
    x_sigmoid2 = tl.sigmoid(x_norm)
    x_silu2 = x_norm * x_sigmoid2
    
    # Elementwise multiplication
    combined2 = x_gelu2 * x_silu2
    
    # Softmax activation
    x_minus_max = combined2 - tl.max(combined2, axis=0)
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    
    # Store result
    tl.store(output_ptr + row_offset + col_offsets, result, mask=mask)

def triton_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps=1e-5):
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
    gelu_silu_layer_norm_gelu_silu_softmax_kernel[grid](
        x, output, weight, bias,
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
    weight = torch.randn(128, device='cuda', dtype=torch.float32)
    bias = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-5
    
    torch_result = torch_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps)
    triton_result = triton_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps)
    
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
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-5
        
        # Warmup
        for _ in range(5):
            _ = torch_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps)
            _ = triton_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_gelu_silu_layer_norm_gelu_silu_softmax(x, normalized_shape, weight, bias, eps)
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