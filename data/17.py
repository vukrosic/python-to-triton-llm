# triton_residual_add_norm.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_residual_add_norm(x, residual, weight, bias, eps):
    \"\"\"Standard PyTorch implementation: layer_norm(x + residual)\"\"\"
    x_res = x + residual
    return torch.nn.functional.layer_norm(x_res, x_res.shape[1:], weight, bias, eps)

# Triton kernel - fused residual + layer norm
@triton.jit
def residual_add_norm_kernel(
    x_ptr, residual_ptr, output_ptr, weight_ptr, bias_ptr,
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
    
    # Residual connection
    x_res = x + residual
    
    # Layer normalization
    mean = tl.sum(x_res, axis=0) / n_cols
    var = tl.sum((x_res - mean) * (x_res - mean), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    result = (x_res - mean) * rstd * weight + bias
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

def triton_residual_add_norm(x, residual, weight, bias, eps=1e-5):
    \"\"\"Triton wrapper function\"\"\"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (n_rows,)
    
    # Launch kernel
    residual_add_norm_kernel[grid](
        x, residual, output, weight, bias,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print(\"Testing correctness...\")
    torch.manual_seed(42)
    x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
    residual = torch.randn(32, 128, device='cuda', dtype=torch.float32)
    weight = torch.randn(128, device='cuda', dtype=torch.float32)
    bias = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-5
    
    torch_result = torch_residual_add_norm(x, residual, weight, bias, eps)
    triton_result = triton_residual_add_norm(x, residual, weight, bias, eps)
    
    diff = (torch_result - triton_result).abs().max().item()
    print(f\"Max difference: {diff}\")
    
    if diff < 1e-2:
        print(\"Correctness test PASSED!\\n\")
    else:
        print(\"Correctness test FAILED!\")
        print(\"Sample values:\")
        print(\"PyTorch:\", torch_result[0, :5])
        print(\"Triton: \", triton_result[0, :5])
        raise AssertionError(\"Results don't match!\")

# Benchmark performance
def benchmark():
    print(\"Benchmarking performance...\")
    sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]
    
    for M, N in sizes:
        print(f\"Size: ({M}, {N})\")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        residual = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-5
        
        # Warmup
        for _ in range(5):
            _ = torch_residual_add_norm(x, residual, weight, bias, eps)
            _ = triton_residual_add_norm(x, residual, weight, bias, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_residual_add_norm(x, residual, weight, bias, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_residual_add_norm(x, residual, weight, bias, eps)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 20
        
        print(f\"  PyTorch: {torch_time*1000:.2f} ms\")
        print(f\"  Triton:  {triton_time*1000:.2f} ms\")
        print(f\"  Speedup: {torch_time/triton_time:.2f}x\\n\")

if __name__ == \"__main__\":
    if not torch.cuda.is_available():
        print(\"CUDA not available!\")
        exit(1)
    
    test_correctness()
    benchmark()