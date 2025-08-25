# triton_matmul_add_rms_norm.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_matmul_add_rms_norm(x, w, bias, residual, weight, eps):
    \"\"\"Standard PyTorch implementation: rms_norm(matmul(x, w) + bias + residual)\"\"\"
    linear = torch.matmul(x, w) + bias
    x_res = linear + residual
    variance = x_res.pow(2).mean(-1, keepdim=True)
    x_norm = x_res * torch.rsqrt(variance + eps)
    return weight * x_norm

# Triton kernel - fused matmul + add + residual + RMS normalization
@triton.jit
def matmul_add_rms_norm_kernel(
    x_ptr, w_ptr, bias_ptr, residual_ptr, weight_ptr, output_ptr,
    x_height, x_width, w_width,
    stride_xh, stride_xw,
    stride_wh, stride_ww,
    stride_rh, stride_rw,
    stride_oh, stride_ow,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Reshape for matrix multiplication
    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, tl.cdiv(x_width, BLOCK_SIZE_K)):
        offs_k_inner = k * BLOCK_SIZE_K + offs_k
        
        # Load x tile
        x_mask = (offs_m < x_height) & (offs_k_inner[None, :] < x_width)
        x_ptrs = x_ptr + offs_m * stride_xh + offs_k_inner[None, :] * stride_xw
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load w tile
        w_mask = (offs_k_inner[:, None] < x_width) & (offs_n < w_width)
        w_ptrs = w_ptr + offs_k_inner[:, None] * stride_wh + offs_n * stride_ww
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Matrix multiply
        accumulator += tl.dot(x, w, allow_tf32=False)
    
    # Add bias
    bias_mask = offs_n < w_width
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    accumulator += bias
    
    # Add residual
    residual_mask = (offs_m < x_height) & (offs_n < w_width)
    residual_ptrs = residual_ptr + offs_m * stride_rh + offs_n * stride_rw
    residual = tl.load(residual_ptrs, mask=residual_mask, other=0.0)
    x_res = accumulator + residual
    
    # Compute RMS normalization
    x_squared = x_res * x_res
    mean_square = tl.sum(x_squared, axis=1) / w_width
    mean_square = mean_square[:, None]
    rms = tl.sqrt(mean_square + eps)
    x_norm = x_res / rms
    
    # Load weight
    weight_mask = offs_n < w_width
    weight_ptrs = weight_ptr + offs_n
    weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
    
    # Scale by weight
    result = x_norm * weight
    
    # Store result
    output_mask = (offs_m < x_height) & (offs_n < w_width)
    output_ptrs = output_ptr + offs_m * stride_oh + offs_n * stride_ow
    tl.store(output_ptrs, result, mask=output_mask)

def triton_matmul_add_rms_norm(x, w, bias, residual, weight, eps=1e-6):
    \"\"\"Triton wrapper function\"\"\"
    x_height, x_width = x.shape
    w_width = w.shape[1]
    
    output = torch.empty((x_height, w_width), device=x.device, dtype=x.dtype)
    
    # Block sizes
    BLOCK_SIZE_M = min(64, triton.next_power_of_2(x_height))
    BLOCK_SIZE_N = min(64, triton.next_power_of_2(w_width))
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid = (
        triton.cdiv(x_height, BLOCK_SIZE_M),
        triton.cdiv(w_width, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    matmul_add_rms_norm_kernel[grid](
        x, w, bias, residual, weight, output,
        x_height, x_width, w_width,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        residual.stride(0), residual.stride(1),
        output.stride(0), output.stride(1),
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    w = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    bias = torch.randn(32, device='cuda', dtype=torch.float32)
    residual = torch.randn(32, 32, device='cuda', dtype=torch.float32)
    weight = torch.randn(32, device='cuda', dtype=torch.float32)
    eps = 1e-6
    
    torch_result = torch_matmul_add_rms_norm(x, w, bias, residual, weight, eps)
    triton_result = triton_matmul_add_rms_norm(x, w, bias, residual, weight, eps)
    
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
        (64, 32, 16),
        (128, 64, 32),
        (256, 128, 64),
    ]
    
    for M, K, N in sizes:
        print(f"Size: ({M}, {K}) x ({K}, {N})")
        x = torch.randn(M, K, device='cuda', dtype=torch.float32)
        w = torch.randn(K, N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        residual = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        # Warmup
        for _ in range(5):
            _ = torch_matmul_add_rms_norm(x, w, bias, residual, weight, eps)
            _ = triton_matmul_add_rms_norm(x, w, bias, residual, weight, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_matmul_add_rms_norm(x, w, bias, residual, weight, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_matmul_add_rms_norm(x, w, bias, residual, weight, eps)
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