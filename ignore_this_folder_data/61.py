# triton_rms_norm_gelu_silu_rms_norm_silu_softmax.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps):
    \"\"\"Standard PyTorch implementation: softmax(silu(rms_norm(gelu(rms_norm(x)) * silu(rms_norm(x)))))\"\"\"
    # First RMS normalization
    variance1 = x.pow(2).mean(-1, keepdim=True)
    x_norm1 = x * torch.rsqrt(variance1 + eps)
    x_rms1 = weight1 * x_norm1
    
    # GELU and SiLU activations
    x_gelu1 = torch.nn.functional.gelu(x_rms1)
    x_silu1 = torch.nn.functional.silu(x_rms1)
    
    # Elementwise multiplication
    x_combined1 = x_gelu1 * x_silu1
    
    # Second RMS normalization
    variance2 = x_combined1.pow(2).mean(-1, keepdim=True)
    x_norm2 = x_combined1 * torch.rsqrt(variance2 + eps)
    x_rms2 = weight2 * x_norm2
    
    # SiLU activation
    x_silu2 = torch.nn.functional.silu(x_rms2)
    
    # Softmax
    return torch.softmax(x_silu2, dim=-1)

# Triton kernel - fused RMS normalization + GELU * SiLU + RMS normalization + SiLU + softmax
@triton.jit
def rms_norm_gelu_silu_rms_norm_silu_softmax_kernel(
    x_ptr, weight1_ptr, weight2_ptr, output_ptr,
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
    weight1 = tl.load(weight1_ptr + col_offsets, mask=mask, other=0.0)
    weight2 = tl.load(weight2_ptr + col_offsets, mask=mask, other=0.0)
    
    # First RMS normalization
    x_squared1 = x * x
    mean_square1 = tl.sum(x_squared1, axis=0) / n_cols
    rms1 = tl.sqrt(mean_square1 + eps)
    x_norm1 = x / rms1
    x_rms1 = x_norm1 * weight1
    
    # Compute first GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled_gelu1 = x_rms1 * 0.7071067811865476  # 1/sqrt(2)
    x_erf_gelu1 = tl.libdevice.erf(x_scaled_gelu1)
    x_gelu1 = 0.5 * x_rms1 * (1 + x_erf_gelu1)
    
    # Compute first SiLU: x * sigmoid(x)
    x_sigmoid_silu1 = tl.sigmoid(x_rms1)
    x_silu1 = x_rms1 * x_sigmoid_silu1
    
    # Elementwise multiplication
    combined1 = x_gelu1 * x_silu1
    
    # Second RMS normalization
    x_squared2 = combined1 * combined1
    mean_square2 = tl.sum(x_squared2, axis=0) / n_cols
    rms2 = tl.sqrt(mean_square2 + eps)
    x_norm2 = combined1 / rms2
    x_rms2 = x_norm2 * weight2
    
    # Compute second SiLU: x * sigmoid(x)
    x_sigmoid_silu2 = tl.sigmoid(x_rms2)
    x_silu2 = x_rms2 * x_sigmoid_silu2
    
    # Compute softmax
    x_minus_max = x_silu2 - tl.max(x_silu2, axis=0)
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

def triton_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps=1e-6):
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
    rms_norm_gelu_silu_rms_norm_silu_softmax_kernel[grid](
        x, weight1, weight2, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    weight1 = torch.randn(128, device='cuda', dtype=torch.float32)
    weight2 = torch.randn(128, device='cuda', dtype=torch.float32)
    eps = 1e-6
    
    torch_result = torch_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps)
    triton_result = triton_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps)
    
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
        weight1 = torch.randn(N, device='cuda', dtype=torch.float32)
        weight2 = torch.randn(N, device='cuda', dtype=torch.float32)
        eps = 1e-6
        
        # Warmup
        for _ in range(5):
            _ = torch_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps)
            _ = triton_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_rms_norm_gelu_silu_rms_norm_silu_softmax(x, weight1, weight2, eps)
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