import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear):
    """Simple PyTorch implementation: LayerNorm + Linear"""
    # Apply LayerNorm
    ln_out = torch.layer_norm(x, (x.size(-1),), weight_ln, bias_ln, eps=1e-5)
    # Apply Linear
    return torch.matmul(ln_out, weight_linear) + bias_linear

# Simplified Triton kernel implementation
@triton.jit
def layer_norm_linear_simple_kernel(
    x_ptr, weight_ln_ptr, bias_ln_ptr, weight_linear_ptr, bias_linear_ptr, output_ptr,
    N, H, K,  # dimensions
    stride_xm, stride_xh,
    stride_wl_h, stride_bl_h,
    stride_wlin_h, stride_wlin_k,
    stride_om, stride_ok,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)  # batch index
    pid_k = tl.program_id(1)  # output dimension
    
    # Pointers for current batch
    x_row_ptr = x_ptr + pid_m * stride_xm
    output_ptr = output_ptr + pid_m * stride_om + pid_k * stride_ok
    
    # Compute mean and variance
    mean = 0.0
    m2 = 0.0
    for h_start in range(0, H, BLOCK_SIZE):
        h_offs = h_start + tl.arange(0, BLOCK_SIZE)
        h_mask = h_offs < H
        
        # Load x values
        x_ptrs = x_row_ptr + h_offs * stride_xh
        x_vals = tl.load(x_ptrs, mask=h_mask, other=0.0)
        
        # Welford's online algorithm for variance
        delta = x_vals - mean
        new_mean = mean + tl.sum(delta * h_mask) / H
        delta2 = x_vals - new_mean
        m2 += tl.sum(delta * delta2 * h_mask)
        mean = new_mean
    
    var = m2 / H
    rstd = tl.rsqrt(var + eps)
    
    # Compute dot product
    accumulator = tl.load(bias_linear_ptr + pid_k)
    for h_start in range(0, H, BLOCK_SIZE):
        h_offs = h_start + tl.arange(0, BLOCK_SIZE)
        h_mask = h_offs < H
        
        # Load data
        x_ptrs = x_row_ptr + h_offs * stride_xh
        x_vals = tl.load(x_ptrs, mask=h_mask, other=0.0)
        
        # Apply layer norm
        norm_val = (x_vals - mean) * rstd
        weight_ln_val = tl.load(weight_ln_ptr + h_offs, mask=h_mask, other=1.0)
        bias_ln_val = tl.load(bias_ln_ptr + h_offs, mask=h_mask, other=0.0)
        ln_out = norm_val * weight_ln_val + bias_ln_val
        
        # Linear transformation
        w_lin_ptrs = weight_linear_ptr + h_offs * stride_wlin_h + pid_k * stride_wlin_k
        w_lin_vals = tl.load(w_lin_ptrs, mask=h_mask, other=0.0)
        
        # Accumulate
        accumulator += tl.sum(ln_out * w_lin_vals)
    
    # Store result
    tl.store(output_ptr, accumulator)

def triton_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear):
    """Simplified Triton implementation: LayerNorm + Linear fused"""
    M, H = x.shape
    _, K = weight_linear.shape
    
    # Allocate output
    output = torch.empty((M, K), device=x.device, dtype=x.dtype)
    
    # Grid and block configuration
    BLOCK_SIZE = min(512, triton.next_power_of_2(H))
    grid = (M, K)
    
    # Launch kernel
    layer_norm_linear_simple_kernel[grid](
        x, weight_ln, bias_ln, weight_linear, bias_linear, output,
        M, H, K,
        x.stride(0), x.stride(1),
        weight_ln.stride(0), bias_ln.stride(0),
        weight_linear.stride(0), weight_linear.stride(1),
        output.stride(0), output.stride(1),
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Test function
def test_correctness():
    print("Testing correctness...")
    torch.manual_seed(42)
    
    # Create test data
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    weight_ln = torch.randn(128, device='cuda', dtype=torch.float32)
    bias_ln = torch.randn(128, device='cuda', dtype=torch.float32)
    weight_linear = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    bias_linear = torch.randn(64, device='cuda', dtype=torch.float32)
    
    # Compute results
    torch_result = torch_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear)
    triton_result = triton_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear)
    
    # Check correctness
    diff = (torch_result - triton_result).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if torch.allclose(torch_result, triton_result, rtol=1e-2, atol=1e-2):
        print("Correctness test passed!\n")
    else:
        print("Test failed!")
        raise AssertionError("Results don't match!")

# Benchmark function
def benchmark():
    print("Running benchmarks...")
    configs = [
        (32, 256, 128),
        (64, 512, 256),
        (128, 1024, 512),
    ]
    
    for batch, hidden, output in configs:
        print(f"Benchmarking B={batch}, H={hidden}, O={output}")
        
        # Create tensors
        x = torch.randn(batch, hidden, device='cuda', dtype=torch.float32)
        weight_ln = torch.randn(hidden, device='cuda', dtype=torch.float32)
        bias_ln = torch.randn(hidden, device='cuda', dtype=torch.float32)
        weight_linear = torch.randn(hidden, output, device='cuda', dtype=torch.float32)
        bias_linear = torch.randn(output, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = torch_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear)
            _ = triton_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = torch_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 10
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = triton_layer_norm_linear_simple(x, weight_ln, bias_ln, weight_linear, bias_linear)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 10
        
        speedup = torch_time / triton_time if triton_time > 0 else 0
        print(f"  PyTorch: {torch_time*1000:.3f} ms")
        print(f"  Triton:  {triton_time*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    
    test_correctness()
    benchmark()