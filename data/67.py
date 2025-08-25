# triton_div_sub_gelu.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_div_sub_gelu(x, y, z):
    \"\"\"Standard PyTorch implementation: gelu(x / y - z)\"\"\"
    intermediate = x / y - z
    return torch.nn.functional.gelu(intermediate)

# Triton kernel - fused div + sub + GELU
@triton.jit
def div_sub_gelu_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to guard memory operations
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # Fused operations: x / y - z
    intermediate = x / y - z
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = intermediate * 0.7071067811865476  # 1/sqrt(2)
    x_erf = tl.libdevice.erf(x_scaled)
    result = 0.5 * intermediate * (1 + x_erf)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_div_sub_gelu(x, y, z):
    \"\"\"Triton wrapper function\"\"\"
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    div_sub_gelu_kernel[grid](
        x, y, z, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print(\"Testing correctness...\")
    torch.manual_seed(42)
    x = torch.randn(256, 512, device='cuda', dtype=torch.float32)
    y = torch.randn(256, 512, device='cuda', dtype=torch.float32) + 1.0  # Ensure no division by zero
    z = torch.randn(256, 512, device='cuda', dtype=torch.float32)
    
    torch_result = torch_div_sub_gelu(x, y, z)
    triton_result = triton_div_sub_gelu(x, y, z)
    
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
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 2048),
    ]
    
    for M, N in sizes:
        print(f\"Size: ({M}, {N})\")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32) + 1.0  # Ensure no division by zero
        z = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = torch_div_sub_gelu(x, y, z)
            _ = triton_div_sub_gelu(x, y, z)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_div_sub_gelu(x, y, z)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_div_sub_gelu(x, y, z)
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