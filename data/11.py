# triton_gelu_silu_fusion.py

import torch
import triton
import triton.language as tl
import time

# Standard PyTorch implementation
def torch_gelu_silu_fusion(x, y):
    \"\"\"Standard PyTorch implementation: GELU(x) * SiLU(y)\"\"\"
    gelu_x = torch.nn.functional.gelu(x)
    silu_y = torch.nn.functional.silu(y)
    return gelu_x * silu_y

# Triton kernel - fused GELU * SiLU
@triton.jit
def gelu_silu_kernel(
    x_ptr, y_ptr, output_ptr,
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
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_scaled = x * 0.7071067811865476  # 1/sqrt(2)
    x_erf = tl.libdevice.erf(x_scaled)
    gelu_x = 0.5 * x * (1 + x_erf)
    
    # Compute SiLU: x * sigmoid(x)
    y_sigmoid = tl.sigmoid(y)
    silu_y = y * y_sigmoid
    
    # Elementwise multiplication
    result = gelu_x * silu_y
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_gelu_silu_fusion(x, y, activation=\"gelu_silu\"):
    \"\"\"Triton wrapper function\"\"\"
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    gelu_silu_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Test correctness
def test_correctness():
    print(\"Testing correctness...\")
    torch.manual_seed(42)
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
    y = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
    
    torch_result = torch_gelu_silu_fusion(x, y)
    triton_result = triton_gelu_silu_fusion(x, y)
    
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
        (512, 256),
        (1024, 512),
        (2048, 1024),
    ]
    
    for M, N in sizes:
        print(f\"Size: ({M}, {N})\")
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            _ = torch_gelu_silu_fusion(x, y)
            _ = triton_gelu_silu_fusion(x, y)
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = torch_gelu_silu_fusion(x, y)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 20
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            _ = triton_gelu_silu_fusion(x, y)
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