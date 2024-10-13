import time
import torch
import tabulate
from triton.testing import do_bench
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

torch.manual_seed(0)
repeats = 200
warmup = 30
dtype = torch.bfloat16
device = 'cuda'
verbose = False

shapes = [
    (16384, 8192, 1280), # LLama 70B TP8 Shape
    (16384, 1024, 8192), # LLama 70B TP8 Shape
    (16384, 8192, 7168), # LLama 70B TP8 Shape
    (16384, 3584, 8192), # LLama 70B TP8 Shape
    (8192, 8192, 8192) # Square shape
]

results = []

for (m, n, k) in shapes:
    # Matmul benchmark
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(n, k, device=device, dtype=dtype).transpose(-1, -2)
    nFLOPS = 2 * m * n * k
    ms = do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=repeats)
    tflops_matmul = nFLOPS / ms * 1e-9
    time.sleep(3)

    nFLOPS_with_bias = 2 * m * n * k + m * n  # FLOPs for matmul and addition


    # # Linear (with bias) benchmark using F.linear
    weight_with_bias = torch.randn(n, k, device=device, dtype=dtype)
    bias = torch.randn(n, device=device, dtype=dtype)
    input_tensor = torch.randn(m, k, device=device, dtype=dtype)
    ms_linear_with_bias = do_bench(lambda: F.linear(input_tensor, weight_with_bias, bias=bias), warmup=warmup, rep=repeats)
    tflops_linear_with_bias = nFLOPS_with_bias / ms_linear_with_bias * 1e-9
    time.sleep(0.25)

    # # F.linear with autocast bf16 with a, b, and c being fp32
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(n, k, device=device, dtype=torch.float32)
    c = torch.randn(n, device=device, dtype=torch.float32)
    with torch.autocast(dtype=dtype, device_type=device):
        ms_autocast = do_bench(lambda: F.linear(a, b, bias=c), warmup=warmup, rep=repeats)
    tflops_autocast = nFLOPS_with_bias / ms_autocast * 1e-9
    time.sleep(0.25)

    # TE Linear (with FP8 autocast) benchmark
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    input_tensor = torch.randn(m, k, device=device)
    linear_layer = te.Linear(k, n, bias=True).to(device)
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        ms_te_linear = do_bench(lambda: linear_layer(input_tensor), warmup=warmup, rep=repeats)
    tflops_te_linear = nFLOPS_with_bias / ms_te_linear * 1e-9
    time.sleep(0.25)

    # Append the results to the list
    results.append([
        f"({m}, {n}, {k})",
        f"{tflops_matmul:.1f} TFLOPS",
        f"{tflops_linear_with_bias:.1f} TFLOPS",
        f"{tflops_autocast:.1f} TFLOPS",
        f"{tflops_te_linear:.1f} TFLOPS"
    ])

# Print results using tabulate
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "bf16 F.linear (with bias)",
    "bf16 F.linear (with bias & amp)",
    "TE Linear (FP8 autocast)"
]
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(tabulate.tabulate(results, headers=headers, tablefmt="grid"))
