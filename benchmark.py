import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
import triton
import triton.testing

# Set the random seed for reproducibility
torch.manual_seed(12345)

# Define dimensions
batch_size = 8 * 1024 # 8 batch size * 1024 seq len
input_dim = 1600 # gpt-2 xl hidden size
output_dim = 1600 * 3 # QKV proj

# Create input tensor
inp = torch.randn((batch_size, input_dim)).cuda()

# Calculate total FLOPs (assuming 2 FLOPs per multiply-add)
total_flops = 2 * batch_size * input_dim * output_dim

# Prepare weights and biases for F.linear
weight = torch.rand((output_dim, input_dim)).cuda()
bias = torch.rand(output_dim).cuda()

# Define the function for benchmarking F.linear
def fn_f_linear():
    out = F.linear(inp, weight, bias)

# Benchmark F.linear using Triton's do_bench
ms_f_linear = triton.testing.do_bench(fn_f_linear, rep=100)
tflops_f_linear = total_flops / ms_f_linear * 1e-9
print(f"F.linear average time per iteration: {ms_f_linear:.6f} ms")
print(f"F.linear TFLOP/s: {tflops_f_linear:.6f}")

# Prepare Transformer Engine's Linear module with FP8 autocast
my_linear = te.Linear(input_dim, output_dim, bias=True).cuda()

from transformer_engine.common.recipe import Format, DelayedScaling
fp8_format = Format.HYBRID
# Reasonable default setting
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

# Warm-up iterations
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    for _ in range(10):
        _ = my_linear(inp)
    torch.cuda.synchronize()

# Define the function for benchmarking te.Linear with FP8 autocast
def fn_te_linear():
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out_fp8 = my_linear(inp)

# Benchmark te.Linear using Triton's do_bench
ms_te_linear = triton.testing.do_bench(fn_te_linear, rep=100)
tflops_te_linear = total_flops * ms_te_linear * 1e-9
print(f"te.Linear with FP8 average time per iteration: {ms_te_linear:.6f} ms")
print(f"te.Linear with FP8 TFLOP/s: {tflops_te_linear:.6f}")
