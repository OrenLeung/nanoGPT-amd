import sys
import time
from functools import partial
from pathlib import Path

import torch

import train
import train_ddp
import train_fsdp


def main():
    bsz_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    benchmark = partial(benchmark_fn, './bsz_search.csv')

    # GPT2 1.5B DDP bf16
    for bsz in bsz_list:
        benchmark('configs/gpt2-1.5b.json', 'DDP', use_fp8=False, bsz=bsz)
        time.sleep(10)

    # GPT2 1.5B DDP fp8
    for bsz in bsz_list:
        benchmark('configs/gpt2-1.5b.json', 'DDP', use_fp8=True, bsz=bsz)
        time.sleep(10)

    # LLaMA 3.1 70B Proxy 4 FSDP bf16
    for bsz in bsz_list:
        benchmark('configs/llama-3.1-70b-proxy4.json', 'FSDP', use_fp8=False, bsz=bsz)
        time.sleep(10)

    # LLaMA 3.1 70B Proxy 4 FSDP fp8
    for bsz in bsz_list:
        benchmark('configs/llama-3.1-70b-proxy4.json', 'FSDP', use_fp8=True, bsz=bsz)
        time.sleep(10)

    # LLaMA 3.1 8B FSDP bf16
    for bsz in bsz_list:
        benchmark('configs/llama-3.1-8b.json', 'FSDP', use_fp8=False, bsz=bsz)
        time.sleep(10)

    # LLaMA 3.1 8B FSDP fp8
    for bsz in bsz_list:
        benchmark('configs/llama-3.1-8b.json', 'FSDP', use_fp8=True, bsz=bsz)
        time.sleep(10)

    # LLaMA 2 7B FSDP bf16
    for bsz in bsz_list:
        benchmark('configs/llama-2-7b.json', 'FSDP', use_fp8=False, bsz=bsz)
        time.sleep(10)

    # LLaMA 2 7B FSDP fp8
    for bsz in bsz_list:
        benchmark('configs/llama-2-7b.json', 'FSDP', use_fp8=True, bsz=bsz)
        time.sleep(10)

    # Mistral v0.1 7B FSDP bf16
    for bsz in bsz_list:
        benchmark('configs/mistral-v0.1.json', 'FSDP', use_fp8=False, bsz=bsz)
        time.sleep(10)

    # Mistral v0.1 7B FSDP fp8
    for bsz in bsz_list:
        benchmark('configs/mistral-v0.1.json', 'FSDP', use_fp8=True, bsz=bsz)
        time.sleep(10)


def benchmark_fn(log_path, cfg_path, strategy, use_fp8, bsz, **kwargs):
    strategy2train = {'single': train.train, 'DDP': train_ddp.train, 'FSDP': train_fsdp.train}
    train_fn = strategy2train[strategy]

    device_name = torch.cuda.get_device_name()
    model_name = Path(cfg_path).stem
    dtype = 'fp8' if use_fp8 else 'bf16' 

    if not Path(log_path).exists():
        with open(log_path, 'w') as f:
            f.write('Model, Strategy, GPU, dtype, Batch Size, TFLOP/s/GPU, MFU')
    with open(log_path, 'a') as f:
        f.write(f'\n{model_name}, {strategy}, {device_name}, {dtype}, {bsz}, ')

    try:
        train_fn(
            cfg_path=cfg_path,
            bsz=bsz,
            use_fp8=use_fp8,
            pt_compile=(not use_fp8),
            log_path=log_path,
            n_steps=64*8,
            **kwargs
        )
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write('OOM, OOM')


if __name__ == '__main__':
    main()
