import json
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gpt import GPTConfig, Fp8GPT
from llama import LLaMAConfig, Fp8LLaMA

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


def train(
    cfg_path: str,
    gpu_id: int = 0,
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 128,
    grad_acc_steps: int = 8,
    pt_compile: bool = False,
    compile_mode: str = 'default',
    profile: bool = False,
    output_dir: str = 'outputs/single_gpu'
):
    logging.basicConfig(level=logging.ERROR)
    torch.manual_seed(3985)
    torch.cuda.set_device(gpu_id)

    # Configure model
    with open(cfg_path) as f:
        cfg_json = json.load(f)
    if cfg_json['arch_name'] == 'gpt':
        cfg_cls, model_cls = GPTConfig, Fp8GPT
    elif cfg_json['arch_name'] == 'llama':
        cfg_cls, model_cls = LLaMAConfig, Fp8LLaMA
    else:
        raise ValueError(f'Model architecture {cfg_json["arch_name"]} not supported.')
    cfg_m = cfg_cls(**cfg_json)
    model = model_cls(**cfg_json).to(gpu_id)
    print(f'Created model {model_cls}')

    if pt_compile:
        print(f'Compiling with {compile_mode} mode')
        model = torch.compile(model, mode=compile_mode)

    # Configure training setup
    dataset = DummyDataset(cfg_m.vocab_size, cfg_m.max_seq_len, bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz, num_workers=n_workers, pin_memory=True, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # Configure profiling
    if profile:
        def trace_export_callback(prof):
            prof.export_chrome_trace(f'{output_dir}/{Path(cfg_path).stem}_trace.json')

        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            with_flops=True,
            on_trace_ready=trace_export_callback
        )
    else:
        prof_ctx = nullcontext()
        flops_per_token = cfg_m.estimate_flops_per_token(model, cfg_json)
        flops_per_iter = flops_per_token * (bsz * cfg_m.max_seq_len)
        if 'H100' in torch.cuda.get_device_name():
            flops_promised = 1979e12
        elif 'MI300X' in torch.cuda.get_device_name():
            flops_promised = 5220e12
        else:
            raise ValueError(f'FLOP/s for device {torch.cuda.get_device_name()} is unknown')

    # Configure FP8
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')

    # Training loop
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.train()
    print('Training start')

    with prof_ctx as prof, tqdm(total=n_steps) as pbar:
        for step_idx, data_batch in enumerate(data_loader):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            input_BT, label_BT = map(lambda t: t.pin_memory().to(gpu_id), data_batch)

            with torch.amp.autocast('cuda', torch.bfloat16):
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits_BTV = model(input_BT, is_first_microbatch=(step_idx % grad_acc_steps == 0))
                    loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                    loss /= grad_acc_steps
            loss.backward()

            if (step_idx + 1) % grad_acc_steps == 0:  # Assume n_steps % grad_acc_steps == 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            end.record()
            torch.cuda.synchronize()

            t = start.elapsed_time(end) / 1e3
            flops_per_sec = flops_per_iter / t
            mfu = flops_per_sec / flops_promised

            pbar.set_description(f'{(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')
            pbar.update()
            
            if profile:
                prof.step()


class DummyDataset(Dataset):
    def __init__(self, vocab_size, max_seq_len, ds_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.ds_len = ds_len

    def __getitem__(self, idx):
        input_T = torch.randint(self.vocab_size, [self.max_seq_len], dtype=torch.int64)
        label_T = torch.cat([input_T[:-1], torch.randint(self.vocab_size, [1])])
        return input_T, label_T

    def __len__(self):
        return self.ds_len
 

if __name__ == '__main__':
    import fire
    fire.Fire(train)
