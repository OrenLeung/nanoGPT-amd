import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gpt import GPTConfig, GPT
from llama import LLaMAConfig, LLaMA


def train(
    cfg_path: str,
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 64,
    grad_acc_steps: int = 4,
    ckpt_freq: int = 16,
    pt_compile: bool = False,
    output_dir: str = 'outputs/'
):
    torch.manual_seed(3985)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(cfg_path) as f:
        cfg_json = json.load(f)

    if cfg_json['arch_name'] == 'gpt':
        cfg_cls, model_cls = GPTConfig, GPT
    elif cfg_json['arch_name'].startswith('llama'):
        cfg_cls, model_cls = LLaMAConfig, LLaMA
    else:
        raise ValueError(f'Model architecture {cfg_json["arch_name"]} not supported.')
    cfg_m = cfg_cls(**cfg_json)
    model = model_cls(**asdict(cfg_m)).to('cuda')
    if pt_compile:
        model = torch.compile(model)

    data_loader = DataLoader(
        SimulatedDataset(cfg_m.vocab_size, cfg_m.max_seq_len),
        batch_size=bsz, num_workers=n_workers
    )

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    scaler = torch.amp.GradScaler()

    flops_per_token = cfg_m.estimate_flops_per_token(**asdict(cfg_m))
    flops_per_iter = 3 * flops_per_token * (bsz * cfg_m.max_seq_len)

    model.train()
    pbar = tqdm(total=n_steps)

    for step_idx, data_batch in zip(range(n_steps), data_loader):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        input_BT, label_BT = map(lambda t: t.pin_memory().to('cuda', non_blocking=True), data_batch)

        with torch.amp.autocast('cuda', torch.float16):
            logits_BTV = model(input_BT)
            loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
            loss /= grad_acc_steps
        scaler.scale(loss).backward()

        if (step_idx + 1) % grad_acc_steps == 0 or step_idx == n_steps - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        if (step_idx + 1) % ckpt_freq == 0 or step_idx == n_steps - 1:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'scaler': scaler.state_dict()
            }
            torch.save(ckpt, output_dir / 'ckpt.pt')

        end.record()
        torch.cuda.synchronize()

        t = start.elapsed_time(end) / 1e3
        flops_per_sec = flops_per_iter / t
        mfu = flops_per_sec / 989.5e12

        pbar.set_description(f'{(flops_per_sec/1e12):.2f}TFLOP/s MFU={mfu:.2%}')
        pbar.update()

    pbar.close()


class SimulatedDataset(Dataset):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        input_T = torch.randint(self.vocab_size, [self.max_seq_len], dtype=torch.int64)
        label_T = torch.cat([input_T[:-1], torch.randint(self.vocab_size, [1])])
        return input_T, label_T

    def __len__(self):
        return 3985
 

if __name__ == '__main__':
    import fire
    fire.Fire(train)
