from tqdm import tqdm
from dataclasses import asdict
from pydantic.dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

# DDP
import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# FP8 Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


def dprint(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)
        
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
        
def create_distributed_data_loader(rank, world_size, bsz, n_steps, cfg_m):
    dataset = DummyDataset(cfg_m.vocab_size, cfg_m.max_seq_len, bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz,
        num_workers=8, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    )
    
    return data_loader

@dataclass
class GPTConfig:
    n_layers: int    # L
    n_heads: int     # H
    d_embd: int      # E
    max_seq_len: int = 1024
    vocab_size: int  = 50304 # V
    arch_name: str = 'gpt'

    def estimate_flops_per_token(self, model, bsz, rank=0):
        head_dim = self.d_embd // self.n_heads
        N = sum(p.numel() for p in model.parameters())  # get param count

        if rank == 0:
            print(f"Number of parameters: {N/1e9:.2f}B")    # print number of billion parameters 

        self.flops_per_token = 6 * N + 12 * self.n_layers * self.n_heads * head_dim * self.max_seq_len

    def __post_init__(self):
        assert self.d_embd % self.n_heads == 0, 'd_embd must be a multiple of n_heads.'

class Fp8GPTBlock(te.TransformerLayer):
    def __init__(self, d_embd, n_heads, **kwargs):
        super().__init__(
			d_embd,
			4*d_embd,
			n_heads,
			hidden_dropout=0.0,
			attention_dropout=0.0,
			layer_type='encoder',
			self_attn_mask_type='causal',
			normalization='LayerNorm',
			bias=True,
			activation='gelu',
			attn_input_format='bshd',
			fuse_qkv_params=True
		)

class Fp8GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, d_embd, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.pos_embd = nn.Embedding(max_seq_len, d_embd)
        self.tsfmr_blks = nn.ModuleList(Fp8GPTBlock(d_embd, **kwargs) for _ in range(n_layers))
        self.out_norm = te.LayerNorm(d_embd)

    def forward(self, idx_BT, is_first_microbatch):
        pos_T = torch.arange(idx_BT.size(1), dtype=torch.int64, device=idx_BT.device)
        x_BTE = self.tok_embd(idx_BT) + self.pos_embd(pos_T).unsqueeze(0)

        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE, is_first_microbatch=is_first_microbatch)

		# Couldn't fuse layer norm with linear due to weight tying
        x_BTE = self.out_norm(x_BTE)
        logits_BTV = x_BTE @ self.tok_embd.weight.T
        
        return logits_BTV

def configure_train_loop(data_loader, cfg_m, bsz, rank=0):
    if rank != 0:
        for step_idx, data_batch in enumerate(data_loader):
            yield step_idx, data_batch
        return

    flops_per_iter = cfg_m.flops_per_token * (bsz * cfg_m.max_seq_len)

    flops_promised = 2610e12
    
    with tqdm(total=len(data_loader)) as pbar:
        for step_idx, data_batch in enumerate(data_loader):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            yield step_idx, data_batch

            end.record()
            torch.cuda.synchronize()

            t = start.elapsed_time(end) / 1e3
            flops_per_sec = flops_per_iter / t
            mfu = flops_per_sec / flops_promised

            pbar.set_description(f'[rank0]  {(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')
            pbar.update()
                

def train(bsz = 28):

    torch.manual_seed(3985)
    world_size = torch.cuda.device_count()
    train_args = (
        world_size, bsz)

    try:
        mp.spawn(train_ddp, train_args, nprocs=world_size)
    except:
        destroy_process_group()

def train_ddp(
    rank, world_size, bsz
):
    # Construct process group
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '30985'})
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    cfg = {
        "n_layers": 48,
        "n_heads": 25,
        "d_embd": 1600,
        "max_seq_len": 1024,
        "vocab_size": 50304,
        "arch_name": "gpt"
    }
    
    use_fp8 = True
    grad_acc_steps = 8
    n_steps = 128*8

    # Configure training setup
    model_cls = Fp8GPT
    cfg_m = GPTConfig(**cfg)
    model = Fp8GPT(**asdict(cfg_m)).to(rank)
    dprint(rank, f'Loaded {model_cls} model.', end=' ')
    cfg_m.estimate_flops_per_token(model, bsz, rank)

    data_loader = create_distributed_data_loader(rank, world_size, bsz, n_steps, cfg_m)
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # DDP
    all_gpus = dist.new_group(backend='nccl')
    model = DDP(model, process_group=all_gpus, gradient_as_bucket_view=True)
    dprint(rank, f'Created DDP model')
        
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
    all_gpus = dist.new_group(backend='nccl')

    # Training loop
    loop_iter = configure_train_loop(data_loader, cfg_m, bsz, rank)
    model.train()

    for step_idx, data_batch in loop_iter:
        input_BT, label_BT = map(lambda t: t.pin_memory().to(rank), data_batch)

        # only sync grads when need doing opt.step()
        model.require_backward_grad_sync = (step_idx == grad_acc_steps - 1)

        with torch.amp.autocast('cuda', torch.bfloat16):
            with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe, fp8_group=all_gpus):
                weight_cache = use_fp8 and (step_idx % grad_acc_steps == 0)
                logits_BTV = model(input_BT, is_first_microbatch=weight_cache)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                loss /= grad_acc_steps
        loss.backward()

        if (step_idx + 1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    destroy_process_group()


if __name__ == '__main__':
    import fire
    fire.Fire(train)
