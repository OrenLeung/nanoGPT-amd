from dataclasses import asdict
from typing import Optional
from pydantic.dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# DDP
import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# FSDP
from functools import partial
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from tqdm import tqdm

# FP8 Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp

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

@dataclass
class LLaMAConfig:
    n_layers: int    # L
    n_heads: int     # H
    n_kv_heads: int  # J
    d_embd: int      # E
    max_seq_len: int # T
    vocab_size: int  # V
    ffn_mult: float
    ffn_factor: int
    rope_base: float
    norm_eps: float
    d_hid: int = Optional[int] # K
    arch_name: str = 'llama'

    def estimate_flops_per_token(self, model, bsz, rank=0):
        head_dim = self.d_embd // self.n_heads
        N = sum(p.numel() for p in model.parameters())  # get param count

        if rank == 0:
            print(f"Number of parameters: {N/1e9:.2f}B")    # print number of billion parameters 

        self.flops_per_token = 6 * N + 12 * self.n_layers * self.n_heads * head_dim * self.max_seq_len

    def __post_init__(self):
        assert self.d_embd % self.n_heads == 0, 'd_embd must be a multiple of n_heads.'
        assert self.d_embd % self.n_kv_heads == 0, 'd_embd must be a multiple of n_kv_heads.'
        assert self.n_kv_heads <= self.n_heads, 'n_kv_heads must not be larger than n_heads.'

        # FFN hidden dimension
        d_hid = int((4 * self.d_embd) * 2 / 3)
        d_hid = int(d_hid * self.ffn_mult)
        self.d_hid = self.ffn_factor * ((d_hid + self.ffn_factor - 1) // self.ffn_factor)                

class Fp8LLaMA(nn.Module):
    def __init__(self, vocab_size, d_embd, n_layers, n_heads, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.tsfmr_blks = nn.ModuleList(
            Fp8LLaMABlock(d_embd, n_heads=n_heads, **kwargs) for _ in range(n_layers)
        )
        self.norm_lm_head = te.LayerNormLinear(
            d_embd, vocab_size, bias=False,
            normalization='RMSNorm', eps=kwargs['norm_eps']
        )

        # Reference: https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
        freq_cis_TE = te.attention.RotaryPositionEmbedding(d_embd//n_heads)(max_seq_len=131072)
        self.register_buffer('freq_cis_TE', freq_cis_TE.to(torch.bfloat16))

    def forward(self, idx_BT, is_first_microbatch):
        x_BTE = self.tok_embd(idx_BT)
        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE, rotary_pos_emb=self.freq_cis_TE, is_first_microbatch=is_first_microbatch)
        logits_BTV = self.norm_lm_head(x_BTE)
        return logits_BTV


class Fp8LLaMABlock(te.TransformerLayer):
    ''' Reference Implementation:
    https://github.com/NVIDIA/TransformerEngine/blob/55dcbb4b02f560d52dc1215a9de348b37487ee3d/docs/examples/te_llama/te_llama.py#L42
    '''
    def __init__(self, d_embd, d_hid, n_heads, n_kv_heads, norm_eps, **kwargs):
        super().__init__(
            hidden_size=d_embd,
            num_attention_heads=n_heads,
            num_gqa_groups=n_heads//n_kv_heads,
            fuse_qkv_params=True,
            attn_input_format='bshd',
            attention_dropout=0.0,
            normalization='RMSNorm',
            layernorm_epsilon=norm_eps,
            ffn_hidden_size=d_hid,
            bias=False,
            activation='swiglu',
            hidden_dropout=0.0
        )

def train(
    bsz: int = 2,
):

    torch.manual_seed(3985)
    world_size = torch.cuda.device_count()
    train_args = (
        world_size,
        bsz
    )
    try:
        mp.spawn(train_fsdp, train_args, nprocs=world_size)
    except:
        destroy_process_group()


def train_fsdp(
    rank, world_size, bsz
):
    # Construct process group
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '30985'})
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    cfg = {
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "d_embd": 4096,
        "max_seq_len": 4096,
        "vocab_size": 128256,
        "ffn_mult": 1.3,
        "ffn_factor": 1024,
        "rope_base": 500000.0,
        "norm_eps": 1e-05,
        "d_hid": 14336,
        "arch_name": "llama"
    }
        
    use_fp8 = True
    grad_acc_steps = 8
    n_steps = 128*8
    # Configure training setup
    cfg_m, model_cls, blk_cls = LLaMAConfig(**cfg), Fp8LLaMA, Fp8LLaMABlock
    model = model_cls(**asdict(cfg_m)).to(rank)
    dprint(rank, f'Loaded {model_cls} model.', end=' ')
    cfg_m.estimate_flops_per_token(model, bsz, rank)  # Need to do before wrapping in FSDP

    data_loader = create_distributed_data_loader(rank, world_size, bsz, n_steps, cfg_m)
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # FSDP
    model = FSDP(
        model,
        device_id=rank,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={blk_cls}),
        use_orig_params=True
    )
    dprint(rank, f'Created FSDP model')

    prepare_te_modules_for_fsdp(model)
    dprint(rank, 'Sharded TE modules for FSDP')

    # Training loop
    loop_iter = configure_train_loop(data_loader, cfg_m, bsz, rank)
    model.train()
    
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
    all_gpus = dist.new_group(backend='nccl')


    for step_idx, data_batch in loop_iter:
        input_BT, label_BT = map(lambda t: t.pin_memory().to(rank), data_batch)

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


    dist.barrier()
    destroy_process_group()


if __name__ == '__main__':
    import fire
    fire.Fire(train)
