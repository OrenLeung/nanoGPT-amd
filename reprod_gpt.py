import contextlib

import torch
import torch.nn.functional as F
import torch.nn as nn

from pydantic.dataclasses import dataclass

@dataclass
class GPTConfig:
    n_layers: int    # L
    n_heads: int     # H
    d_embd: int      # E
    max_seq_len: int = 1024
    vocab_size: int  = 50304 # V
    arch_name: str = 'gpt'

    @staticmethod
    def estimate_flops_per_token(model, config):
        # get param count
        N = sum(p.numel() for p in model.parameters())
        
        # print param count in B
        print(f"Param count: {N/1e9}B")
                 
        head_dim = config['d_embd'] // config['n_heads'] 
         
        flops_per_token = 6 * N + 12 * config['n_layers'] * config['n_heads'] * head_dim * config['max_seq_len']
        
        return flops_per_token

    def __post_init__(self):
        assert self.d_embd % self.n_heads == 0, 'd_embd must be a multiple of n_heads.'

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, d_embd, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.pos_embd = nn.Embedding(max_seq_len, d_embd)
        
        
        # self.tsfmr_blks = nn.ModuleList(GPTBlock(d_embd, **kwargs) for _ in range(n_layers))
        import transformer_engine.pytorch as te
        self.tsfmr_blks = nn.ModuleList(te.TransformerLayer(
                    d_embd,
                    d_embd * 4,
                    kwargs['n_heads'],
                    layer_number=i+1,
                    # Optional, for speedups
                    fuse_qkv_params=True,
                    attn_input_format='bshd'
                ) 
                for i in range(n_layers)                       
                )
        
        self.out_norm = nn.LayerNorm(d_embd)

    def forward(self, idx_BT):
        pos_T = torch.arange(idx_BT.size(1), dtype=torch.int64, device=idx_BT.device)
        x_BTE = self.tok_embd(idx_BT) + self.pos_embd(pos_T).unsqueeze(0)

        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE)

        x_BTE = self.out_norm(x_BTE)
        logits_BTV = x_BTE @ self.tok_embd.weight.T  # Weight tying

        return logits_BTV

def train(
    gpu_id: int = 0,
    bsz: int = 8,
    grad_acc_steps: int = 8,
):
    torch.manual_seed(3985)
    torch.cuda.set_device(gpu_id)

    cfg_json = {
        "n_layers": 48,
        "n_heads": 25,
        "d_embd": 1600,
        "max_seq_len": 1024,
        "vocab_size": 50304,
    }

    cfg_m = GPTConfig(**cfg_json)
    model = GPT(**cfg_json).to(gpu_id)

    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    flops_per_token = cfg_m.estimate_flops_per_token(model, cfg_json)
    flops_per_iter = flops_per_token * (bsz * cfg_m.max_seq_len)

    flops_promised = 2600e12

    model.train()
    
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    fp8_format = Format.HYBRID
    # Reasonable default setting
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    # Note: wrapped ctx in a function because the te.fp8_autocast object cannot be reused as a context for some reason.
    @contextlib.contextmanager
    def ctx():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                yield

    with ctx():
         for step_idx in range(100):
            input_BT = torch.randint(50304, [8, 1024], dtype=torch.int64).to('cuda:0')
            label_BT = torch.randint(50304, [8, 1024], dtype=torch.int64).to('cuda:0')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            logits_BTV = model(input_BT)
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

            print(f'{(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')

if __name__ == '__main__':
    import fire
    fire.Fire(train)