from dataclasses import asdict

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
