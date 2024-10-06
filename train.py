import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_embd, n_heads, **kwargs):
        super().__init__()
        self.d_head = d_embd // n_heads  # D
        self.attn_proj = nn.Linear(d_embd, 3*d_embd)
        self.out_proj = nn.Linear(d_embd, d_embd)
 
    def forward(self, x_BTE):
        qkv = self.attn_proj(x_BTE).split(x_BTE.size(-1), -1)
        split_attn_head = lambda z: z.unflatten(-1, [-1, self.d_head]).transpose(1, 2)
        q_BHTD, k_BHTD, v_BHTD = map(split_attn_head, qkv)
        o_BHTD = F.scaled_dot_product_attention(q_BHTD, k_BHTD, v_BHTD, dropout_p=0.0, is_causal=True)
        o_BTE = o_BHTD.transpose(1, 2).flatten(-2)
        y_BTE = self.out_proj(o_BTE)
        return y_BTE

class GPTBlock(nn.Module):
    def __init__(self, d_embd, **kwargs):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_embd)
        self.attn = CausalSelfAttention(d_embd, **kwargs)
        self.ffn_norm = nn.LayerNorm(d_embd)
        self.ffn = nn.Sequential(
            nn.Linear(d_embd, 4*d_embd),
            nn.GELU(),
            nn.Linear(4*d_embd, d_embd)
        )

    def forward(self, x_BTE):
        x_BTE = x_BTE + self.attn(self.attn_norm(x_BTE))
        y_BTE = x_BTE + self.ffn(self.ffn_norm(x_BTE))
        return y_BTE

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, d_embd, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.pos_embd = nn.Embedding(max_seq_len, d_embd)
        self.tsfmr_blks = nn.ModuleList(GPTBlock(d_embd, **kwargs) for _ in range(n_layers))
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
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 128,
    pt_compile: bool = False,
):
    torch.manual_seed(3985)
    torch.cuda.set_device(0)

    cfg_json = {
    "n_layers": 1,
    "n_heads": 12,
    "d_embd": 768,
    "max_seq_len": 1024,
    "vocab_size": 50304,
    }

    model = GPT(**cfg_json).to('cuda:0')
    if pt_compile:
        model = torch.compile(model)

    dataset = SimulatedDataset(cfg_json["vocab_size"], cfg_json["max_seq_len"], bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz, num_workers=n_workers, pin_memory=True, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)

    model.train()

    for step_idx, data_batch in enumerate(data_loader):
        input_BT, label_BT = map(lambda t: t.pin_memory().to("cuda:0"), data_batch)

        with torch.amp.autocast('cuda', torch.bfloat16):
            logits_BTV = model(input_BT)
            loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        print(f"finish {step_idx} step")



class SimulatedDataset(Dataset):
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


if __name__ == "__main__":
    import fire
    fire.Fire(train)