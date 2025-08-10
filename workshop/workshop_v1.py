"""
Simplest version of Transformers
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

B = 8 # batch size
L = 16 # max sequence length
E = 64 # embedding dimension
H = 4 # attention head count

class CausalSelfAttention(nn.Module):
    """
    Input -> q, k, v projection -> attention -> cross-head projection
    """
    def __init__(self):
        super().__init__()
        # 3 is for q, k and v, its actually 3 * H * (E / H), for multi-head impl
        # another way is to write projects head by head
        self.qkv_proj = nn.Linear(E, 3 * E, bias=False)
        # output projection
        self.proj = nn.Linear(E, E, bias=False) # its actually H * (E / H)
        # mask, registered bugger var will not attend training but it will go to GPU mem.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1, 1, L, L, dtype=bool)), # max seq len
            persistent=False # not be saved in state_dict in checkpoint
            ) # pre-make mask

    def forward(self, x):
        B, T, E = x.size() # T could be less than L the max seq length, overting B and E just in case they are changed
        # x: [B, T, E]
        assert E % H == 0
        # q, k, v projection
        x = self.qkv_proj(x) # [B, T, E] @ [E, 3E] -> [B, T, 3E] = [B, T, 3 * H * (E / H)]
        # q, k, v split
        q, k, v = x.split(E, dim=2) # [B, T, 3E] -> 3 of [B, T, E]
        # q, k, v reshape for multi-head groups, [B, T, E] -> [B, H, T, E/H]
        q = q.view(B, T, H, E // H).transpose(1, 2) # [B, T, H, E/H] -> [B, H, T, E/H]
        k = k.view(B, T, H, E // H).transpose(1, 2)
        v = v.view(B, T, H, E // H).transpose(1, 2)
        
        # attention formula : attn_matrix = softmax(Q * Kt / sqrt(E//H)) * V 
        # attention matrix: q * kt -> [B, H, Tq, Tk]
        attn = (q @ k.transpose(2, 3) / math.sqrt(E // H)) # [B, H, T, T]
        # cut pre-made tril mask from [L, L] into [T, T], fill -inf before softmax, when softmax it will approach 0
        attn = attn.masked_fill(~self.mask[:, :, :T, :T], float('-inf')) # [B, H, T, T]
        attn = F.softmax(attn, dim=-1) # softmax on k's dimension, so every query get its attention on all the keys
        # above we got the "weights", then update V
        attn = attn @ v # [B, H, T, T] * [B, H, T, E/H] -> [B, H, T, E/H]
        # back to [B, T, E]
        attn = attn.transpose(1, 2).contiguous() # [B, T, H, E/H]
        attn = attn.view(B, T, E) # [B, T, H, E/H] -> [B, T, E]
        # projection for cross-head mixing: [B, T, E]
        attn = self.proj(attn)
        return attn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # [B, T, E] -> 
        self.net = nn.Sequential(
            nn.Linear(E, 4 * E, bias=False),
            nn.GELU(),
            nn.Linear(4 * E, E, bias=False),
        )

    def forward(self, x):
        return self.net(x)

class Transformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CausalSelfAttention()
        self.mlp = MLP()
    
    def forward(self, x):
        x = x + self.attention(x) # residual connections
        x = x + self.mlp(x) # residual connections
        return x

if __name__ == "__main__":
    # B-> batch size
    # L -> seq length
    # E -> vocab embedding dimension
    batched_input = torch.randn(B, L, E)
    trans = Transformers()
    output = trans(batched_input)
    print(output.shape)