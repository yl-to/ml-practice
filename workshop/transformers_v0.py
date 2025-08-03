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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # [B, T, E] -> 
        pass

    def forward(self, x):
        return 1 # place holder

class Transformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CausalSelfAttention()
        self.mlp = MLP()
    
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

if __name__ == "__main__":
    # B-> batch size
    # L -> seq length
    # E -> vocab embedding dimension
    batched_input = torch.randn(B, L, E)
    trans = Transformers()
    output = trans(batched_input)
    print(output.shape)