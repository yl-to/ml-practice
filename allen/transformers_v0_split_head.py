import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

B = 8
L = 64
E = 128
H = 4

class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention() for _ in range(H)])
        

    def forward(self, x):
        heads_outputs = [head(x) for head in self.heads]
        heads_outputs = torch.cat(heads_outputs, dim=-1)
        return heads_outputs    

class SingleHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(E, E // H, bias=False) # e//h = d
        self.k_proj = nn.Linear(E, E // H, bias=False)
        self.v_proj = nn.Linear(E, E // H, bias=False)
        self.mask = torch.tril(torch.ones(1, L, L, dtype=torch.bool))
    
    def forward(self, x):
        q = self.q_proj(x) # b, l, d
        k = self.k_proj(x) # b, l, d
        v = self.v_proj(x) # b, l, d

        att_score = q @ k.transpose(1, 2) # b, l, d @ b, d, l -> b, l(q), l(k)
        att_score = att_score / ((E//H) ** 0.5)
        att_score = att_score.masked_fill(~self.mask, float('-inf'))
        att_score = F.softmax(att_score, dim=-1) # softmax on k's dim
        att_score = att_score @ v # b, l, l @ b, l, d -> b, l, d
        return att_score

if __name__ == "__main__":
    x = torch.randn(B, L, E)


    model = MultiheadAttention()
    optimizer = Adam(model.parameters(), lr=1e-5)
    criteria = nn.MSELoss()
    for i in range(1000):
        optimizer.zero_grad()
        output = model(x)
        loss = criteria(output, x)
        loss.backward()
        optimizer.step()
        print(loss)

    
    