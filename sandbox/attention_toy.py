import torch
import torch.nn.functional as F

torch.manual_seed(0)

# Tiny 2-token, 1-head, d=2 example
x = torch.tensor([[1.0, -1.0],
                  [0.5,  0.5]])  # [T=2, d]
Wq = torch.tensor([[0.7, -0.2],
                   [0.3,  0.4]])  # [d,d]
Wk = torch.tensor([[0.6,  0.1],
                   [-0.5, 0.2]])
Wv = torch.tensor([[0.2,  0.8],
                   [0.5, -0.1]])

q = x @ Wq  # [2,2]
k = x @ Wk
v = x @ Wv

def attn(scores, v):
    w = F.softmax(scores, dim=-1)
    out = w @ v
    return w, out

def causal_mask(T):
    m = torch.zeros(T, T, dtype=torch.bool)
    m = torch.triu(m, diagonal=1)  # upper triangle True = masked
    return m

T = x.size(0)
scores = q @ k.T  # unscaled
scaled_scores = scores / (2.0 ** 0.5)  # scale by 1/sqrt(d)

# Case 1: 正确（缩放 + 掩码在 softmax 之前）
m = causal_mask(T)
masked = scaled_scores.masked_fill(m, float("-inf"))
w1, out1 = attn(masked, v)

# Case 2: 去掉缩放（仅掩码）
masked_no_scale = scores.masked_fill(m, float("-inf"))
w2, out2 = attn(masked_no_scale, v)

# Case 3: 掩码放到 softmax 之后（错误示例）
w3_bad = F.softmax(scaled_scores, dim=-1)
w3_bad = w3_bad.masked_fill(m, 0.0)
out3_bad = w3_bad @ v

print("q=\n", q)
print("k=\n", k)
print("scores=\n", scores)
print("scaled_scores=\n", scaled_scores)
print("\nCase1 weights (good)=\n", w1)
print("Case2 weights (no scale)=\n", w2)
print("Case3 weights (mask after softmax, bad)=\n", w3_bad)
print("\nOutputs:")
print("out1 (good)=\n", out1)
print("out2 (no scale)=\n", out2)
print("out3 (bad)=\n", out3_bad)