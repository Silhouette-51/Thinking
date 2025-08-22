# Minimal Mechanism Diagram — Decoder-only Transformer (Cognitive-first)

Legend
- Shapes: [B=batch, T=seq_len, d=model_dim, h=heads, dh=d/h, V=vocab]
- Dist: distribution notes; "O(1)" ≈ 不随维度增长
- Mask: causal mask 在 softmax 之前把未来位置设为 -inf（权重与梯度=0）
- Scale: 1/√dh 使打分方差保持 O(1)，避免 softmax 饱和

Flow (one block; repeat N layers)
[Tokens] idx [B,T]
   │ lookup
   ▼
[Embedding] E[idx] [B,T,d]
   │ add
   │ + [Positional] P[T,d]
   ▼
x0 = [B,T,d]
   │ (Pre-LN) LN(x0) → 近似零均值/单位方差
   ├─ Attention path:
   │     Q = x0 W_Q → [B,T,d]→reshape→[B,h,T,dh]
   │     K = x0 W_K → [B,h,T,dh]
   │     V = x0 W_V → [B,h,T,dh]
   │     scores = (Q @ K^T)/√dh + Mask  → [B,h,T,T]
   │     A = softmax(scores, dim=-1)     → [B,h,T,T]
   │     H = A @ V                       → [B,h,T,dh] → concat → [B,T,d]
   │     x1 = x0 + H     (residual)
   │
   └─ MLP path:
         y  = LN(x1)
         z  = MLP(y) = GeLU(y W1 + b1) W2 + b2   → [B,T,d]
         x2 = x1 + z   (residual)
         └─ output of this block → 下层或读出层

Readout (after last block)
 logits = x2 W_out + b_out  → [B,T,V]
 loss   = CrossEntropy(logits, targets)  (自回归：targets 为右移一位的 idx)

Invariants (self-check)
1) Shapes: d = h×dh；Q/K/V 形状 [B,h,T,dh]；scores [B,h,T,T]
2) Mask timing: Mask 在 softmax 之前 → 上三角注意力权重与梯度为 0
3) Scale: scores 需除以 √dh，使分数 std ≈ O(1)
4) Residuals: 子层均以 x + f(·) 回到维度 d
5) Pre-LN: 进入注意力/MLP 前经 LN，输出近似零均值/单位方差

Notes
- 生成时按时间步自回归，因果掩码确保"只能看见过去"
- 多头作用：并行在不同子空间建立相关性，保持每头的 dh 较小，有利于稳定