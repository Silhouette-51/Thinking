# Thinking — Transformer 认知先行 Dojo

宗旨
- 用"机制叙事 + 单因子实验 + 压缩为规则"塑造你的思维方式；数学只作为校准工具。
- 每轮产出：1 张机制图 + 3 条可复用规则 + 5 件工程交付（脚本/命令/曲线/样例/规则）。

目录
- reports/: 你的认知实战报告（自由写作，可参考模板）
- drills/: 思维操练题
- sandbox/: 最小"概念沙盒"脚本（先从注意力开始）
- diagrams/: 最小机制图

第一轮（48 小时内）
1) 画最小机制图（tokens→emb→pos→QKV→scores(+mask,scale)→weights→V聚合→logits→loss）
2) 跑 sandbox/attention_toy.py，观察"掩码/缩放"对权重与输出的影响
3) 写出你的"不变量卡片"（3–5 条）与 3 条规则；可先不跑大训练

提交
- 新建 reports/session_01.md，按你的思维反思为主（模板仅供参考）
- 我会用苏格拉底式提问点评并给出下一步挑战
