"""从「远程运行结果/lab1.ipynb」里读出 20 个评估点，画训练曲线（报告用图）。

判据：一张图两条曲线（训练/验证），标题写条件，图例带末值，信息框写全部配置。
跑法：ml 环境 python report/make_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "远程运行结果" / "lab1.ipynb"
OUT = ROOT / "report" / "figures" / "fig4_training_curve.png"

# ---- 中文字体（本机有的优先；找不到就退到英文，别画成方框）----
for name in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False


def flat(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(flat(x) for x in v)
    return str(v)


nb = json.loads(NB.read_text())
rows = []
for cell in nb["cells"]:
    for o in cell.get("outputs", []):
        txt = flat(o.get("text", ""))
        for line in txt.splitlines():
            m = re.match(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)", line.strip())
            if m:
                rows.append(tuple(float(x) for x in m.groups()))

assert len(rows) == 20, f"期望 20 个评估点，实际 {len(rows)}"
steps = [r[0] for r in rows]
train = [r[1] for r in rows]
val = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(6.3, 3.9), dpi=220)
ax.plot(steps, train, color="#1f77b4", lw=1.8, marker="o", ms=4,
        label=f"训练集  (末值 {train[-1]:.4f})")
ax.plot(steps, val, color="#d62728", lw=1.8, marker="s", ms=3.6, ls="--",
        label=f"验证集  (末值 {val[-1]:.4f})")
ax.set_title("SkyPile 中文子集训练曲线（bs=10、10,000 步 ≈ 1.68 epoch）", fontsize=11.5, pad=9)
ax.set_xlabel("训练步数 (iter)", fontsize=9.5)
ax.set_ylabel("交叉熵损失 (nats)", fontsize=9.5)
ax.set_xlim(-200, 9800)
ax.set_ylim(3.0, 13.6)
ax.grid(alpha=0.3, ls=":")
ax.tick_params(labelsize=9)
ax.legend(loc="center right", fontsize=9, framealpha=0.95)

# 初始 loss 的理论值作参考线（基准线，不算曲线）
ax.axhline(11.5157, color="gray", lw=1.0, ls=":", zorder=0)

info = (
    "语料 SkyPile 中文子集：train 15,237,971 / val 1,686,758 token；"
    "step 0 的 loss 11.5982 ≈ ln(100277)=11.5157\n"
    "模型 6 层 / 6 头 / 384 维 / block 256，参数量 49,243,008（词嵌入与输出层共享权重）\n"
    "AdamW lr 1e-3→1e-4（cosine 衰减，warmup 100 步），weight decay 0.1，梯度裁剪 1.0\n"
    "dropout 0.2，seed 1228；单卡 RTX 3080 10 GB；每 500 步在 200 个 batch 上评估"
)
ax.text(0.012, 0.985, info, transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.7, alpha=0.95))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("已存:", OUT)
print("评估点数:", len(rows), "| train 末值 %.4f | val 末值 %.4f | 最低 val %.4f" %
      (train[-1], val[-1], min(val)))
