#!/usr/bin/env bash
# 远程一键跑通：环境自检 → 装依赖 → 准备语料 → 执行整个 notebook（含存档与生成）
# 用法：在 lab1 目录下 `bash remote_run.sh`（或 `bash remote_run.sh > run.log 2>&1 &` 后台跑）
# 说明：全程不修改 notebook 里任何超参，max_iters=5000，与官方配置一致。
set -u

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
LOG="run.log"
OUT_NB="lab1_out.ipynb"

# tiktoken 词表走随包带的本机缓存，避免从 openaipublic.blob.core.windows.net 下载超时
if [ -d ../tiktoken-cache ]; then
  export TIKTOKEN_CACHE_DIR="$(cd ../tiktoken-cache && pwd)"
  echo "tiktoken 词表用本地缓存: $TIKTOKEN_CACHE_DIR"
fi

say() { printf '\n\033[1;36m===== %s =====\033[0m\n' "$*"; }

say "0. 机器与显卡"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || echo "!! nvidia-smi 不可用"
python -V
python - <<'PY'
import sys
try:
    import torch
    print("torch", torch.__version__, "| cuda 可用:", torch.cuda.is_available(),
          "| 设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("torch 还没装好:", e)
PY

say "1. 装依赖（含 requirements.txt + tensorboard，走清华镜像）"
python -m pip install -q -i "$PIP_MIRROR" -r requirements.txt
python -m pip install -q -i "$PIP_MIRROR" tensorboard
python -c "import torch, tiktoken, tensorboardX, jupyter, tensorboard; print('依赖就绪')"

say "2. 准备语料（若已有 train.bin / val.bin 则跳过）"
if [ -f data/train.bin ] && [ -f data/val.bin ]; then
  echo "已存在，跳过"
else
  python data/prepare.py | tail -3
fi
ls -lh data/*.bin

say "3. 执行整个 notebook（这一步是真正的训练，5000 步）"
START=$(date +%s)
python -m jupyter nbconvert --to notebook --execute lab1.ipynb \
  --output "$OUT_NB" --ExecutePreprocessor.timeout=14400 \
  --ExecutePreprocessor.kernel_name=python3 2>&1 | tail -40
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo "nbconvert 退出码: $RC | 耗时 $(( (END-START)/60 )) 分 $(( (END-START)%60 )) 秒"

say "4. 产物"
ls -lh out/ 2>/dev/null || echo "!! 没有 out/ 目录，训练格可能没跑完"
echo "带输出的 notebook: $OUT_NB"
echo "检查点: out/ckpt.pt"

say "4b. 生成瘦身检查点（去掉优化器状态，593MB -> 199MB，推理格照样能用）"
python - <<'PY'
import os, torch
full = "out/ckpt.pt"
if not os.path.exists(full):
    print("还没有 " + full); raise SystemExit
ck = torch.load(full, map_location="cpu")
keep = {k: ck[k] for k in ck if k != "optimizer"}
slim = "out/ckpt_slim.pt"
torch.save(keep, slim)
print("瘦身前的键:", sorted(ck.keys()))
print("瘦身后: %.1f MB -> %.1f MB" % (os.path.getsize(full) / 1e6, os.path.getsize(slim) / 1e6))
PY

say "5. 训练日志里的关键行（loss 曲线数字）"
python - <<'PY'
import json, pathlib
p = pathlib.Path("lab1_out.ipynb")
if not p.exists():
    print("还没有 " + str(p)); raise SystemExit
nb = json.loads(p.read_text())
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    txt = "".join(o.get("text", "") or "" for o in c.get("outputs", []) if o.get("output_type") == "stream")
    txt += "".join("".join(o.get("data", {}).get("text/plain", [])) for o in c.get("outputs", []))
    for line in txt.splitlines():
        if ("loss" in line and ("iter" in line or "step" in line)) or "saving checkpoint" in line:
            print(f"[cell {i}] {line.strip()}")
PY

say "完成。请把 out/ckpt.pt 与 $OUT_NB 下载回本机，然后回平台控制台【关机】"
