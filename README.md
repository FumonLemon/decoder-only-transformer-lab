# Decoder-only Transformer 语言模型（从零实现）

哈工大《自然语言处理》（车万翔）实验一：**不使用 `nn.Transformer` 等现成模块，从零手写一个 decoder-only Transformer（GPT）并用中文语料训练它**。

模型规格：decoder-only、6 层、6 头、`n_embd=384`、`block_size=256`、词表 100277（`cl100k_base`），权重共享（`transformer.wte.weight` 与 `lm_head.weight` 同一张量），总参数 **49,243,008**。

训练数据：[SkyPile](https://huggingface.co/datasets/SkyPile) 中文子集 `2021-43_zh_head_0000-0.01.jsonl`（课程提供），切成 **15,237,971 / 1,686,758** 个 token 的训练/验证集。

## 结果一览

在单卡 RTX 3080（10 GB 显存，恒源云按量实例）上训练 **10,000 步**（batch 10 × 256 token × 10000 步 = 25.6 M token，约 1.68 epoch），验证集 loss 从 **11.60 → 3.65**，全流程 38 个 notebook 单元零报错。

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 11.5982 | 11.5975 |
| 500 | 5.6033 | 5.6089 |
| 1000 | 5.0618 | 5.0902 |
| 2000 | 4.5978 | 4.6189 |
| 3000 | 4.2732 | 4.3121 |
| 5000 | 3.9015 | 3.9656 |
| 7500 | 3.6780 | 3.7539 |
| 9500 | 3.5654 | **3.6549** |

- `step 0` 的 loss ≈ 11.59，与随机初始化下的理论值 `ln(100277) = 11.5157` 吻合，说明前向与损失函数正确。
- train / val 始终相差 0.09 以内，没有过拟合；单批次过拟合自检（batch=2，100 步）能到 0.0117，说明反向传播与优化器通路正常。
- 生成样例（prompt `我们这堂课要学习`，temperature 0.8、top_k 200）：

  > 我们这堂课要学习惯，我们党很显准，我们就是世界下去的中央颁契机，是一个走出的民族中共……

  中文词法像样、局部搭配合理，但全局不连贯——这是 49 M 参数、1.68 epoch 的合理水平。

## 目录结构

```
.
├── lab1/
│   ├── lab1.ipynb          # 填空实现（源码版；官方原始超参 batch_size=20）
│   ├── requirements.txt
│   ├── remote_run.sh       # 云 GPU 一键脚本：装依赖 → 备语料 → nbconvert 跑完整本 → 存瘦身检查点
│   ├── data/
│   │   ├── prepare.py      # jsonl → uint32 memmap，切分 train.bin / val.bin
│   │   └── *.jsonl / *.bin # 语料与中间产物（见 .gitignore，不入库）
│   └── img/                # 讲义插图
├── 远程运行结果/
│   └── lab1.ipynb          # 云上 Run All 的成品：带训练日志、TensorBoard、生成样例（交作业用这份）
├── report/                 # 实验报告
│   ├── 实验报告-模板.docx      # 老师给的空白模板
│   ├── build_report.py        # 按模板生成报告（克隆模板自带段落样式，不新建样式）
│   ├── make_figures.py        # 从 notebook 输出里取 20 个评估点画训练曲线
│   └── figures/               # 报告用图（训练曲线 + 三个模块自检图）
├── 实验要求.md              # 课程要求整理（评分点、提交物、FAQ）
└── .vscode/settings.json   # 解释器指向 conda 环境 nlplab
```

## 在新设备上开工

```bash
git clone git@github.com:FumonLemon/decoder-only-transformer-lab.git
cd decoder-only-transformer-lab
python -m pip install -r lab1/requirements.txt
```

仓库里没有三样大文件（见 `.gitignore`，都是可再生的或另行同步的）：

| 缺的东西 | 补法 |
|---|---|
| `lab1/data/2021-43_zh_head_0000-0.01.jsonl`（39 MB） | 从课程材料下载，放进 `lab1/data/` |
| `lab1/data/train.bin`、`val.bin`（58 MB / 6.4 MB） | `cd lab1 && python data/prepare.py`（约 2.6 s） |
| `out/ckpt.pt`（565 MB 完整版 / 199 MB 瘦身版） | 从云盘或作业提交包取；只做代码改动可以不要 |

## 环境与复现

环境：Python 3.10 + PyTorch（`lab1/requirements.txt`，本机 conda 环境名 `nlplab`）。

```bash
# 1) 装依赖
python -m pip install -r lab1/requirements.txt

# 2) 备语料（约 2.6 s：jsonl -> uint32 memmap，train/val 切分）
cd lab1 && python data/prepare.py

# 3) 逐格运行 lab1.ipynb（7 个填空格都有 assert 自检 + 四联图/三联图可视化）
#    训练格默认跑满 10000 步，随后是 TensorBoard 与文本生成
```

想跑得更省事（例如在云 GPU 上无人值守）：

```bash
cd lab1 && bash remote_run.sh      # 产出 lab1_out.ipynb + out/ckpt.pt
```

## 关于显存：官方 `batch_size=20` 在 10 GB 卡上跑不动

这是本项目唯一一处偏离官方超参的地方，记录在此。

预热探针（前向 + 反向 + `optimizer.step()` 各一次，固定形状）实测峰值显存：

| batch_size | 20 | 10 | 8 |
|---|---:|---:|---:|
| 峰值显存 (GiB) | 8.98 | 4.96 | 3.72 |

在可用显存 9.65 GiB 的 3080 上按官方配置实跑，第一步反向传播即 OOM：

```
OutOfMemoryError: Tried to allocate 1.91 GiB. GPU 0 has a total capacity of 9.65 GiB
of which 1.40 GiB is free. Process has 8.24 GiB memory in use. Of the allocated memory
7.07 GiB is allocated by PyTorch, and 939.96 MiB is reserved by PyTorch but unallocated.
```

前向占 7.07 GiB，反向还需 1.91 GiB（logits 梯度 `[20,256,100277]`，约 2.05 GB），而只剩 1.40 GiB；另有 939.96 MiB 是评估阶段（`estimate_loss` 在 iter 0 先跑 400 个 batch）留下、分配器无法复用的碎片。

按 **"总训练 token 数不变"** 的原则等比例换配置：

| 超参 | 官方 | 本项目 | 理由 |
|---|---|---|---|
| `batch_size` | 20 | **10** | 峰值显存 8.98 → 4.96 GiB |
| `max_iters` | 5000 | **10000** | 10×10000×256 = 20×5000×256 = 25.6 M token |
| `lr_decay_iters` | 5000 | **10000** | cosine 衰减形状与该步数对齐 |
| `eval_interval` | 250 | **500** | 评估次数仍为 20 次，评估开销不翻倍 |

训练总计算量不变，学习率调度形状不变；实测验证集 loss 最终 3.6549，与课程给出的"约 3.5"相差 0.15。

## 检查点结构

`out/ckpt.pt`（565 MiB，未入库）是一个 dict：

```python
{'model': ..., 'optimizer': ..., 'model_args': {...}, 'iter_num': 9500, 'best_val_loss': 3.6549}
```

- 权重共享校验：`state_dict["transformer.wte.weight"] is state_dict["lm_head.weight"]` → `True`（同一存储）
- `state_dict` 各张量 `numel()` 求和为 88,142,592，比去重后的 49,243,008 多出的部分是：共享张量被计两遍（38,506,368×2）加上 6 个 256×256 因果掩码 buffer（6×65,536 = 393,216）
- 推理格只读 `model` 与 `model_args` 两个键，因此"只保留模型权重"的瘦身版（约 199 MB）也能直接跑生成

## 备注

- 本机（Apple M3，无 NVIDIA 卡）CPU 实测约 3.5 s/步，跑满 5000 步约 5.8 小时，因此正式训练放在云 GPU 上；本机只用 50 步小配置做过端到端冒烟（`nbconvert --execute` 整本零报错）。
- 训练进度在 JupyterLab 里看 tqdm 进度条与 `iter N: loss` / `step N: train loss ..., val loss ...` 两类日志；命令行 `nbconvert` 方式不会把单元输出实时打到终端。
- 参考：课程 wiki（实验要求原文）、[nanoGPT](https://github.com/karpathy/nanoGPT)（本实验框架的风格来源）、SkyPile 数据集。
