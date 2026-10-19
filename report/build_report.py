"""按老师模板填《自然语言处理》实验一报告。

原则：模板是骨架，本脚本只产生内容 —— 所有正文段落都从模板自带的
'Body Text First Indent' 段落克隆，图注用模板自带的 '图形标注' 样式，
封面表格就地填值，不新建任何样式。

跑法：ml 环境 python report/build_report.py
产出：report/自然语言处理-实验一-<学号>-<姓名>.docx（姓名等来自 report/report_meta.json，该文件不进版本库）
"""
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "report" / "实验报告-模板.docx"
META_P = ROOT / "report" / "report_meta.json"
if not META_P.exists():
    raise SystemExit(f"缺少 {META_P}：请在本地建一个 json 写明姓名/学号/日期（该文件已 .gitignore）")
META = json.loads(META_P.read_text())
OUT = ROOT / "report" / f"自然语言处理-实验一-{META['学号']}-{META['姓名']}.docx"
FIG = ROOT / "report" / "figures"

# ─────────────────────────── 从 notebook 里取真实数字 ───────────────────────────
import re

nb = json.loads((ROOT / "远程运行结果" / "lab1.ipynb").read_text())


def flat(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(flat(x) for x in v)
    return str(v)


LOSS = []
GEN = ""
for cell in nb["cells"]:
    for o in cell.get("outputs", []):
        txt = flat(o.get("text", ""))
        for line in txt.splitlines():
            m = re.match(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)$", line.strip())
            if m:
                LOSS.append(tuple(m.group(1) if i == 0 else float(m.group(i + 1)) for i in range(3)))
        if "我们这堂课要学习" in txt:
            GEN = txt.strip()

assert len(LOSS) == 20, f"评估点数应为 20，实得 {len(LOSS)}"
assert GEN, "没从 notebook 输出里找到生成样例"
GEN_SAMPLES = [s.strip() for s in GEN.split("-" * 15) if s.strip()]
assert len(GEN_SAMPLES) >= 4, GEN_SAMPLES


# ─────────────────────────── 内容块 ───────────────────────────
# ("h", 标题)  ("p", 正文)  ("bullet", 条目)  ("fig", (文件名, 宽度英寸, 图注))
# ("table_loss", None)  ("quote", 多行文本——等宽小字块)  ("table_env", None)
BLOCKS_1 = [
    ("h", "（一）实验任务与整体流程"),
    ("p", "本实验要求不使用 nn.Transformer、nn.MultiheadAttention 等现成模块，用 PyTorch 从零实现一个 "
          "decoder-only Transformer（GPT）语言模型，并用 SkyPile 中文预训练数据集的子集训练它，最终能根据"
          "提示词续写中文文本。需要实现七个部分：Token Embedding、Positional Encoding、简易 Attention Mask、"
          "多头自注意力、FFN 层、Decoder Block，以及包含前向与损失函数的完整 GPT 模型；此外还要完成语料"
          "预处理、训练与推理三段流程。"),
    ("p", "整体流程是一条直线：jsonl 语料 → prepare.py 转成 uint32 二进制 memmap 并切分训练/验证集 → "
          "按 block_size=256 随机采样 batch → 前向（词嵌入 + 位置编码 + 6 个 Decoder Block + 末层 "
          "LayerNorm + 输出头）→ 交叉熵损失 → 反向传播与 AdamW 更新 → 每 500 步在 200 个 batch 上评估并"
          "保存最优检查点 → 加载检查点做采样生成。训练与生成共用同一套前向，只在最后一个位置取 logits 生成"
          "下一个 token，这一点是理解整个模型的关键。"),
    ("h", "（二）实验环境与数据"),
    ("p", "本机（Apple M3，无 NVIDIA 显卡）只用于代码自检与 50 步端到端冒烟；正式训练放在按量租用的单卡"
          "云主机上。两边环境如下表。"),
    ("table_env", None),
    ("p", "语料为课程提供的 SkyPile 中文子集 2021-43_zh_head_0000-0.01.jsonl（41 MB）。prepare.py 把它转成 "
          "numpy uint32 数组后按 90%/10% 切分，得到训练集 15,237,971 个 token、验证集 1,686,758 个 token，"
          "本机耗时 2.6 秒，产出 train.bin（58 MB）与 val.bin（6.4 MB）。词表用 tiktoken 的 cl100k_base，"
          "V=100277，因此随机初始化时理论损失为 ln(100277)=11.5157，这条基准线后面用来判断前向是否正确。"),
    ("h", "（三）七个模块的实现与自检"),
    ("p", "下面按数据流的顺序给出每个模块做了什么、怎么实现的、以及用什么判据确认它写对了。所有判据都是"
          "真跑出来的数字，不是估计。"),
    ("bullet", "1. Token Embedding。查表：output = E[idx]，其中 E∈R^{V×C}；等价于对每个 token 做 one-hot 后"
               "乘 E。实现只有一行 nn.Embedding(vocab_size, n_embd)。自检：取 V=4、C=5，输入 [0,1] 得到 (2,5) "
               "的输出，其第 0/1 行与权重矩阵的对应行逐位相等，且与 one-hot 矩阵相乘的结果一致（均为 True）。"),
    ("bullet", "2. Positional Encoding。同样是一张可学习的查表 wpe，但表的大小是 block_size=256 而不是序列"
               "长度：位置 id 由 arange(T) 给出，取前 T 行与词向量逐元素相加（不是拼接）。自检：wpe.weight.shape "
               "= (256,5)；同一 token 出现在两个位置时最终向量不同（True），且输出恰等于词向量加位置向量（True）。"),
    ("bullet", "3. 简易 Attention Mask。用 torch.tril(ones(T,T)) 造下三角，再把 mask==0 的位置填 -inf，softmax "
               "之后未来位置的权重才是严格的 0。自检：8 个位置时上三角（未来位置）的最大权重为 0.0e+00，第 0 行"
               "恰为 [1,0,…,0] 且每行和为 1。反向对照：如果把 -inf 误写成 0，形状、行和都不报错，但第 0 行变成 "
               "[0.3371, 0.0947, …]，即位置 0 把 33.7% 的权重分给了它不该看到的未来 token —— 这类错误只能靠"
               "肉眼检查，所以检查点必须写成“第 0 行必须是 [1,0,…]”。"),
    ("bullet", "4. 多头自注意力。一个线性层把 C 维投成 3C 得 q/k/v（bias=False），view(B,T,3C) 拆开后 "
               "view(B,T,H,C/H).transpose(1,2) 把头维提到前面成 (B,H,T,C/H)；注意力分数 "
               "att = q·kᵀ/√(C/H)，用注册为 buffer 的 (1,1,block,block) 掩码 masked_fill 掉未来位置，"
               "softmax 后过 dropout，再与 v 相乘，最后 transpose+contiguous+view 合回 (B,T,C)。"
               "参数量 4C²=589,824。"),
    ("bullet", "5. FFN 层。c_fc（C→4C）+ GELU + c_proj（4C→C）的两层感知机，参数量 8C²=1,181,568。"
               "形状自检：(2,8,384)→(2,8,1536)→(2,8,384)。"),
    ("bullet", "6. Decoder Block。前置归一化 + 两处残差：x = x + 多头注意力(LN(x))，再 x = x + FFN(LN(x))。"
               "两个 LayerNorm 各 2C=768 个参数，单层合计 1,772,928（= 589,824 + 1,181,568 + 1,536）。"),
    ("bullet", "7. 完整 GPT。wte + wpe + dropout + 6 个 Decoder Block + 末层 LayerNorm + lm_head；"
               "lm_head 与 wte 共享同一张权重矩阵（只算一次参数，state_dict 里会看到两个键指向同一存储）。"
               "前向：pos=arange(T)，x=drop(wte(idx)+wpe(pos))，逐层过 Block，再过 ln_f，logits=lm_head(x)，"
               "损失 cross_entropy(logits.view(-1,V), targets.view(-1))；生成时只取 x[:, [-1], :] 的最后位置。"),
    ("p", "把所有模块的参数加起来，与程序的实测值逐位吻合，这是“模型搭对了”的第一个硬证据；"
          "词嵌入与输出层共享权重，所以它占了总参数的 78.2%。"),
    ("table_param", None),
    ("p", "另外三个判据同样是真跑的：① 权重共享校验 wte.weight is lm_head.weight 为 True；"
          "② 初始损失 11.5982 / 11.5975，与 ln(100277)=11.5157 只差 0.08（随机初始化下的应有水平）；"
          "③ 过拟合自检：只用 1 个小 batch（batch=2）反复训练 100 步，损失从 11.5855 掉到 0.0117"
          "（25 步 2.2585、50 步 0.0693、75 步 0.0185），说明前向、反向与优化器通路确实通了 —— "
          "如果梯度写错，损失不会这样干净地降到接近 0。"),
    ("h", "（四）训练：官方配置在 10 GB 卡上的显存问题"),
    ("p", "课程 FAQ 说 batch_size=20 约需 11.6 GB 显存。本实例的 nvidia-smi 报 10240 MiB，可用显存 "
          "9.65 GiB（平台页面上写的“显存 20 GB”与实测不符），因此先用一次性探针量了三档 batch 的真实峰值："
          "batch_size=20 为 8.98 GiB、10 为 4.96 GiB、8 为 3.72 GiB。按官方配置实跑时，第 0 步评估正常"
          "（train loss 11.5972 / val loss 11.5967），但紧接着的反向传播就 OOM："),
    ("quote", "OutOfMemoryError: Tried to allocate 1.91 GiB. GPU 0 has a total capacity of 9.65 GiB\n"
              "of which 1.40 GiB is free. Of the allocated memory 7.07 GiB is allocated by PyTorch,\n"
              "and 939.96 MiB is reserved by PyTorch but unallocated."),
    ("p", "拆开看：前向已分配 7.07 GiB，反向还要再拿 1.91 GiB（即 logits 梯度的 [20,256,100277] 单精度"
          "张量，约 2.05 GB），而只剩 1.40 GiB，差约 0.5 GiB；另外 939.96 MiB 是评估阶段（在 step 0 先跑"
          "400 个 batch）留下、分配器既没释放也无法复用的碎片。也就是说探针量到的 8.98 GiB 只是单步稳态，"
          "真实需求是它加上评估碎片，两者合起来超过了这张卡的容量。"),
    ("p", "解决办法不是简单地砍 batch_size，而是按“总训练 token 数不变”的原则等比例替换超参："
          "batch_size 20→10、max_iters 5000→10000、lr_decay_iters 5000→10000、eval_interval 250→500。"
          "这样 10×10000×256 与 20×5000×256 都等于 25.6 M token（约 1.68 个 epoch），cosine 衰减的形状、"
          "评估点个数（20 次）和总计算量都不变，只有“一步看多少数据”变了；代价是每步梯度噪声略大、"
          "warmup 从占 2% 变成占 1%，这两点会在末值上体现出来（见下）。"),
    ("h", "（五）训练结果"),
    ("p", "训练共 10,000 步，每 500 步在 200 个 batch 上评估一次，得到 20 个点。损失单调下降，没有反弹，"
          "训练集与验证集全程相差不超过 0.09，说明没有过拟合。"),
    ("table_loss", None),
    ("fig", ("fig4_training_curve.png", 5.7,
             "图 1  训练曲线：两条曲线分别对应训练集与验证集，横轴为训练步数，纵轴为交叉熵损失；"
             "虚线为随机初始化的理论值 ln(V)。")),
    ("p", "三个可以从图与表里直接读出的结论：① 前 500 步损失从 11.60 掉到 5.60，这一段学的是高频词与"
          "常用搭配；之后进入长尾，到 5,000 步才降到 3.97，最后 5,000 步只再降 0.31，呈现典型的"
          "“先陡后平”语言模型学习曲线。② 训练与验证损失几乎重合（末端 3.5654 vs 3.6549），"
          "说明 49 M 参数的模型在 25.6 M token 上没有记忆训练集。③ 末端验证损失 3.6549，比课程给出的"
          "“约 3.5”高 0.15，原因有三：最后一次评估发生在第 9,500 步而不是 10,000 步；warmup 在总步数"
          "翻倍后只占 1%（官方为 2%）；batch 减半后梯度噪声更大，末期的优化精度略低。"),
    ("p", "加载检查点后，以“我们这堂课要学习”为开头、temperature=0.8、top_k=200 采样，得到如下两段"
          "（共 5 段，完整输出见提交的 notebook）："),
    ("quote", "\n\n———————— 样例 1 ————————\n" + GEN_SAMPLES[0] +
              "\n\n———————— 样例 4 ————————\n" + GEN_SAMPLES[3]),
    ("p", "可以看出模型已经掌握了中文的词法与局部搭配（“我们这堂课要学习惯”“教学、教学、教学”这类"
          "搭配是语料里高频出现的），但句子之间没有全局逻辑、还会自我复读。这是 49 M 参数、只训练 1.68 个"
          "epoch 的合理水平：它在做的是“根据上文预测下一个 token”，而不是“组织一篇讲话”。"),
    ("h", "（六）程序运行效果截图"),
    ("p", "下面四组是 notebook 单元格的真实运行输出：前三组是各模块的自检可视化（用来确认实现正确），"
          "最后一组是训练过程的日志。"),
    ("fig", ("fig1_token_embedding.png", 5.0,
             "图 2  Token Embedding 自检：子图 2 为每个 token 的 one-hot 向量，子图 3 为权重矩阵，"
             "子图 4 为查表结果，恰好等于子图 3 的前两行（即“查表 ≡ one-hot 相乘”）。")),
    ("fig", ("fig2_positional.png", 5.0,
             "图 3  位置编码自检：子图 1 为词向量、子图 2 为位置向量、子图 3 为二者之和，"
             "验证了位置信息是以“相加”而非“拼接”的方式进入模型的。")),
    ("fig", ("fig3_mask.png", 5.0,
             "图 4  因果掩码自检：子图 2 为下三角掩码（上三角全暗），子图 4 为掩码后的注意力权重矩阵，"
             "第 0 行只有对角线一格亮，逐行变宽，说明每个位置只能看到自己和左侧的 token。")),
    ("quote", "num decayed parameter tensors: 26, with 49,221,504 parameters\n"
              "num non-decayed parameter tensors: 38, with 21,504 parameters\n"
              "Training: 0%|          | 0/10000 [00:00<?, ?it/s]\n"
              "step 0: train loss 11.5982, val loss 11.5975\n"
              "step 500: train loss 5.6033, val loss 5.6089\n"
              "……（每 500 步一行，共 20 行，完整内容见 notebook）\n"
              "step 9500: train loss 3.5654, val loss 3.6549\n"
              "saving checkpoint to out\n"
              "iter 9990: loss 3.6907"),
]

BLOCKS_2 = [
    ("h", "（一）收获"),
    ("p", "这次实验最大的收获不是“写出了一个 GPT”，而是把 Transformer 的数据流在形状层面走通了一遍："
          "一条序列进来是 (B,T) 的整数下标，查表后变 (B,T,C)，多头注意力内部拆成 (B,H,T,C/H) 再合回来，"
          "FFN 先把最后一维放大 4 倍再压回去，最后经输出头变成 (B,T,V) 才能与 (B,T) 的目标算交叉熵。"
          "每写一个模块我都先问“进出形状是什么”，形状对了，逻辑基本就对了；反过来，七个模块里凡是出错的"
          "地方，几乎都能在形状上找到线索。"),
    ("p", "第二个收获是理解了“权重共享”不是一句口号。state_dict 里 wte.weight 与 lm_head.weight 同存，"
          "但二者是同一块存储（data_ptr 相同），所以参数总数要把它们按一份算：把 state_dict 各张量求和会"
          "得到 88,142,592，比真实参数量 49,243,008 多出的部分，正是这块被算了两遍的 38,506,368 加上 6 个"
          "256×256 的因果掩码 buffer（6×65,536=393,216）。这个对账做通之后，我对“哪些东西是参数、"
          "哪些只是随模型搬来搬去的常量”才真正分清。"),
    ("p", "第三个收获是关于因果掩码。写 -inf 与写 0 在形状、行和、不报错这三件事上完全一样，但写 0 会让"
          "每个位置提前看到未来 token，模型在训练时“偷看答案”，推理时却没有答案可看。这个对照实验让我"
          "记住：数值上不报错 ≠ 逻辑上正确，必须用“第 0 行应当只有一格亮”这种形态判据去检查。"),
    ("h", "（二）遇到的问题与解决"),
    ("bullet", "1. 本机没有 NVIDIA 显卡，官方给的 device=\"cuda\" 会直接抛 AssertionError。改为按 "
               "torch.cuda.is_available() 自动选择设备后，本机能跑但很慢：实测 3.503 秒/步，跑满 5,000 步"
               "约 4.9 小时，加上评估接近 5.8 小时。顺便验证了“有 GPU 就一定更快”是错觉 —— 同一份代码在 "
               "Apple MPS 上是 4.122 秒/步，反而更慢；而且最初量到的 4.229/5.870 秒是错的，因为 MPS 是异步的，"
               "不调用 synchronize 时计时器只量到“把任务放进队列”的时间。"),
    ("bullet", "2. 10 GB 显存放不下官方 batch_size=20（细节见第一部分第四节）。这里的教训是不要用"
               "“单步峰值占 93%”去判断能跑：评估阶段留下的碎片、反向新增的张量都要算进去，余量不足 10% "
               "就该换配置。最终按“总 token 数不变”把 batch 减半、步数翻倍，等价性是有算式支撑的"
               "（10×10000×256 = 20×5000×256），而不是随便调小超参凑合跑。"),
    ("bullet", "3. 单元格之间存在隐式依赖。位置编码那一格要读 GPTConfig，而 GPTConfig 里的 "
               "vocab_size = enc.n_vocab 依赖 tokenizer 那一格，顺序弄错就 NameError；把某一格单独抽出来"
               "放到脚本里跑也必然失败，因为 notebook 的变量是靠内核跨单元格存活的。解决办法是记住一条"
               "固定的运行链：环境格 → tokenizer 格 → 配置格 → 嵌入格 → 后续各格，交付前重启内核再"
               "“运行所有单元格”一遍。"),
    ("bullet", "4. 工具链上的两个小坑：环境里缺 tensorboard 包时，%tensorboard 那一格会抛 "
               "ModuleNotFoundError，把“运行所有单元格”卡在推理格之前，补装 7 个依赖后正常；"
               "JupyterLab 里 tqdm 进度条刷新不及时（长时间显示 0%），可靠的进度信号是逐行打印的 "
               "iter loss 与每 500 步一次的 train/val loss。"),
    ("h", "（三）不够满意的地方"),
    ("p", "末端验证损失 3.6549 与课程给出的 3.5 还有 0.15 的差距。虽然有“最后一次评估在 9,500 步、"
          "warmup 占比变小、batch 减半噪声更大”这三条解释，但更严谨的做法是在显存放得下的前提下再跑一"
          "组 batch_size=20 的官方配置做对照（需要 20 GB 以上显存），或者把步数再延长到 2 个 epoch 以上看"
          "曲线是否还能继续下降。这两项受算力与时间限制没有做，写在这里留作后续。"),
    ("p", "另外，这次只验证了“能训练、能生成”，没有做超参扫描（学习率、层数、维度），也没有对生成质量"
          "做定量评估（如困惑度以外的指标）。作为第一次从零实现 Transformer 的实验，这些留到以后再补。"),
]


# ─────────────────────────── 填表工具 ───────────────────────────
def set_text(par_el, text, parent, style_from=None, bold=False, size=None, align=None):
    """把段落元素的文字换掉，保留它自己的 pPr 与首个 run 的 rPr。"""
    par = Paragraph(par_el, parent)
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    run = par.add_run(text)
    if bold:
        run.font.bold = True
    if size:
        run.font.size = Pt(size)
    if align is not None:
        par.alignment = align
    return par


def main():
    shutil.copy(TPL, OUT)
    doc = Document(str(OUT))
    body = doc.element.body

    # ---- 封面 ----
    cover = doc.tables[0]
    values = {"姓名：": META["姓名"], "学号": META["学号"], "日期": META["日期"]}
    for row in cover.rows:
        label = row.cells[0].text.strip()
        for key, val in values.items():
            if label.startswith(key):
                cell = row.cells[1]
                p = cell.paragraphs[0]
                for r in list(p.runs):
                    r._element.getparent().remove(r._element)
                run = p.add_run(val)
                run.font.size = Pt(16)
                # 中文字体与标签对齐
                run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")

    # ---- 标题 ----
    for par in doc.paragraphs:
        if par.text.strip().startswith("实验X"):
            set_text(par._p, "实验一：从零开始实现基于 Transformer 的语言模型", par._parent)
            break
    else:
        sys.exit("没找到标题占位段落")

    # ---- 找两个二级标题，删掉模板里的说明文字 ----
    heads = {p.text.strip(): p for p in doc.paragraphs if p.style.name == "Heading 2"}
    h1, h2 = heads["一、实验过程及结果"], heads["二、实验心得"]

    body_src = None          # 正文段落样板（克隆它的 pPr/rPr）
    cap_src = None           # 图注样板（'图形标注' 样式）
    for par in doc.paragraphs:
        if body_src is None and par.style.name == "Body Text First Indent" and par.text.strip():
            body_src = par
        if cap_src is None and par.style.name == "图形标注" and not par.text.strip():
            cap_src = par
    assert body_src is not None and cap_src is not None

    # 删掉模板的两处占位说明（每节各有 2 段：'***********。' 与指路句）
    for par in list(doc.paragraphs):
        t = par.text.strip()
        if t in ("***********。",) or t.startswith("简要描述设计思路") or t.startswith("此次实验的收获"):
            par._p.getparent().remove(par._p)

    # ---- 插内容 ----
    def insert_blocks(anchor_p, blocks):
        anchor = anchor_p._p
        for kind, payload in blocks:
            if kind in ("h", "p", "bullet"):
                new_el = deepcopy(body_src._p)
                anchor.addnext(new_el)
                if kind == "h":
                    par = set_text(new_el, payload, body_src._parent, bold=True)
                    par.paragraph_format.first_line_indent = Pt(0)
                    par.paragraph_format.space_before = Pt(8)
                    par.paragraph_format.space_after = Pt(2)
                elif kind == "bullet":
                    set_text(new_el, payload, body_src._parent)
                else:
                    set_text(new_el, payload, body_src._parent)
                anchor = new_el
            elif kind == "quote":
                new_el = deepcopy(body_src._p)
                anchor.addnext(new_el)
                par = set_text(new_el, payload, body_src._parent, size=9)
                par.paragraph_format.left_indent = Pt(18)
                par.paragraph_format.first_line_indent = Pt(0)
                par.paragraph_format.space_before = Pt(4)
                par.paragraph_format.space_after = Pt(4)
                anchor = new_el
            elif kind == "fig":
                fname, width, caption = payload
                new_el = deepcopy(body_src._p)
                anchor.addnext(new_el)
                par = Paragraph(new_el, body_src._parent)
                for r in list(par.runs):
                    r._element.getparent().remove(r._element)
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                par.add_run().add_picture(str(FIG / fname), width=Inches(width))
                cap_el = deepcopy(cap_src._p)
                new_el.addnext(cap_el)
                set_text(cap_el, caption, cap_src._parent, align=WD_ALIGN_PARAGRAPH.CENTER)
                anchor = cap_el
            elif kind == "table_env":
                rows = [
                    ("项目", "本机（自检 / 冒烟）", "云主机（正式训练）"),
                    ("硬件", "Apple M3 / 16 GB，无 NVIDIA 显卡", "RTX 3080 10 GB，24 核 / 45 GB 内存"),
                    ("Python", "3.10.21（conda 环境 nlplab）", "3.11（平台镜像）"),
                    ("PyTorch", "2.14.1（CPU）", "2.9.1（CUDA 12.8）"),
                    ("用途", "逐模块自检、50 步端到端冒烟", "10,000 步正式训练与推理"),
                ]
                anchor = add_table(anchor, rows, widths=[1.5, 3.0, 2.9],
                                   caption="表 1  实验环境与机器配置")
            elif kind == "table_param":
                rows = [
                    ("模块", "参数量", "说明"),
                    ("词嵌入 wte", "38,506,368", "100277×384，与输出层共享同一张矩阵"),
                    ("位置编码 wpe", "98,304", "256×384，可学习"),
                    ("Decoder Block ×6", "10,637,568", "单层 1,772,928 = 注意力 589,824 + FFN 1,181,568 + 两个 LN 1,536"),
                    ("末层 LayerNorm", "768", "2×384"),
                    ("合计", "49,243,008", "词嵌入占 78.2%"),
                ]
                anchor = add_table(anchor, rows, widths=[2.0, 1.9, 3.5],
                                   caption="表 2  参数量分解（与程序实测值逐位吻合）")
            elif kind == "table_loss":
                rows = [("步数", "训练集损失", "验证集损失")] + [
                    (f"{s}", f"{tr:.4f}", f"{va:.4f}") for s, tr, va in LOSS
                ]
                anchor = add_table(anchor, rows, widths=[2.0, 2.7, 2.7], font=9.5,
                                   caption="表 3  训练过程中的损失（20 个评估点，每 500 步一次）")
            else:
                raise ValueError(kind)
        return Paragraph(anchor, body_src._parent)

    def add_table(anchor_el, rows, widths=None, font=10, caption=None):
        cap_el = None
        if caption:
            cap_el = deepcopy(cap_src._p)
            anchor_el.addnext(cap_el)
            set_text(cap_el, caption, cap_src._parent, align=WD_ALIGN_PARAGRAPH.CENTER)
        n_row, n_col = len(rows), len(rows[0])
        t = doc.add_table(rows=n_row, cols=n_col)
        t.style = doc.styles["Table Grid"]
        t.autofit = True
        for ri, row in enumerate(rows):
            for ci, val in enumerate(row):
                cell = t.cell(ri, ci)
                p = cell.paragraphs[0]
                for r in list(p.runs):
                    r._element.getparent().remove(r._element)
                run = p.add_run(str(val))
                run.font.size = Pt(font)
                run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
                if ri == 0:
                    run.font.bold = True
                if ci > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        (cap_el if cap_el is not None else anchor_el).addnext(t._tbl)   # 从文末搬到锚点之后
        return t._tbl

    insert_blocks(h1, BLOCKS_1)
    insert_blocks(h2, BLOCKS_2)

    doc.save(str(OUT))
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
