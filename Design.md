# 基于EEG-文本对齐的情绪识别
## 双编码器对比学习框架 · SEED-VII 数据集

---

## 一、数据集：SEED-VII — 特征结构

```
数据来源    : 20个 .mat 文件，每个被试一个（subject_1.mat ~ subject_20.mat）
试次数量    : 每个被试 80 个试次（de_1 ~ de_80），对应 80 段电影片段
特征类型    : 差分熵（Differential Entropy, DE），4秒非重叠滑窗
数据维度    : de_i.shape = (T, 5, 62)
                T  : 试次 i 中 4 秒窗口的数量（范围：13 ~ 87）
                5  : 频率子带 [delta, theta, alpha, beta, gamma]
                62 : EEG 电极通道（10-20 国际标准系统）
```

`de_i` 中的每一行为一个独立样本，即一个完整的 4 秒 DE 特征向量。
无需进行额外的滑窗或分帧处理。

### 数据规模

| 范围                         | 数量                        |
|------------------------------|-----------------------------|
| 被试数量                     | 20                          |
| 每个被试的试次数             | 80                          |
| 每个被试的窗口总数（∑T）     | ~3,487                      |
| 全部被试的窗口总数           | ~69,740                     |
| LOSO 每折训练集窗口数        | ~66,250（19个被试）         |
| 最短试次长度                 | 13 个窗口（de_51）          |
| 最长试次长度                 | 87 个窗口（de_59）          |

---

## 二、输入格式

```
原始样本（numpy）: (5, 62)    — 频带 × 通道
模型输入         : (62, 5)    — 通道 × 频带
                    ^   ^
                    |   └── 5 个频率子带（频谱维度）
                    └─────  62 个电极通道（空间维度）
```

> 送入 CNN 前，在最前面插入一个单一维度：
> `(62, 5)` → `unsqueeze(0)` → `(1, 62, 5)` → 批量输入：`(B, 1, 62, 5)`

---

## 三、整体架构

```
[单个 4 秒 EEG 窗口]                      [文本描述]
输入: (B, 62, 5)                           三级文本语料库
       |                                          |
       v                                          v
+-------------------------+        +---------------------------+
|      EEG 编码器          |        |    文本编码器（LLM）        |
|   EEGNet-DE 变体         |        |  （CLIP ViT-L/14，冻结）   |
|                         |        |                           |
|  B1. 频谱卷积            |        |  L1: 情绪原型句              |
|  B2. 空间深度可分离卷积   |        |  L2: 片段情感描述            |
|  B3. 可分离频谱卷积       |        |  L3: 试次完整叙述            |
|  B4. 频谱多头注意力       |        |                           |
+-----------+-------------+        +-------------+-------------+
            |                                    |
            v                                    v
      EEG 投影头                           文本投影头
     （可训练）                            （可训练）
            |                                    |
            +------------------+-----------------+
                               |
                    共享潜在空间（D=128）
                               |
                    非对称对比损失
                    EEG → 文本方向为主（alpha=0.75）
```

---

## 四、EEG 编码器：EEGNet-DE 变体

### 4.1 设计动机

将 EEGNet 的核心结构（频谱卷积 → 空间深度可分离卷积 → 可分离卷积）重新适配到 DE 特征格式 `(62, 5)` 中。

| EEGNet 原始维度         | 本设计中的对应含义         | 说明                                        |
|------------------------|--------------------------|---------------------------------------------|
| 时间维度（采样点）       | **频谱维度**（5 个子带）   | DE 并非时间序列，而是按频带计算的熵值向量       |
| 空间维度（EEG 通道）     | **空间维度**（62 通道）    | 电极空间概念保持一致，结构不变                 |
| 时间卷积（跨时间点）     | **频谱卷积**（跨频带）     | 捕获相邻频率子带之间的局部模式                 |
| 空间深度可分离卷积       | **空间深度可分离卷积**     | 对 62 通道进行空间滤波，结构完全相同            |
| 可分离卷积（时间汇聚）   | **可分离频谱卷积**         | 对频率子带进行综合汇聚                         |
| —                      | **频谱多头注意力**（新增）  | 捕获 5 个子带间的全局交互（如 alpha-beta 耦合）|

---

### 4.2 模块流程图

```
输入
(B, 62, 5)
    |
    v  unsqueeze(dim=1)
(B, 1, 62, 5)
    |
    +----------------------------------------------------------+
    |  Block 1：频谱卷积（Spectral Convolution）                |
    |    Conv2d(1, F1, kernel=(1,3), padding=(0,1))            |
    |    BatchNorm2d(F1) → GELU                                |
    |    输出: (B, F1, 62, 5)                                   |
    +----------------------------------------------------------+
    |
    +----------------------------------------------------------+
    |  Block 2：空间深度可分离卷积（Spatial DepthwiseConv）      |
    |    Conv2d(F1, F1×D, kernel=(62,1),                       |
    |           groups=F1, bias=False)                         |
    |    BatchNorm2d(F1×D) → GELU                              |
    |    Dropout2d(p=0.25)                                     |
    |    输出: (B, F1×D, 1, 5)                                  |
    +----------------------------------------------------------+
    |
    +----------------------------------------------------------+
    |  Block 3：可分离频谱卷积（Separable Spectral Conv）        |
    |    [深度卷积 Depthwise]                                   |
    |    Conv2d(F1×D, F1×D, kernel=(1,3),                      |
    |           groups=F1×D, padding=(0,1), bias=False)        |
    |    [逐点卷积 Pointwise]                                   |
    |    Conv2d(F1×D, F2, kernel=1, bias=False)                |
    |    BatchNorm2d(F2) → GELU                                |
    |    Dropout(p=0.25)                                       |
    |    输出: (B, F2, 1, 5)                                    |
    +----------------------------------------------------------+
    |
    |  squeeze(dim=2) → (B, F2, 5)
    |  permute(0,2,1) → (B, 5, F2)   [将频带视为序列]
    |
    +----------------------------------------------------------+
    |  Block 4：频谱多头自注意力（Spectral Self-Attention）      |
    |    LayerNorm(F2)                                         |
    |    MultiHeadAttention(                                   |
    |      d_model=F2, heads=4,                               |
    |      FFN_dim=F2×2, dropout=0.1)                         |
    |    残差连接（Residual Connection）                         |
    |    在频带维度上做全局平均池化（Global AvgPool）            |
    |    输出: (B, F2)                                          |
    +----------------------------------------------------------+
    |
    +----------------------------------------------------------+
    |  嵌入 MLP（Embedding MLP）                                |
    |    Linear(F2 → 128) → GELU                               |
    |    LayerNorm(128)                                        |
    |    Linear(128 → 128)                                     |
    |    输出: (B, 128)                                         |
    +----------------------------------------------------------+
    |
    +----------------------------------------------------------+
    |  EEG 投影头（EEG Projection Head）                        |
    |    Linear(128 → 128) → GELU                              |
    |    LayerNorm(128)                                        |
    |    Linear(128 → 128)                                     |
    |    L2 归一化                                              |
    |    输出: (B, 128)                                         |
    +----------------------------------------------------------+
```

---

### 4.3 超参数设置

| 符号       | 取值 | 说明                                    |
|------------|------|-----------------------------------------|
| `F1`       | 16   | 频谱卷积滤波器数量（Block 1 输出通道数）  |
| `D`        | 2    | 空间深度可分离卷积的深度乘数              |
| `F1×D`     | 32   | 空间深度可分离卷积的输出通道数            |
| `F2`       | 32   | 可分离频谱卷积的输出通道数                |
| `heads`    | 4    | 频谱多头注意力的头数                      |
| `FFN_dim`  | 64   | 频谱多头注意力中前馈网络维度（`F2×2`）    |
| `drop_conv`| 0.25 | 空间卷积 / 可分离卷积后的 Dropout 比率    |
| `drop_attn`| 0.10 | 频谱多头注意力内部的 Dropout 比率         |

---

### 4.4 参数规模估算

| 模块                                         | 参数量（估算） |
|----------------------------------------------|---------------|
| Block 1 — 频谱卷积 + BN                       | ~530          |
| Block 2 — 空间深度可分离卷积 + BN             | ~2,100        |
| Block 3 — 可分离频谱卷积 + BN                 | ~1,350        |
| Block 4 — 频谱多头注意力（d=32, h=4, FFN=64） | ~6,400        |
| 嵌入 MLP（32→128→128）                        | ~20,600       |
| EEG 投影头（128→128→128）                     | ~33,000       |
| **EEG 编码器 + 投影头 合计**                  | **~64,000（约 0.064M）** |

> 相较于原始 Spatial-Spectral CNN（~0.5M），参数量缩减至约 **1/8**。
> 在 70k 样本、20 被试 LOSO 的实验规模下，过拟合风险更低，收敛更快。
> 若表达能力不足，可将参数扩展为 `F1=32, F2=64`，总量约 ~0.2M。

---

### 4.5 PyTorch 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMHA(nn.Module):
    """
    频谱多头自注意力模块：将 5 个频率子带视为一段短序列进行建模。
    输入  : (B, 5, F2)
    输出  : (B, F2)  — 在频带维度上全局平均池化后的结果
    """
    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, d_model)
        # --- 自注意力子层 ---
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + attn_out
        # --- 前馈网络子层 ---
        x = x + self.ffn(self.norm2(x))
        # --- 在频带维度上全局平均池化 ---
        return x.mean(dim=1)              # (B, d_model)


class EEGNetDE(nn.Module):
    """
    EEGNet-DE：针对差分熵（DE）特征设计的 EEGNet 变体。

    输入  : (B, 62, 5)   — 通道数 × 频带数
    输出  : (B, 128)     — L2 归一化后的嵌入向量

    架构说明：
        Block 1  — 频谱卷积：捕获相邻频带的局部模式
        Block 2  — 空间深度可分离卷积：学习每个频谱特征图的空间滤波器
        Block 3  — 可分离频谱卷积：对频带维度进行深度卷积后逐点混合通道
        Block 4  — 频谱多头自注意力：捕获跨频带的全局依赖关系
        Embed MLP — 将特征映射至嵌入空间
        Proj Head — 投影至共享对比学习空间并做 L2 归一化
    """
    def __init__(
        self,
        n_channels : int   = 62,    # EEG 通道数
        n_bands    : int   = 5,     # 频率子带数
        F1         : int   = 16,    # 频谱卷积滤波器数量
        D          : int   = 2,     # 空间深度乘数
        F2         : int   = 32,    # 可分离卷积输出通道数
        heads      : int   = 4,     # 注意力头数
        ffn_dim    : int   = 64,    # 前馈网络维度
        drop_conv  : float = 0.25,  # 卷积层后的 Dropout 比率
        drop_attn  : float = 0.10,  # 注意力层内的 Dropout 比率
        embed_dim  : int   = 128,   # 嵌入 MLP 输出维度
        proj_dim   : int   = 128,   # 投影头输出维度（对比学习空间维度）
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands    = n_bands

        # ── Block 1：频谱卷积 ───────────────────────────────────────────────
        # kernel (1, 3)：在相邻频带方向上做局部卷积
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(F1),
            nn.GELU(),
        )

        # ── Block 2：空间深度可分离卷积 ─────────────────────────────────────
        # kernel (n_channels, 1)：在每个频谱特征图上学习全通道空间滤波器
        self.block2 = nn.Sequential(
            nn.Conv2d(
                F1, F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.Dropout2d(p=drop_conv),
        )

        # ── Block 3：可分离频谱卷积 ─────────────────────────────────────────
        # 深度卷积（跨频带方向局部卷积）+ 逐点卷积（通道混合）
        self.block3 = nn.Sequential(
            # 深度卷积（Depthwise）
            nn.Conv2d(
                F1 * D, F1 * D,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=F1 * D,
                bias=False
            ),
            # 逐点卷积（Pointwise）
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.Dropout(p=drop_conv),
        )
        # Block 3 输出形状: (B, F2, 1, n_bands)

        # ── Block 4：频谱多头自注意力 ───────────────────────────────────────
        self.spectral_attn = SpectralMHA(
            d_model=F2,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=drop_attn
        )
        # 输出形状: (B, F2)

        # ── 嵌入 MLP ────────────────────────────────────────────────────────
        self.embed_mlp = nn.Sequential(
            nn.Linear(F2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── EEG 投影头 ──────────────────────────────────────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x : (B, 62, 5)  —— 批量 EEG DE 特征，通道 × 频带
        返回:
            z : (B, 128)    —— L2 归一化后的对比学习嵌入向量
        """
        # (B, 62, 5) → (B, 1, 62, 5)
        x = x.unsqueeze(1)

        # Block 1：频谱卷积  → (B, F1, 62, 5)
        x = self.block1(x)

        # Block 2：空间深度可分离卷积  → (B, F1*D, 1, 5)
        x = self.block2(x)

        # Block 3：可分离频谱卷积  → (B, F2, 1, 5)
        x = self.block3(x)

        # 重塑维度，准备输入注意力模块
        # (B, F2, 1, 5) → squeeze → (B, F2, 5) → permute → (B, 5, F2)
        x = x.squeeze(2).permute(0, 2, 1)

        # Block 4：频谱多头自注意力  → (B, F2)
        x = self.spectral_attn(x)

        # 嵌入 MLP  → (B, embed_dim=128)
        x = self.embed_mlp(x)

        # 投影头  → (B, proj_dim=128)
        x = self.proj_head(x)

        # L2 归一化
        x = F.normalize(x, p=2, dim=-1)

        return x
```

---

## 五、文本编码器：冻结 CLIP + 可训练投影头

```
骨干网络  : CLIP 文本编码器（ViT-L/14），全部参数冻结，输出维度 = 768
文本投影头 : Linear(768→256) → GELU → LayerNorm → Linear(256→128) → L2 归一化
输出维度  : (*, 128)
```

### 三级文本语料库（由 DeepSeek 生成）

| 级别 | 内容                   | 长度          | 数量                | 对齐对象         | 作用     |
|------|------------------------|---------------|---------------------|------------------|----------|
| L1   | 情绪原型句              | ≤20 词        | 每类 10 条，共 70 条 | 窗口级 EEG       | 主要损失 |
| L2   | 片段情感描述            | 30–50 词      | 每试次 1 条，共 80 条| 窗口级 EEG       | 辅助损失 |
| L3   | 试次完整叙述            | 100–150 词    | 每试次 1 条，共 80 条| 试次级 EEG       | 辅助损失 |

---

## 六、损失设计

### 6.1 软标签矩阵（跨被试假负样本缓解）

```
相同试次（视频），相同被试  → 相似度 = 1.0  （自身正样本）
相同试次（视频），不同被试  → 相似度 = 0.9  （跨被试正样本）
不同试次（视频），相同情绪  → 相似度 = 0.3  （弱正样本）
不同情绪                    → 相似度 = 0.0  （负样本）

对每一行做归一化，使其构成合法的概率分布。
```

### 6.2 窗口级非对称对比损失

```
L_win_L1 : EEG 窗口 vs. 7 个 L1 情绪原型嵌入（硬标签，7 类交叉熵）
L_win_L2 : EEG 窗口 vs. 批内 L2 片段文本嵌入（软标签矩阵）

L_win_L1 = 0.75 × CE(sim(EEG, L1_protos), emotion_label)
          + 0.25 × CE(sim(L1_protos, EEG).T, emotion_label)

L_win_L2 = 0.75 × SoftCE(sim(EEG, L2_batch), soft_labels)
          + 0.25 × SoftCE(sim(L2_batch, EEG).T, soft_labels.T)

L_window  = 0.6 × L_win_L1 + 0.4 × L_win_L2
```

### 6.3 试次级聚合损失

```
收集一个试次内所有窗口的嵌入向量: (T_i, 128)
利用 SEED-VII 连续情绪强度标签进行注意力加权池化:

  attn_score_i = Linear(128→1)(window_emb_i) + 0.3 × intensity_label_i
  trial_emb    = sum(softmax(attn_scores) × window_embs)
  trial_emb    = L2-normalize(trial_emb)                  形状: (128,)

将每条 trial_emb 与对应的唯一 L3 叙述文本嵌入进行对齐。
试次级采用 one-hot 标签（每条试次恰好对应一条 L3 文本）:

  L_trial = 0.75 × CE(sim(trial_EEG, L3_text), diag_labels)
           + 0.25 × CE(sim(L3_text, trial_EEG).T, diag_labels)

试次级批量：每个 epoch 计算一次；80 试次 × 19 被试 = 1,520 个向量。
```

### 6.4 总损失

```
阶段一（对比预训练）:
  L_total = 1.0 × L_window + 0.3 × L_trial

阶段二（分类微调）:
  L_total = 1.0 × L_cls + 0.1 × L_window
  L_cls   : 7 类情绪的标准交叉熵损失
```

---

## 七、训练方案

### 阶段一：对比预训练（~100 个 epoch）

| 配置项      | 值                                                                      |
|-------------|-------------------------------------------------------------------------|
| 冻结参数    | CLIP 文本编码器（全部参数）                                               |
| 可训练参数  | EEGNet-DE 编码器、EEG 投影头、文本投影头                                  |
| 优化器      | AdamW，lr=1e-3，weight_decay=1e-4                                        |
| 学习率调度  | 余弦退火（Cosine Annealing）+ 10 epoch 线性预热（Warmup）                 |
| 批量大小    | 256 个窗口                                                               |
| 批量规则    | 每批覆盖全部 7 种情绪类别；每条试次至少来自 ≥2 个被试的样本，以保证批内软标签正样本对的存在 |
| 温度系数    | τ = 0.07，作为可学习标量                                                  |

### 阶段二：分类微调（~30 个 epoch）

| 配置项      | 值                                                            |
|-------------|---------------------------------------------------------------|
| 冻结参数    | CLIP 文本编码器                                               |
| 新增分类头  | `Linear(128→64) → GELU → Dropout(0.3) → Linear(64→7)`       |
| 优化器      | AdamW，lr=1e-4                                                |
| 损失函数    | L_cls + 0.1 × L_window                                       |
| 评估方式    | LOSO（留一被试交叉验证，共 20 折）                             |

---

## 八、推理（测试阶段无需文本输入）

```
方案 A — 零样本推理（Zero-shot）:
  EEG 窗口 → (62, 5) → EEGNet-DE 编码器 → (128,)
  → 与 7 个预编码 L1 情绪原型嵌入（每类 10 句取均值）计算余弦相似度
  → argmax → 预测情绪类别

方案 B — 微调推理（主要方案）:
  EEG 窗口 → (62, 5) → EEGNet-DE 编码器 → (128,)
  → MLP 分类头 → 7 类 Softmax → 预测情绪类别
```

---

## 九、核心设计约束汇总

```
 1. 输入格式    : 每个样本 = de_i 的一行，形状 (5,62) → 转置为 (62,5)。
 2. 无需分窗    : DE 已为 4 秒非重叠窗口；T 行即 T 个独立样本，无需再次滑窗。
 3. 数据文件    : 20 个 .mat 文件，每个被试一个；每文件包含 de_1 ~ de_80。
 4. 数据规模    : ~69,740 个窗口总计；每个 LOSO 训练折约 66,250 个窗口。
 5. 无时间维度  : 单窗口内不存在时间轴；编码器仅做空间-频谱联合建模。
 6. 试次结构    : T 个窗口在试次内的时序关系仅用于聚合模块（注意力加权池化）。
 7. 参数规模    : ~0.064M（EEGNet-DE）；轻量化设计有效降低 ~70k 样本下的过拟合风险。
                  如表达能力不足，可扩展至 F1=32, F2=64（约 ~0.2M 参数）。
 8. 文本编码器  : 全部参数冻结；仅训练投影头（768→128）。
 9. 软标签      : 必须使用；同试次跨被试样本对为高置信度正样本。
10. 潜在空间维度: D=128；在当前数据规模下兼顾表达能力与过拟合控制。
11. EEGNet 维度映射: 原始时间轴 → 本设计中的频谱轴（5 个子带）；空间轴保持不变（62 通道）。
12. 频谱多头注意力: 补偿 EEGNet 频谱卷积的局部感受野局限，显式建模跨频带全局依赖。
```

---

## 十、设计变更对比（与原始方案）

| 对比维度           | 原始 Spatial-Spectral CNN                          | EEGNet-DE 变体                                                       |
|--------------------|----------------------------------------------------|----------------------------------------------------------------------|
| 空间处理           | 基于 10-20 布局将电极映射为 2D 网格后做 DepthwiseConv | 直接对全部 62 通道做 DepthwiseConv（无需显式 2D 网格映射）              |
| 频谱处理           | 单独的多头自注意力（作用于 5 个子带）                | 频谱卷积（局部）+ 可分离卷积 + 多头注意力（全局）—— 三层级联             |
| 中间表示           | 空间展平后为 `(B, 32, 5)`                           | 可分离卷积后为 `(B, F2=32, 1, 5)`                                     |
| 架构范式           | 定制化空间-频谱处理流水线                            | 源自 EEGNet 的卷积层级结构（在 BCI 领域具有充分的文献验证基础）          |
| 参数总量           | ~0.5M                                              | **~0.064M**（可扩展至 ~0.2M）                                         |
| 归纳偏置           | 手动构建 2D 电极网格拓扑（基于先验知识）             | 通道级深度可分离卷积（数据驱动的空间滤波器学习，无需手工设计网格结构）    |