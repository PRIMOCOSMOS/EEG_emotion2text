你是一位深度学习与脑机接口领域的专家，正在优化一个基于 EmotionCLIP 框架的 EEG 情感识别项目（EEG_emotion2text）。该项目使用 SEED-VII 数据集，其关键规格如下：
- 7 类情绪：happy, sad, fear, disgust, surprise, anger, neutral
- 20 名被试 × 4 sessions × 每 session 20 trials = 共 1600 trials
- 62 通道 EEG（国际 10-20 系统），采样率 1000Hz
- 预提取特征：DE（微分熵）和 PSD（功率谱密度），覆盖 delta/theta/alpha/beta/gamma₁/gamma₂ 共 6 个频段
- 输入为 4D 张量 X ∈ R^{T×F×H×W}，其中 62 通道被映射为 2D 电极拓扑网格（稀疏类图像表示），F≥6 为频段数

架构核心组件：
1. EEG 编码器（SST-LegoViT）：空间多尺度卷积 → 频谱 Legoformer → 时间 Transformer
2. 文本编码器：预训练 CLIP Text Encoder
3. 投影头：将 EEG/文本特征映射到共享嵌入空间
4. 对比损失：InfoNCE / CLIP Loss 跨模态对齐

当前核心问题：对比学习阶段过拟合。请根据以下方案逐模块实施优化：

===== EEG 编码器 — SST-LegoViT =====

【空间多尺度卷积模块】
- 输入为 62 通道映射后的稀疏 2D 电极拓扑网格（非标准图像，大部分像素为零填充），空间分辨率极低
- 引入空间通道 Dropout（p=0.15），在 H×W 网格上随机将若干电极位置置零，模拟电极脱落/接触不良
- 将标准 Conv2D 替换为 Depthwise Separable Conv 以压缩参数量（可降低约 8-9 倍）
- 精简卷积核数量：多尺度分支从 64/128 降至 32/64
- 空间增强：因输入是稀疏电极网格而非自然图像，不可使用传统图像旋转/翻转；应改为基于电极邻接关系的局部空间扰动（随机交换相邻电极特征值，概率 p=0.1）或对网格中的非零位置添加空间高斯噪声

【频谱 Legoformer 模块】
- SEED-VII 频段维度 F≥6（含 gamma₂ 51-75Hz），序列长度仍然较短，注意力头易过拟合
- 频段随机 Masking（p=0.2，每次随机屏蔽 1 个频段），迫使模型不依赖单一频段
- 注意力头数从 8 降至 2-4
- 在 softmax(QK^T/√d) 后加 attention dropout（rate=0.3-0.5）
- 对 DE/PSD 特征注入高斯噪声（σ=0.05-0.1）作为频域扰动增强
- 考虑频段间的分组注意力（grouped attention）：将相邻频段（如 alpha+beta、gamma₁+gamma₂）分组，减少全连接注意力的参数量

【时间 Transformer 模块】（参数量最大，过拟合风险最高）
- 层数从 6 降至 2-3（EEG 情感的时间动态远比 NLP 简单）
- hidden_dim 从 512 降至 128-256
- 引入 LayerDrop（p=0.1-0.2），训练时随机跳过 Transformer 层
- 时间窗口随机裁剪：训练时随机截取 70-90% 的时间步长度
- R-Drop 正则化：同一输入两次前向传播（不同 dropout mask），最小化两次输出的 KL 散度
- 梯度裁剪 max_norm=1.0
- 位置编码使用可学习的相对位置编码（而非固定正弦位置编码），更适配 EEG 的时间结构

===== 文本编码器 — CLIP Text Encoder =====

- SEED-VII 有 7 类情绪标签（而非 3 类），文本嵌入空间比 SEED 略丰富，但仍远少于自然语言场景
- 完全冻结底层，仅解冻最后 1-2 层
- 文本提示多样化（Prompt Ensemble）：设计 8+ 种模板，覆盖不同语义表达，每次前向传播随机选取
  示例模板：
  "The human feels {emotion} now"
  "This person is experiencing {emotion}"
  "Neural activity indicates {emotion} emotion"
  "The brain signal shows a {emotion} state"
  "EEG patterns suggest {emotion} feeling"
  "The participant's mood is {emotion}"
  "Current emotional state: {emotion}"
  "The subject feels {emotion} at this moment"
- 训练时对文本嵌入添加微小高斯扰动（σ=0.05）后重新 L2 归一化，防止 EEG 编码器仅学会映射到 7 个固定点
- 可考虑利用 SEED-VII 提供的连续情绪强度标签（continuous intensity labels）生成更细粒度的文本描述，如 "The human feels strongly happy" vs "The human feels slightly happy"，增加文本端的多样性

===== 投影头 =====

- proj_dim 从 512 降至 128-256
- 加入 BatchNorm + Dropout(0.2-0.3)
- 层数限制在 2-3 层
- 输出经 L2 归一化后送入对比损失

===== 对比损失 =====

- 可学习温度参数 τ，init=0.07，clamp=[0.01, 0.5]
- 对 InfoNCE 标签施加 Label Smoothing（ε=0.1），防止对完全匹配过于自信
- 联合损失：L_total = λ_clip·L_contrastive + λ_cls·L_CE + λ_center·L_center
  建议权重 λ_clip=1.0, λ_cls=0.5, λ_center=0.01
- 7 类分类下 batch 内负样本同类碰撞概率更高（~1/7≈14.3%），建议在 InfoNCE 中加入同类负样本过滤（Supervised Contrastive Loss 形式），避免将同类样本作为负样本推开

===== 训练策略 =====

- 两阶段训练：
  阶段一（对比预训练）：训练 EEG 编码器 + 投影头，lr=1e-3，强增强，冻结 CLIP 文本编码器
  阶段二（监督微调）：冻结 EEG 编码器底层（空间+频谱），仅微调时间 Transformer 最后 1-2 层 + 分类 MLP，lr=1e-5，弱增强，早停 patience=10-15
- SEED-VII 有 20 被试 × 4 sessions，采用 Leave-One-Subject-Out (LOSO) 交叉验证评估跨被试泛化
- 每个 session 有 20 trials，可进一步做 session-wise 归一化以缓解跨时间分布偏移
- 对 20 被试做 subject-wise z-score 归一化，减少被试间差异

===== EEG 数据增强 Pipeline（针对 4D 张量 T×F×H×W 的 DE/PSD 特征）=====

注意：输入不是原始 EEG 波形而是预提取的 DE/PSD 特征，因此不可使用波形级增强（如时域翻转、滤波），应在特征空间上操作：
- 时间裁剪：沿 T 轴随机截取 70-90%
- 频段屏蔽：沿 F 轴随机置零 1 个频段（p=0.2）
- 电极 Dropout：沿 H×W 随机置零若干电极位置（p=0.1-0.15）
- 高斯噪声：对整个 4D 张量添加 N(0, σ²)，σ=0.05-0.1
- 时间平移：沿 T 轴 circular shift ±1-3 步
- 幅度缩放：全局乘以 U(0.8, 1.2) 的随机因子
- Mixup/CutMix：在 batch 内混合同类样本的特征表示（α=0.2）
每次随机选 2 种叠加应用。

===== 实施优先级 =====

1. 🔴 最高优先：Temporal Transformer 瘦身（层数/维度缩减）+ EEG 特征空间增强 Pipeline
2. 🟠 高优先：投影头正则化（Dropout+BN+降维）+ 温度参数调优 + 标签平滑 + Supervised Contrastive Loss
3. 🟡 中优先：Prompt 多样化 + 利用连续强度标签生成细粒度文本 + 联合损失 + 两阶段训练
4. 🟢 辅助：空间模块轻量化 + 频段 Masking + LayerDrop + 被试/session 归一化

每次仅改动一个模块，通过消融实验确认效果后再叠加下一个改进。在 LOSO 设置下同时监控训练集和验证集（留出被试）的 loss/accuracy 曲线，诊断过拟合程度。