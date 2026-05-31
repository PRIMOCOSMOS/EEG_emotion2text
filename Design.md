# 基于EEG-文本对齐的情绪识别
## 双编码器对比学习框架 · SEED-VII 数据集（两层文本版）

---

## 一、数据集：SEED-VII — 特征结构

```
数据来源    : 20个 .mat 文件，每个被试一个（subject_1.mat ~ subject_20.mat）
试次数量    : 每个被试 80 个试次（de_1 ~ de_80），对应 80 段电影片段
特征类型    : 差分熵（Differential Entropy, DE），4秒非重叠滑窗
数据维度    : de_i.shape = (T, 5, 62)
                T  : 试次 i 中 4 秒窗口的数量（范围：13 ~ 87）
                5  : 频率子带 [delta, theta, alpha, beta, gamma]
                62 : EEG 电极通道
```

`de_i` 每一行为一个 4 秒窗口样本。模型输入使用 `(62, 5)`。

---

## 二、标签来源与解析

情绪标签优先从 `Saveinfo` 读取：

```
Saveinfo/*.csv  (例如: 1_20221001_1_save_info.csv)
行格式: emotion tasks,movie\七类\1\happy\xxx.mp4,0.6
```

解析规则：
1. 从第二列视频路径提取情绪目录（如 `happy`）。
2. 映射到七类标签：neutral / joy / sadness / fear / disgust / anger / surprise。
3. 每个被试取前 80 行作为 trial 标签。
4. 若某被试缺少 Saveinfo 文件，回退到 `label.mat`。

---

## 三、两层文本监督设计

### 3.1 文本层级

| 层级 | 监督粒度 | 内容来源 | 备注 |
|------|----------|----------|------|
| L1   | Window Level | 情绪标签级共享文本 | 同一情绪的所有窗口共享 |
| L2   | Trial Level  | 80 个视频描述 | 按 trial id 对齐 |

> 不再使用 L3（完整叙述层）。

### 3.2 融合方式

文本编码器使用冻结 CLIP Text Encoder，输出后进行两层融合：

```
z_text = normalize(Proj( w1 * f(L1) + w2 * f(L2) ))
默认: w1=0.4, w2=0.6
```

---

## 四、模型架构

```
EEG 输入 (B,62,5)
    -> EEGNet-DE 变体编码器
    -> EEG projection head
    -> z_eeg (B,128)

文本输入 (L1,L2)
    -> CLIP Text Encoder (冻结)
    -> Text projection head
    -> z_text (B,128)

在共享空间进行对比学习
```

---

## 五、损失函数（更新版）

### 5.1 相似度规则（不再使用同试次约束）

```
相同情绪                -> 强正样本 (weight = 1.0)
不同情绪                -> 负样本   (weight = 0.0)
```

对每一行做归一化，得到合法概率分布，用于 soft contrastive CE。

### 5.2 间距约束（Margin）

增加一条 margin 约束：
1. `sim(同情绪) - sim(不同情绪) >= m_pos_neg`

总损失：

```
L = alpha * L_eeg2text + (1-alpha) * L_text2eeg + lambda * L_margin
默认: alpha=0.75, lambda=0.20
```

---

## 六、CSV 协议（两层文本）

文件：`text_protocol_l1_l2.csv`

```csv
emotion,l1_text,trial,l2_text
neutral,This EEG window reflects neutral affective evidence.,,
joy,This EEG window reflects joyful affective evidence.,,
sadness,This EEG window reflects sad affective evidence.,,
fear,This EEG window reflects fearful affective evidence.,,
disgust,This EEG window reflects disgust-related affective evidence.,,
anger,This EEG window reflects angry affective evidence.,,
surprise,This EEG window reflects surprised affective evidence.,,
,,1,Trial 1 video description.
,,2,Trial 2 video description.
```

规则：
1. L1 必须按情绪填写（一个情绪一条共享文本），不使用全局共享文本。
2. `emotion` 支持 happy/neutral/sad/fear/disgust/anger/surprise（内部映射到七类标准标签）。
3. `trial=1..80` 填写每个视频的 L2 描述。
4. 缺失项自动使用默认模板。

---

## 七、Kaggle 训练策略

1. 默认支持断点保存：`latest_*.pt`。
2. 支持续训：`resume=True` 自动加载 latest。
3. 支持最大训练时长：`max_train_hours` + `time_buffer_minutes`，在被 Kaggle 强制结束前主动退出并落盘。
4. 输出目录：`/kaggle/working/eeg2text_ckpt`。