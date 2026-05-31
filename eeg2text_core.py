import os
import random
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer


EMOTION_NAMES = ["neutral", "joy", "sadness", "fear", "disgust", "anger", "surprise"]

L1_PROTOTYPE = {
    "neutral": "The person feels calm and emotionally balanced.",
    "joy": "The person feels happy, positive, and energetic.",
    "sadness": "The person feels low, withdrawn, and sorrowful.",
    "fear": "The person feels anxious, tense, and threatened.",
    "disgust": "The person feels aversion and rejection.",
    "anger": "The person feels irritated, hostile, and angry.",
    "surprise": "The person experiences sudden astonishment.",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_default_l1_texts() -> Dict[str, str]:
    # Window-level text; user can set all emotions to the same text if needed.
    return {emo: L1_PROTOTYPE[emo] for emo in EMOTION_NAMES}


def build_default_l2_texts(n_trials: int = 80) -> Dict[int, str]:
    return {trial_idx: f"Trial {trial_idx}: emotional reaction to the video clip." for trial_idx in range(1, n_trials + 1)}


DEFAULT_L1_TEXTS = build_default_l1_texts()
DEFAULT_L2_TEXTS = build_default_l2_texts(80)

EMOTION_ALIASES = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "fear": "fear",
    "disgust": "disgust",
    "anger": "anger",
    "angry": "anger",
    "surprise": "surprise",
}


def normalize_emotion_name(raw: str) -> str:
    key = (raw or "").strip().lower()
    key = key.replace(" ", "")
    return EMOTION_ALIASES.get(key, "neutral")


def emotion_name_to_label(name: str) -> int:
    norm = normalize_emotion_name(name)
    return EMOTION_NAMES.index(norm)


def _extract_subject_id(text: str) -> Optional[int]:
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    return int(nums[0])


def _extract_emotion_from_video_path(video_path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", video_path) if p]
    if len(parts) < 2:
        return "neutral"
    # 形如 movie\七类\1\happy\x.mp4 -> 倒数第二段是情绪
    return normalize_emotion_name(parts[-2])


def load_saveinfo_trial_labels(saveinfo_csv: str, n_trials: Optional[int] = 80) -> np.ndarray:
    labels = []
    with open(saveinfo_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            emotion = _extract_emotion_from_video_path(row[1])
            labels.append(emotion_name_to_label(emotion))

    if n_trials is None:
        return np.array(labels, dtype=np.int64)

    if len(labels) < n_trials:
        raise ValueError(f"Saveinfo 行数不足: {saveinfo_csv}, got={len(labels)}, expected>={n_trials}")
    return np.array(labels[:n_trials], dtype=np.int64)


def find_saveinfo_files(data_root: str, saveinfo_dir: Optional[str] = None) -> List[Path]:
    if saveinfo_dir:
        root = Path(saveinfo_dir)
    else:
        root = Path(data_root)
    files = []
    files.extend(sorted(root.rglob("*_save_info.csv")))
    files.extend(sorted(root.rglob("*save*info*.csv")))
    files.extend(sorted(root.rglob("*save*info*")))
    files.extend(sorted(root.rglob("*.CSV")))
    seen = set()
    uniq = []
    for p in files:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        name = p.name.lower()
        if "trigger" in name:
            continue
        if "save" not in name or "info" not in name:
            continue
        if p.is_dir():
            continue
        uniq.append(p)
    return uniq


def build_subject_label_map_from_saveinfo(data_root: str, saveinfo_dir: Optional[str] = None, n_trials: int = 80):
    saveinfo_files = find_saveinfo_files(data_root, saveinfo_dir)
    grouped: Dict[int, List[Path]] = {}
    for fp in saveinfo_files:
        sid = _extract_subject_id(fp.stem)
        if sid is None:
            continue
        grouped.setdefault(sid, []).append(fp)

    subject_label_map: Dict[int, np.ndarray] = {}
    for sid, files in grouped.items():
        merged: List[int] = []
        for fp in sorted(files):
            try:
                arr = load_saveinfo_trial_labels(str(fp), n_trials=None)
            except Exception:
                continue
            merged.extend(arr.tolist())
            if len(merged) >= n_trials:
                break
        if len(merged) >= n_trials:
            subject_label_map[sid] = np.array(merged[:n_trials], dtype=np.int64)
    return subject_label_map


def load_two_level_texts_from_csv(text_csv_path: str, n_trials: int = 80):
    # 协议(单文件):
    # 必备列: emotion,l1_text,trial,l2_text
    # L1: emotion + l1_text
    # L2: trial + l2_text (trial=1..80)
    # 若缺失字段，自动回退默认文本。
    l1_texts = build_default_l1_texts()
    l2_texts = build_default_l2_texts(n_trials)

    def _get(row: Dict[str, str], keys: List[str]) -> str:
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k])
        return ""

    with open(text_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emo_raw = (_get(row, ["emotion", "Emotion", "emo", "label"]) or "").strip().lower()
            l1_raw = (_get(row, ["l1_text", "L1_text", "l1", "L1", "window_text"]) or "").strip()
            if l1_raw:
                if emo_raw:
                    emo = normalize_emotion_name(emo_raw)
                    l1_texts[emo] = l1_raw

            trial_raw = (_get(row, ["trial", "Trial", "trial_id", "id"]) or "").strip()
            l2_raw = (_get(row, ["l2_text", "L2_text", "l2", "L2", "trial_text", "description"]) or "").strip()
            if trial_raw and l2_raw:
                try:
                    trial = int(trial_raw)
                except Exception:
                    trial = -1
                if 1 <= trial <= n_trials:
                    l2_texts[trial] = l2_raw

    return l1_texts, l2_texts


@dataclass
class CFG:
    data_root: str = "/kaggle/input/datasets/primocosmos/seed-vii-kaggle/EEG_features"
    work_dir: str = "/kaggle/working/eeg2text_ckpt"

    # Model
    f1: int = 16
    depth_mult: int = 2
    f2: int = 32
    embed_dim: int = 128
    attn_heads: int = 4
    drop_conv: float = 0.25
    drop_attn: float = 0.10

    # Loss
    temperature: float = 0.07
    alpha: float = 0.75

    # Train
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    amp: bool = True
    val_ratio: float = 0.1
    seed: int = 42

    # Resume and time budget
    resume: bool = True
    save_every_n_steps: int = 100
    max_train_hours: float = 8.8
    time_buffer_minutes: int = 8
    saveinfo_dir: Optional[str] = "/kaggle/input/datasets/primocosmos/seed-vii-kaggle/save_info"
    text_csv_path: Optional[str] = None
    l1_l2_text_csv_path: Optional[str] = "/kaggle/input/datasets/primocosmos/seed-vii-kaggle/Emotion2text/text_protocol_template.csv"

    # Text encoder
    clip_name: str = "openai/clip-vit-large-patch14"
    max_text_len: int = 64
    l1_weight: float = 0.4
    l2_weight: float = 0.6
    same_emotion_weight: float = 1.0
    pos_neg_margin: float = 0.20
    margin_loss_weight: float = 0.20


def find_subject_files(data_root: str) -> List[Path]:
    root = Path(data_root)
    mats = sorted(root.rglob("*.mat"))
    subject_files = []
    for p in mats:
        name = p.stem.lower()
        if "subject" in name or "sub" in name:
            subject_files.append(p)
    if len(subject_files) == 0:
        for p in mats:
            name = p.stem.lower()
            if "label" in name or "readme" in name:
                continue
            subject_files.append(p)
    return subject_files


def load_trial_labels(data_root: str, n_trials: int = 80) -> np.ndarray:
    root = Path(data_root)
    candidate = list(root.rglob("*label*.mat"))
    for p in candidate:
        obj = sio.loadmat(p)
        for k, v in obj.items():
            if k.startswith("__"):
                continue
            arr = np.array(v).squeeze()
            if arr.ndim == 1 and len(arr) >= n_trials:
                arr = arr[:n_trials].astype(np.int64)
                if arr.min() == 1:
                    arr = arr - 1
                return arr
    raise FileNotFoundError("未找到 trial label 文件，请在 data_root 下提供 label .mat。")


def load_subject_windows(subject_file: Path, trial_labels: np.ndarray) -> List[Dict]:
    mat = sio.loadmat(subject_file)
    samples = []

    for trial_idx in range(1, len(trial_labels) + 1):
        key = f"de_{trial_idx}"
        if key not in mat:
            continue
        arr = np.asarray(mat[key], dtype=np.float32)
        if arr.ndim != 3:
            continue

        # Design.md: (T,5,62) -> model input (T,62,5)
        arr = np.transpose(arr, (0, 2, 1))
        label = int(trial_labels[trial_idx - 1])
        label = max(0, min(label, len(EMOTION_NAMES) - 1))

        for t in range(arr.shape[0]):
            samples.append(
                {
                    "subject": subject_file.stem,
                    "trial": trial_idx,
                    "label": label,
                    "eeg": arr[t],
                }
            )

    return samples


def load_all_samples(data_root: str) -> List[Dict]:
    return load_all_samples_with_saveinfo(data_root=data_root, saveinfo_dir=None)


def load_all_samples_with_saveinfo(data_root: str, saveinfo_dir: Optional[str] = None) -> List[Dict]:
    subject_files = find_subject_files(data_root)
    if len(subject_files) == 0:
        raise FileNotFoundError(f"未找到 subject_*.mat: {data_root}")

    subject_label_map = build_subject_label_map_from_saveinfo(data_root, saveinfo_dir=saveinfo_dir, n_trials=80)
    fallback_labels = None

    all_samples = []
    for sf in subject_files:
        sid = _extract_subject_id(sf.stem)
        labels = subject_label_map.get(sid)
        if labels is None:
            if fallback_labels is None:
                fallback_labels = load_trial_labels(data_root, n_trials=80)
            labels = fallback_labels
        all_samples.extend(load_subject_windows(sf, labels))
    return all_samples


class EEGTextWindowDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        l1_texts: Optional[Dict[str, str]] = None,
        l2_texts: Optional[Dict[int, str]] = None,
    ):
        self.rows = rows
        self.l1_texts = l1_texts if l1_texts is not None else DEFAULT_L1_TEXTS
        self.l2_texts = l2_texts if l2_texts is not None else DEFAULT_L2_TEXTS

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        x = torch.tensor(row["eeg"], dtype=torch.float32)
        trial = row["trial"]
        emotion = EMOTION_NAMES[int(row["label"])]
        return {
            "eeg": x,
            "label": torch.tensor(row["label"], dtype=torch.long),
            "subject": row["subject"],
            "trial": trial,
            "text_l1": self.l1_texts.get(emotion, DEFAULT_L1_TEXTS[emotion]),
            "text_l2": self.l2_texts.get(trial, DEFAULT_L2_TEXTS[trial]),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "subject": [b["subject"] for b in batch],
        "trial": [b["trial"] for b in batch],
        "text_l1": [b["text_l1"] for b in batch],
        "text_l2": [b["text_l2"] for b in batch],
    }


def split_train_val(rows: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    idx = np.arange(len(rows))
    np.random.shuffle(idx)
    n_val = int(len(rows) * val_ratio)
    val_idx = set(idx[:n_val].tolist())
    train_rows, val_rows = [], []
    for i, row in enumerate(rows):
        if i in val_idx:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


class SpectralTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class EEGEncoder(nn.Module):
    def __init__(self, f1=16, depth_mult=2, f2=32, embed_dim=128, heads=4, drop_conv=0.25, drop_attn=0.1):
        super().__init__()
        f1d = f1 * depth_mult

        self.b1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(f1),
            nn.GELU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(f1, f1d, kernel_size=(62, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1d),
            nn.GELU(),
            nn.Dropout2d(drop_conv),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(f1d, f1d, kernel_size=(1, 3), padding=(0, 1), groups=f1d, bias=False),
            nn.Conv2d(f1d, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.GELU(),
            nn.Dropout(drop_conv),
        )

        self.spec_attn = SpectralTransformerBlock(d_model=f2, nhead=heads, dropout=drop_attn)
        self.embed_mlp = nn.Sequential(
            nn.Linear(f2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.spec_attn(x)
        x = x.mean(dim=1)
        x = self.embed_mlp(x)
        x = self.proj_head(x)
        return F.normalize(x, dim=-1)


class TextTower(nn.Module):
    def __init__(self, clip_name: str, embed_dim: int = 128, max_len: int = 64):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.encoder = CLIPTextModel.from_pretrained(clip_name)
        self.max_len = max_len
        self.cache: Dict[str, torch.Tensor] = {}

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def _encode_text(self, texts: List[str], dev: torch.device) -> torch.Tensor:
        self.encoder.eval()
        out_feats: List[torch.Tensor] = []
        uncached: List[str] = []
        for t in texts:
            if t not in self.cache:
                uncached.append(t)

        if uncached:
            tok = self.tokenizer(
                uncached,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            tok = {k: v.to(dev) for k, v in tok.items()}
            feats = self.encoder(**tok).pooler_output.detach().float().cpu()
            for t, f in zip(uncached, feats):
                self.cache[t] = f

        for t in texts:
            out_feats.append(self.cache[t])

        return torch.stack(out_feats, dim=0).to(dev)

    def forward(self, text_l1: List[str], text_l2: List[str], dev: torch.device, l1_weight: float = 0.4, l2_weight: float = 0.6) -> torch.Tensor:
        f1 = self._encode_text(text_l1, dev)
        f2 = self._encode_text(text_l2, dev)
        fused = l1_weight * f1 + l2_weight * f2
        z = self.proj(fused)
        return F.normalize(z, dim=-1)


def asymmetric_contrastive_loss(eeg_z: torch.Tensor, txt_z: torch.Tensor, temperature: float, alpha: float):
    raise RuntimeError("Use asymmetric_soft_contrastive_loss with metadata.")


def build_similarity_targets(
    labels: torch.Tensor,
    device: torch.device,
    same_emotion_weight: float = 1.0,
) -> torch.Tensor:
    # Updated rule:
    # same emotion -> positive
    # different emotion -> negative
    bsz = int(labels.shape[0])
    sim = torch.zeros((bsz, bsz), dtype=torch.float32, device=device)

    for i in range(bsz):
        for j in range(bsz):
            same_emotion = int(labels[i].item()) == int(labels[j].item())

            if same_emotion:
                sim[i, j] = same_emotion_weight
            else:
                sim[i, j] = 0.0

    # Row-wise normalization to valid probability distribution.
    row_sum = sim.sum(dim=1, keepdim=True)
    zero_mask = row_sum.squeeze(1) <= 0
    if zero_mask.any():
        zero_idx = torch.where(zero_mask)[0]
        sim[zero_idx, zero_idx] = 1.0
        row_sum = sim.sum(dim=1, keepdim=True)

    target = sim / row_sum.clamp_min(1e-12)
    return target


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def asymmetric_soft_contrastive_loss(
    eeg_z: torch.Tensor,
    txt_z: torch.Tensor,
    temperature: float,
    alpha: float,
    labels: torch.Tensor,
    same_emotion_weight: float = 1.0,
    pos_neg_margin: float = 0.20,
    margin_loss_weight: float = 0.20,
):
    logits = (eeg_z @ txt_z.t()) / temperature

    target_e2t = build_similarity_targets(
        labels,
        logits.device,
        same_emotion_weight=same_emotion_weight,
    )
    target_t2e = target_e2t.t()
    target_t2e = target_t2e / target_t2e.sum(dim=1, keepdim=True).clamp_min(1e-12)

    loss_e2t = soft_cross_entropy(logits, target_e2t)
    loss_t2e = soft_cross_entropy(logits.t(), target_t2e)

    # Margin term: same-emotion pairs should be closer than different-emotion pairs.
    sim = eeg_z @ txt_z.t()
    bsz = sim.size(0)
    margin_terms = []
    for i in range(bsz):
        label_i = int(labels[i].item())
        pos_mask = torch.tensor(
            [int(labels[j].item()) == label_i for j in range(bsz)],
            dtype=torch.bool,
            device=sim.device,
        )
        neg_mask = torch.tensor(
            [int(labels[j].item()) != label_i for j in range(bsz)],
            dtype=torch.bool,
            device=sim.device,
        )

        s_pos = sim[i][pos_mask].mean() if pos_mask.any() else sim[i, i]
        s_neg = sim[i][neg_mask].mean() if neg_mask.any() else (s_pos - 1.0)

        margin_terms.append(F.relu(pos_neg_margin - (s_pos - s_neg)))

    margin_loss = torch.stack(margin_terms).mean() if len(margin_terms) > 0 else torch.tensor(0.0, device=sim.device)

    base_loss = alpha * loss_e2t + (1.0 - alpha) * loss_t2e
    loss = base_loss + margin_loss_weight * margin_loss

    with torch.no_grad():
        # Keep a simple retrieval metric for monitoring.
        hard_target = torch.arange(logits.size(0), device=logits.device)
        pred = logits.argmax(dim=1)
        acc = (pred == hard_target).float().mean().item()
    return loss, {"batch_retrieval_acc": acc}


def ensure_work_dir(work_dir: str) -> None:
    os.makedirs(work_dir, exist_ok=True)