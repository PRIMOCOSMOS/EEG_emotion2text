import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_trial_texts(n_trials: int = 80) -> Dict[int, Dict[str, str]]:
    corpus = {}
    for trial_idx in range(1, n_trials + 1):
        corpus[trial_idx] = {
            "level2": f"Trial {trial_idx}: emotional reaction to a movie clip segment.",
            "level3": (
                f"Trial {trial_idx} captures a full temporal emotional narrative from exposure, "
                "reaction onset, and sustained affective state over the viewing period."
            ),
        }
    return corpus


TRIAL_TEXTS = build_trial_texts(80)


@dataclass
class CFG:
    data_root: str = "/kaggle/input/seed-vii-eeg-feature"
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

    # Text encoder
    clip_name: str = "openai/clip-vit-large-patch14"
    max_text_len: int = 64


def find_subject_files(data_root: str) -> List[Path]:
    root = Path(data_root)
    mats = sorted(root.rglob("*.mat"))
    subject_files = []
    for p in mats:
        name = p.stem.lower()
        if "subject" in name or "sub" in name:
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
    subject_files = find_subject_files(data_root)
    if len(subject_files) == 0:
        raise FileNotFoundError(f"未找到 subject_*.mat: {data_root}")

    labels = load_trial_labels(data_root, n_trials=80)
    all_samples = []
    for sf in subject_files:
        all_samples.extend(load_subject_windows(sf, labels))
    return all_samples


class EEGTextWindowDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

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
            "text_l1": L1_PROTOTYPE[emotion],
            "text_l2": TRIAL_TEXTS[trial]["level2"],
            "text_l3": TRIAL_TEXTS[trial]["level3"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "subject": [b["subject"] for b in batch],
        "trial": [b["trial"] for b in batch],
        "text_l1": [b["text_l1"] for b in batch],
        "text_l2": [b["text_l2"] for b in batch],
        "text_l3": [b["text_l3"] for b in batch],
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

        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def _encode_text(self, texts: List[str], dev: torch.device) -> torch.Tensor:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        tok = {k: v.to(dev) for k, v in tok.items()}
        out = self.encoder(**tok)
        return out.pooler_output

    def forward(self, text_l1: List[str], text_l2: List[str], text_l3: List[str], dev: torch.device) -> torch.Tensor:
        f1 = self._encode_text(text_l1, dev)
        f2 = self._encode_text(text_l2, dev)
        f3 = self._encode_text(text_l3, dev)
        fused = 0.5 * f1 + 0.3 * f2 + 0.2 * f3
        z = self.proj(fused)
        return F.normalize(z, dim=-1)


def asymmetric_contrastive_loss(eeg_z: torch.Tensor, txt_z: torch.Tensor, temperature: float, alpha: float):
    logits = (eeg_z @ txt_z.t()) / temperature
    target = torch.arange(logits.size(0), device=logits.device)
    loss_e2t = F.cross_entropy(logits, target)
    loss_t2e = F.cross_entropy(logits.t(), target)
    loss = alpha * loss_e2t + (1.0 - alpha) * loss_t2e

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == target).float().mean().item()
    return loss, {"batch_retrieval_acc": acc}


def ensure_work_dir(work_dir: str) -> None:
    os.makedirs(work_dir, exist_ok=True)