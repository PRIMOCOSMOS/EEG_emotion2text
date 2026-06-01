"""
SEED-VII data loading + 4D topographic representation construction.

This module PORTS the original data/label/CSV protocols from
PRIMOCOSMOS/EEG_emotion2text (Saveinfo parsing, subject/label discovery,
Kaggle path normalization) and ADDS the 4D topographic input pipeline
required by the EmotionCLIP / SST-LegoViT technical path (Yan et al., 2025).

Original DE feature:  de_i.shape = (T, 5, 62)
    T  : number of 4s windows in trial i
    5  : frequency sub-bands [delta, theta, alpha, beta, gamma]
    62 : EEG electrodes

EmotionCLIP input:    (frames, bands, H, W)
    Each 4s DE window is treated as ONE temporal frame split into
    `image_frames` sub-frames (here we replicate / pool the single window
    into the requested number of frames). Each of the 5 bands is mapped from
    62 electrodes onto a 9x9 scalp topographic grid (SEED electrode layout)
    and bicubic-resized to (H, W). See MIGRATION.md for details.
"""

import os
import re
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Emotion vocabulary (7-class SEED-VII), ported from the original repo.
# --------------------------------------------------------------------------- #
EMOTION_NAMES = ["neutral", "joy", "sadness", "fear", "disgust", "anger", "surprise"]

# Human-readable class words used to build CLIP text prompts (EmotionCLIP path).
EMOTION_PROMPT_WORD = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fearful",
    "disgust": "disgusted",
    "anger": "angry",
    "surprise": "surprised",
}

EMOTION_ALIASES = {
    "neutral": "neutral", "calm": "neutral",
    "happy": "joy", "joy": "joy",
    "sad": "sadness", "sadness": "sadness",
    "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust",
    "anger": "anger", "angry": "anger",
    "surprise": "surprise", "surprised": "surprise",
}


def normalize_emotion_name(raw: str) -> str:
    key = (raw or "").strip().lower().replace(" ", "")
    return EMOTION_ALIASES.get(key, "neutral")


def emotion_name_to_label(name: str) -> int:
    return EMOTION_NAMES.index(normalize_emotion_name(name))


def class_prompt_words(num_classes: int = 7) -> List[str]:
    """Return the list of class words for the active label set."""
    return [EMOTION_PROMPT_WORD[e] for e in EMOTION_NAMES[:num_classes]]


# --------------------------------------------------------------------------- #
# Path helpers (ported from original repo).
# --------------------------------------------------------------------------- #
def _extract_subject_id(text: str) -> Optional[int]:
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else None


def _parse_subject_session(stem: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse (subject_id, session_id) from a Saveinfo filename stem.

    SEED-VII Saveinfo files are named like:
        <subject>_<date>_<session>_save_info   e.g. "1_20221001_3_save_info"
        <subject>_<session>_save_info          e.g. "1_3_save_info"
    Each session contains 20 trials; 4 sessions x 20 = 80 trials, and these MUST
    be concatenated in ascending session order (1->2->3->4) so they align
    one-to-one with trial 1..80 in the text CSV.

    Strategy:
      * subject_id = first integer in the name.
      * session_id = the integer token that lies in [1..8] (excludes the long
        date token), preferring the LAST such qualifying token (the session
        index sits right before "save_info"). Falls back to last integer.
    Returns (subject_id, session_id); session_id may be None if unpar. in which
    case the caller keeps natural order.
    """
    # strip trailing "save_info"/"saveinfo" so it doesn't contribute digits
    core = re.sub(r"[_\- ]*save[_\- ]*info.*$", "", stem, flags=re.IGNORECASE)
    nums = re.findall(r"\d+", core)
    if not nums:
        return None, None
    subject_id = int(nums[0])
    session_id = None
    # session candidates: short tokens (len<=2) with value in 1..8, excluding the
    # subject token position when possible. Take the last qualifying token.
    for tok in nums[1:][::-1]:
        if len(tok) <= 2 and 1 <= int(tok) <= 8:
            session_id = int(tok)
            break
    if session_id is None and len(nums) >= 2:
        # fallback: last integer that is not the (long) date token
        for tok in nums[1:][::-1]:
            if len(tok) <= 2:
                session_id = int(tok)
                break
    return subject_id, session_id


def _normalize_kaggle_input_path(path_str: str) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p)
    parts = [x for x in p.parts if x]
    try:
        idx = parts.index("datasets")
        if idx + 2 < len(parts):
            dataset = parts[idx + 2]
            tail = parts[idx + 3:]
            alt = Path("/") / "kaggle" / "input" / dataset
            for t in tail:
                alt = alt / t
            if alt.exists():
                return str(alt)
    except ValueError:
        pass
    return str(p)


def _extract_emotion_from_video_path(video_path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", video_path) if p]
    if len(parts) < 2:
        return "neutral"
    return normalize_emotion_name(parts[-2])


# --------------------------------------------------------------------------- #
# Label parsing from Saveinfo (ported).
# --------------------------------------------------------------------------- #
def load_saveinfo_trial_labels(saveinfo_csv: str, n_trials: Optional[int] = 80) -> np.ndarray:
    labels = []
    with open(saveinfo_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            labels.append(emotion_name_to_label(_extract_emotion_from_video_path(row[1])))
    if n_trials is None:
        return np.array(labels, dtype=np.int64)
    if len(labels) < n_trials:
        raise ValueError(f"Saveinfo rows insufficient: {saveinfo_csv}, got={len(labels)}, need>={n_trials}")
    return np.array(labels[:n_trials], dtype=np.int64)


def find_saveinfo_files(data_root: str, saveinfo_dir: Optional[str] = None) -> List[Path]:
    root = Path(_normalize_kaggle_input_path(saveinfo_dir or data_root))
    files = []
    files.extend(sorted(root.rglob("*_save_info.csv")))
    files.extend(sorted(root.rglob("*save*info*.csv")))
    files.extend(sorted(root.rglob("*save*info*")))
    seen, uniq = set(), []
    for p in files:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        name = p.name.lower()
        if "trigger" in name or "save" not in name or "info" not in name or p.is_dir():
            continue
        uniq.append(p)
    return uniq


def build_subject_label_map_from_saveinfo(data_root, saveinfo_dir=None, n_trials=80,
                                          trials_per_session=20, verbose=False):
    """Build {subject_id -> [label_0 .. label_{n_trials-1}]} by concatenating the
    4 per-session Saveinfo files IN ASCENDING SESSION ORDER.

    Each session file holds `trials_per_session` (default 20) trials; sessions
    1->2->3->4 are concatenated so the resulting 80 labels align one-to-one with
    trial 1..80 in the text CSV (4 x 20 = 80).
    """
    # group files as (session_id, path) per subject
    grouped: Dict[int, List[Tuple[Optional[int], Path]]] = {}
    for fp in find_saveinfo_files(data_root, saveinfo_dir):
        sid, sess = _parse_subject_session(fp.stem)
        if sid is None:
            continue
        grouped.setdefault(sid, []).append((sess, fp))

    subject_label_map: Dict[int, np.ndarray] = {}
    for sid, sess_files in grouped.items():
        # sort by session id (None last); ties broken by filename for determinism
        sess_files = sorted(sess_files, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0, str(x[1])))
        merged: List[int] = []
        order_log = []
        for sess, fp in sess_files:
            try:
                arr = load_saveinfo_trial_labels(str(fp), n_trials=None).tolist()
            except Exception:
                continue
            merged.extend(arr)
            order_log.append((sess, fp.name, len(arr)))
            if len(merged) >= n_trials:
                break
        if merged:
            subject_label_map[sid] = np.array(merged[:n_trials], dtype=np.int64)
            if verbose:
                print(f"[saveinfo] subject {sid}: session order -> "
                      + " | ".join(f"s{s}:{nm}({c})" for s, nm, c in order_log)
                      + f"  total={len(merged)} (kept {min(len(merged), n_trials)})")
                # sanity: warn if any session does not have exactly trials_per_session
                bad = [(s, nm, c) for s, nm, c in order_log if c != trials_per_session]
                if bad and len(order_log) > 1:
                    print(f"[saveinfo][WARN] subject {sid}: some sessions != "
                          f"{trials_per_session} trials: {bad}")
    return subject_label_map


def find_subject_files(data_root: str) -> List[Path]:
    root = Path(_normalize_kaggle_input_path(data_root))
    mats = sorted(root.rglob("*.mat"))
    subject_files = [p for p in mats if "subject" in p.stem.lower() or "sub" in p.stem.lower()]
    if not subject_files:
        subject_files = [p for p in mats if "label" not in p.stem.lower() and "readme" not in p.stem.lower()]
    return subject_files


def load_trial_labels(data_root: str, n_trials: int = 80) -> np.ndarray:
    root = Path(_normalize_kaggle_input_path(data_root))
    for p in list(root.rglob("*label*.mat")):
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
    raise FileNotFoundError("No trial label .mat found under data_root.")


# --------------------------------------------------------------------------- #
# Two-level text protocol CSV (ported). We keep L2 (trial-level) text and
# expose it as an EXTRA prompt source for the EmotionCLIP text tower.
# --------------------------------------------------------------------------- #
def build_default_l2_texts(n_trials: int = 80) -> Dict[int, str]:
    return {i: f"Trial {i}: emotional reaction to the video clip." for i in range(1, n_trials + 1)}


def load_two_level_texts_from_csv(text_csv_path: str, n_trials: int = 80) -> Tuple[Dict[str, str], Dict[int, str]]:
    """Returns (l1_texts_by_emotion, l2_texts_by_trial).

    L1 (emotion-level shared text) is kept for backward compatibility / docs.
    L2 (trial-level descriptions) feeds extra prompts in the new pipeline.
    """
    l1_texts: Dict[str, str] = {}
    l2_texts = build_default_l2_texts(n_trials)

    def _get(row, keys):
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k])
        return ""

    with open(text_csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            emo_raw = (_get(row, ["emotion", "Emotion", "emo", "label"]) or "").strip().lower()
            l1_raw = (_get(row, ["l1_text", "L1_text", "l1", "L1", "window_text"]) or "").strip()
            if l1_raw and emo_raw:
                l1_texts[normalize_emotion_name(emo_raw)] = l1_raw

            trial_raw = (_get(row, ["trial", "Trial", "trial_id", "id"]) or "").strip()
            l2_raw = (_get(row, ["l2_text", "L2_text", "l2", "L2", "trial_text", "description"]) or "").strip()
            if trial_raw and l2_raw:
                try:
                    trial = int(trial_raw)
                except Exception:
                    continue
                if 1 <= trial <= n_trials:
                    l2_texts[trial] = l2_raw
    return l1_texts, l2_texts


# --------------------------------------------------------------------------- #
# 4D topographic conversion (NEW; required by EmotionCLIP path).
# --------------------------------------------------------------------------- #
# SEED 62-channel -> 9x9 grid mapping (same layout used by the reference repo's
# datasets/dataset.py `_data_1D_to_2D_`). SEED-VII shares the SEED 62-channel
# 10-20 montage, so the layout is reused directly.
def _band_vector_to_2d(data_1d: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Map a 62-d electrode vector to a (H,W) topographic image (bicubic)."""
    data_1d = (data_1d - np.mean(data_1d)) / (np.std(data_1d) + 1e-8)
    grid = np.zeros((9, 9), dtype=np.float32)
    grid[0, 3:6] = data_1d[0:3]
    grid[1, 3], grid[1, 5] = data_1d[3], data_1d[4]
    for i in range(5):
        grid[i + 2, :] = data_1d[5 + i * 9: 5 + (i + 1) * 9]
    grid[7, 1:8] = data_1d[50:57]
    grid[8, 2:7] = data_1d[57:62]
    img = np.array(Image.fromarray(grid).resize(size, resample=Image.BICUBIC), dtype=np.float32)
    return img


def de_window_to_4d(de_window: np.ndarray, image_frames: int, image_channels: int,
                    image_height: int, image_width: int) -> np.ndarray:
    """Convert one DE window (5, 62) -> 4D tensor (frames, channels, H, W).

    SEED-VII provides only DE over 5 bands (no PSD, no intra-window frames).
    To match SST-LegoViT's (frames, bands, H, W) expectation:
      * The 5 DE bands are each turned into a 9x9 -> (H,W) topographic map.
      * To fill `image_channels` (default 10 = 5 DE + 5 "PSD" slots in the
        reference), we duplicate the DE band maps; this keeps the dual-stream
        Legoformer (DE/PSD) structurally valid. See MIGRATION.md.
      * The single 4s window is replicated across `image_frames` temporal
        frames so the temporal transformer receives a valid sequence.
    de_window: shape (5, 62)
    """
    bands = de_window.shape[0]  # 5 for SEED-VII
    size = (image_width, image_height)
    band_imgs = np.stack([_band_vector_to_2d(de_window[b], size) for b in range(bands)], axis=0)  # (5,H,W)

    # Build the channel (spectral) axis up to image_channels.
    if image_channels <= bands:
        chan = band_imgs[:image_channels]
    else:
        reps = int(np.ceil(image_channels / bands))
        chan = np.concatenate([band_imgs] * reps, axis=0)[:image_channels]  # (C,H,W)

    # Replicate the window across frames.
    frame = chan[np.newaxis, ...]                       # (1,C,H,W)
    frames = np.repeat(frame, image_frames, axis=0)     # (frames,C,H,W)
    return frames.astype(np.float32)


# --------------------------------------------------------------------------- #
# Subject window loading (ported + z-score), now storing raw DE windows.
# --------------------------------------------------------------------------- #
def load_subject_windows(subject_file: Path, trial_labels: np.ndarray,
                         normalize_subject_zscore: bool = True) -> List[Dict]:
    mat = sio.loadmat(subject_file)
    trial_data = []
    for trial_idx in range(1, len(trial_labels) + 1):
        key = f"de_{trial_idx}"
        if key not in mat:
            continue
        arr = np.asarray(mat[key], dtype=np.float32)  # (T,5,62)
        if arr.ndim != 3:
            continue
        label = int(max(0, min(int(trial_labels[trial_idx - 1]), len(EMOTION_NAMES) - 1)))
        trial_data.append((trial_idx, arr, label))

    if normalize_subject_zscore and trial_data:
        all_win = np.concatenate([x[1] for x in trial_data], axis=0)  # (N,5,62)
        mu = all_win.mean(axis=0, keepdims=True)
        sigma = np.clip(all_win.std(axis=0, keepdims=True), 1e-6, None)
        trial_data = [(ti, (arr - mu) / sigma, lb) for ti, arr, lb in trial_data]

    samples = []
    for trial_idx, arr, label in trial_data:
        for t in range(arr.shape[0]):
            samples.append({
                "subject": subject_file.stem,
                "trial": trial_idx,
                "label": label,
                "de": arr[t],  # (5,62)
            })
    return samples


def load_all_samples_with_saveinfo(data_root, saveinfo_dir=None, n_trials=80,
                                   normalize_subject_zscore=True) -> List[Dict]:
    data_root = _normalize_kaggle_input_path(data_root)
    if saveinfo_dir:
        saveinfo_dir = _normalize_kaggle_input_path(saveinfo_dir)

    subject_files = find_subject_files(data_root)
    if not subject_files:
        raise FileNotFoundError(f"No subject_*.mat found under: {data_root}")

    print(f"save_info files detected: {len(find_saveinfo_files(data_root, saveinfo_dir))}")
    subject_label_map = build_subject_label_map_from_saveinfo(
        data_root, saveinfo_dir, n_trials, verbose=True)
    print(f"saveinfo subjects parsed: {len(subject_label_map)}")

    fallback_labels = None
    all_samples = []
    for sf in subject_files:
        sid = _extract_subject_id(sf.stem)
        labels = subject_label_map.get(sid)
        if labels is None:
            if fallback_labels is None:
                fallback_labels = load_trial_labels(data_root, n_trials)
            labels = fallback_labels
        all_samples.extend(load_subject_windows(sf, labels, normalize_subject_zscore))
    return all_samples


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EEGTopoDataset(Dataset):
    """Yields {image, label, trial, subject}.

    PERFORMANCE: the 4D topographic map is computed LAZILY in __getitem__
    (on the fly) instead of materializing all windows in __init__. Only the raw
    DE windows (5, 62) are kept (tiny), so memory stays ~O(N * 5 * 62) floats
    instead of O(N * frames * channels * H * W) which is ~8000x larger and was
    causing long stalls / OOM before the first epoch on Kaggle.

    With DataLoader(num_workers>0) the per-window conversion runs in parallel
    across worker processes, fully overlapping with GPU compute.
    """

    def __init__(self, rows: List[Dict], image_frames: int, image_channels: int,
                 image_height: int, image_width: int):
        self.image_frames = image_frames
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width

        self.labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
        self.trials = [int(r["trial"]) for r in rows]
        self.subjects = [str(r["subject"]) for r in rows]
        # keep only the raw (5,62) DE windows packed contiguously (small).
        self.de = torch.from_numpy(
            np.ascontiguousarray(np.stack([r["de"] for r in rows], axis=0))
        ).to(torch.float32)  # (N, 5, 62)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        de = self.de[idx].numpy()  # (5,62)
        img = de_window_to_4d(de, self.image_frames, self.image_channels,
                              self.image_height, self.image_width)  # (frames,C,H,W)
        return {
            "image": torch.from_numpy(img),
            "label": self.labels[idx],
            "trial": self.trials[idx],
            "subject": self.subjects[idx],
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "trial": [b["trial"] for b in batch],
        "subject": [b["subject"] for b in batch],
    }


# --------------------------------------------------------------------------- #
# Splits (ported).
# --------------------------------------------------------------------------- #
def split_train_val_by_subject(rows, val_ratio=0.1, seed=42):
    subjects = sorted({str(r["subject"]) for r in rows})
    if len(subjects) <= 1:
        return rows, []
    rng = np.random.default_rng(seed)
    perm = subjects.copy()
    rng.shuffle(perm)
    n_val = min(max(1, round(len(subjects) * val_ratio)), len(subjects) - 1)
    val_subjects = set(perm[:n_val])
    train_rows = [r for r in rows if str(r["subject"]) not in val_subjects]
    val_rows = [r for r in rows if str(r["subject"]) in val_subjects]
    return train_rows, val_rows


def split_train_val(rows, val_ratio=0.1, seed=42):
    idx = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(idx)
    n_val = int(len(rows) * val_ratio)
    val_idx = set(idx[:n_val].tolist())
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows
