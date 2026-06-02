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


# --------------------------------------------------------------------------- #
# Label scheme: configurable 7-class (fine) <-> 3-class valence aggregation.
# --------------------------------------------------------------------------- #
# Default valence mapping of the 7 SEED-VII emotions -> {negative, neutral, positive}.
# `surprise` is treated as positive by default (SEED-VII film-elicited surprise is
# mostly pleasant amazement); this is configurable via VALENCE_GROUPS below.
VALENCE_NAMES = ["negative", "neutral", "positive"]

VALENCE_GROUPS = {
    "negative": ["sadness", "fear", "disgust", "anger"],
    "neutral":  ["neutral"],
    "positive": ["joy", "surprise"],
}

# Class words used to render CLIP prompts for the 3-class valence scheme.
VALENCE_PROMPT_WORD = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def _fine_to_valence_index_map(valence_groups: Dict[str, List[str]]) -> List[int]:
    """Return a length-7 list mapping fine label idx -> valence label idx
    (index into VALENCE_NAMES)."""
    emo_to_val = {}
    for vname, emos in valence_groups.items():
        vi = VALENCE_NAMES.index(vname)
        for e in emos:
            emo_to_val[normalize_emotion_name(e)] = vi
    mapping = []
    for emo in EMOTION_NAMES:
        if emo not in emo_to_val:
            raise ValueError(f"Emotion '{emo}' is not assigned to any valence group.")
        mapping.append(emo_to_val[emo])
    return mapping


class LabelScheme:
    """Encapsulates the active label space and the fine(7)->active mapping.

    mode = "fine"    -> 7 classes (neutral, joy, sadness, fear, disgust, anger, surprise)
    mode = "valence" -> 3 classes (negative, neutral, positive), aggregated from fine.

    The data pipeline ALWAYS parses raw 7-class labels first; this scheme then
    maps them to the active label space. So switching modes never touches the
    Saveinfo/CSV parsing or the EEG data.
    """

    def __init__(self, mode: str = "fine", valence_groups: Optional[Dict[str, List[str]]] = None):
        mode = (mode or "fine").lower()
        if mode not in ("fine", "valence"):
            raise ValueError(f"label scheme mode must be 'fine' or 'valence', got {mode!r}")
        self.mode = mode
        self.valence_groups = valence_groups or VALENCE_GROUPS
        if mode == "valence":
            self.names = list(VALENCE_NAMES)
            self.prompt_word = dict(VALENCE_PROMPT_WORD)
            self._fine2active = _fine_to_valence_index_map(self.valence_groups)
        else:
            self.names = list(EMOTION_NAMES)
            self.prompt_word = dict(EMOTION_PROMPT_WORD)
            self._fine2active = list(range(len(EMOTION_NAMES)))

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def map_fine(self, fine_label: int) -> int:
        """Map a raw 7-class label index to the active label index."""
        fine_label = int(max(0, min(int(fine_label), len(EMOTION_NAMES) - 1)))
        return self._fine2active[fine_label]

    def class_prompt_words(self) -> List[str]:
        return [self.prompt_word[n] for n in self.names]

    def describe(self) -> str:
        if self.mode == "fine":
            return f"LabelScheme(mode=fine, {self.num_classes} classes: {self.names})"
        grp = {v: self.valence_groups[v] for v in VALENCE_NAMES}
        return (f"LabelScheme(mode=valence, {self.num_classes} classes: {self.names}; "
                f"groups={grp}")


def make_label_scheme(cfg_data: dict) -> "LabelScheme":
    """Build a LabelScheme from the cfg['data'] dict.

    Recognized keys:
      label_mode: "fine" | "valence"  (default "fine")
      valence_groups: optional {valence_name: [emotion,...]} override
    """
    mode = cfg_data.get("label_mode", "fine")
    groups = cfg_data.get("valence_groups", None)
    return LabelScheme(mode=mode, valence_groups=groups)


def class_prompt_words(num_classes: int = 7) -> List[str]:
    """Backward-compatible helper: return class words for the FINE scheme
    (truncated to num_classes). Prefer LabelScheme.class_prompt_words()."""
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


# --------------------------------------------------------------------------- #
# Robust .mat reading (v7 via scipy, v7.3 via h5py) + feature normalization.
# --------------------------------------------------------------------------- #
def _load_mat_any(path) -> Dict[str, np.ndarray]:
    """Load a .mat as {key: ndarray}, supporting BOTH classic (<=v7) and
    HDF5-based v7.3 files.

    scipy.io.loadmat cannot read MATLAB v7.3 (HDF5) files and raises
    NotImplementedError; SEED-VII subject files can be large and are sometimes
    saved as v7.3. We transparently fall back to h5py in that case. h5py stores
    arrays transposed vs MATLAB, so we transpose back to MATLAB column order.
    """
    path = str(path)
    try:
        m = sio.loadmat(path)  # classic v5/v6/v7
        return {k: v for k, v in m.items() if not k.startswith("__")}
    except NotImplementedError:
        pass  # -> v7.3, use h5py
    except Exception as e:
        raise RuntimeError(f"Failed to read .mat (classic reader): {path}: {e}") from e

    try:
        import h5py
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"{path} appears to be a MATLAB v7.3 (HDF5) file but h5py is not "
            f"installed. Run `pip install h5py` (preinstalled on Kaggle)."
        ) from e

    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if k.startswith("#"):
                continue
            try:
                arr = np.array(f[k])
            except Exception:
                continue
            # h5py returns data transposed relative to MATLAB; undo it.
            if arr.ndim >= 2:
                arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
            out[k] = arr
    return out


def _mat_get(mat: Dict[str, np.ndarray], key: str):
    """Case-insensitive, whitespace-tolerant key lookup into a loaded .mat dict."""
    if key in mat:
        return mat[key]
    low = key.lower()
    for k, v in mat.items():
        if k.lower() == low:
            return v
    return None


def _normalize_feat_TBN(arr: np.ndarray, n_bands: int = 5, n_chan: int = 62) -> Optional[np.ndarray]:
    """Coerce a per-trial feature array to canonical shape (T, n_bands, n_chan).

    Handles: leading/trailing singleton dims, and (5,62,T)/(T,62,5)/(62,5,T)
    axis permutations by locating the band(=5) and channel(=62) axes. Returns
    float32 with NaN/Inf replaced by 0; returns None if it can't be coerced.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        # (5,62) single window -> (1,5,62); or (62,5) -> transpose
        if arr.shape == (n_bands, n_chan):
            arr = arr[np.newaxis, ...]
        elif arr.shape == (n_chan, n_bands):
            arr = arr.T[np.newaxis, ...]
        else:
            return None
    if arr.ndim != 3:
        return None

    # Identify which axes correspond to bands(5) and channels(62).
    shape = arr.shape
    band_axis = next((i for i, s in enumerate(shape) if s == n_bands), None)
    chan_axis = next((i for i, s in enumerate(shape) if s == n_chan and i != band_axis), None)
    if band_axis is None or chan_axis is None:
        return None
    time_axis = ({0, 1, 2} - {band_axis, chan_axis}).pop()
    arr = np.transpose(arr, (time_axis, band_axis, chan_axis))  # (T, 5, 62)

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


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


def feat_window_to_4d(de_window: np.ndarray, image_frames: int, image_channels: int,
                      image_height: int, image_width: int,
                      psd_window: "np.ndarray | None" = None,
                      use_psd: bool = True) -> np.ndarray:
    """Convert one window's DE (and PSD) features -> 4D tensor (frames, C, H, W).

    SEED-VII provides BOTH DE and PSD per window, each shaped (5, 62) (5 freq
    bands x 62 electrodes). To match the official EmotionCLIP "DE_PSD" input
    (image_channels=10) we build the spectral/channel axis as:

    [ DE_band0..DE_band4 , PSD_band0..PSD_band4 ] (DE first, PSD second)

    This ordering is intentional: the Legoformer dual-stream splits the channel
    axis in half (`x[:, :C//2]` = DE stream, `x[:, C//2:]` = PSD stream), so the
    first half MUST be DE and the second half PSD -- exactly the official
    "DE first, PSD second" layout (their indices=[0,2,4,6,8, 1,3,5,7,9]).

    Each (62,) band vector is mapped to a 9x9 scalp topographic grid and
    bicubic-resized to (H, W).

    Fallbacks:
      * If `psd_window` is None or `use_psd=False`, the DE maps are duplicated to
        fill the requested channels (legacy behaviour; logged once upstream).
      * The single window is replicated across `image_frames` temporal frames
        (SEED-VII DE/PSD are window-level aggregates without intra-window frames).
    de_window / psd_window: shape (5, 62)
    """
    size = (image_width, image_height)
    de_maps = np.stack([_band_vector_to_2d(de_window[b], size)
                        for b in range(de_window.shape[0])], axis=0)  # (5,H,W)

    if use_psd and psd_window is not None:
        psd_maps = np.stack([_band_vector_to_2d(psd_window[b], size)
                             for b in range(psd_window.shape[0])], axis=0)  # (5,H,W)
        chan = np.concatenate([de_maps, psd_maps], axis=0)  # (10,H,W) DE|PSD
    else:
        chan = de_maps  # (5,H,W)

    # Fit to requested image_channels (truncate or tile to be safe).
    if chan.shape[0] >= image_channels:
        chan = chan[:image_channels]
    else:
        reps = int(np.ceil(image_channels / chan.shape[0]))
        chan = np.concatenate([chan] * reps, axis=0)[:image_channels]

    frames = np.repeat(chan[np.newaxis, ...], image_frames, axis=0)  # (frames,C,H,W)
    return frames.astype(np.float32)


# Backward-compatible alias (DE-only) kept for any external callers.
def de_window_to_4d(de_window, image_frames, image_channels, image_height, image_width):
    return feat_window_to_4d(de_window, image_frames, image_channels,
                             image_height, image_width, psd_window=None, use_psd=False)


# --------------------------------------------------------------------------- #
# Subject window loading (ported + z-score), now storing raw DE windows.
# --------------------------------------------------------------------------- #
def load_subject_windows(subject_file: Path, trial_labels: np.ndarray,
                         normalize_subject_zscore: bool = True,
                         label_scheme: Optional["LabelScheme"] = None,
                         de_key: str = "de", use_psd: bool = True) -> List[Dict]:
    """Load per-window DE (and PSD) features for one subject .mat.

    SEED-VII .mat files contain, per trial i (1..80):
        de_i      (T,5,62)  raw differential entropy
        de_LDS_i  (T,5,62)  LDS-smoothed DE
        psd_i     (T,5,62)  power spectral density
    de_key selects "de_LDS" (LDS-smoothed, recommended) or "de" (raw). use_psd
    loads psd_i too. Robust to MATLAB v7.3 files, axis permutations, singleton
    dims, case differences, and NaN/Inf. DE and PSD are z-scored INDEPENDENTLY
    (per subject) before being stacked into the DE/PSD dual-stream channel axis.
    """
    mat = _load_mat_any(subject_file)
    have_psd = use_psd and any(_mat_get(mat, f"psd_{i}") is not None
                               for i in range(1, len(trial_labels) + 1))

    trial_data = []  # (trial_idx, de(T,5,62), psd or None, label)
    for trial_idx in range(1, len(trial_labels) + 1):
        raw = _mat_get(mat, f"{de_key}_{trial_idx}")
        if raw is None:                          # fall back to raw de_ if de_LDS_ missing
            raw = _mat_get(mat, f"de_{trial_idx}")
        if raw is None:
            continue
        de = _normalize_feat_TBN(raw)            # -> (T,5,62) float32, NaN-safe
        if de is None or de.shape[0] == 0:
            continue
        psd = None
        if have_psd:
            praw = _mat_get(mat, f"psd_{trial_idx}")
            if praw is not None:
                psd = _normalize_feat_TBN(praw)
                if psd is not None and psd.shape[0] != de.shape[0]:
                    # window-count mismatch -> crop both to the shorter (safe)
                    T = min(psd.shape[0], de.shape[0])
                    de, psd = de[:T], psd[:T]
        fine = int(max(0, min(int(trial_labels[trial_idx - 1]), len(EMOTION_NAMES) - 1)))
        label = label_scheme.map_fine(fine) if label_scheme is not None else fine
        trial_data.append((trial_idx, de, psd, label))

    # Per-subject z-score, DE and PSD normalized separately.
    if normalize_subject_zscore and trial_data:
        all_de = np.concatenate([x[1] for x in trial_data], axis=0)       # (N,5,62)
        de_mu = all_de.mean(axis=0, keepdims=True)
        de_sd = np.clip(all_de.std(axis=0, keepdims=True), 1e-6, None)
        psd_list = [x[2] for x in trial_data if x[2] is not None]
        if psd_list:
            all_psd = np.concatenate(psd_list, axis=0)
            psd_mu = all_psd.mean(axis=0, keepdims=True)
            psd_sd = np.clip(all_psd.std(axis=0, keepdims=True), 1e-6, None)
        normed = []
        for ti, de, psd, lb in trial_data:
            de_n = (de - de_mu) / de_sd
            psd_n = ((psd - psd_mu) / psd_sd) if (psd is not None) else None
            normed.append((ti, de_n, psd_n, lb))
        trial_data = normed

    samples = []
    for trial_idx, de, psd, label in trial_data:
        for t in range(de.shape[0]):
            samples.append({
                "subject": subject_file.stem,
                "trial": trial_idx,
                "label": label,
                "de": de[t],                              # (5,62)
                "psd": psd[t] if psd is not None else None,  # (5,62) or None
            })
    return samples


def load_all_samples_with_saveinfo(data_root, saveinfo_dir=None, n_trials=80,
                                   normalize_subject_zscore=True,
                                   label_scheme: Optional["LabelScheme"] = None,
                                   de_key: str = "de", use_psd: bool = True) -> List[Dict]:
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
    if label_scheme is not None:
        print(f"label scheme: {label_scheme.describe()}")
    print(f"feature config: de_key='{de_key}', use_psd={use_psd}")

    fallback_labels = None
    all_samples = []
    n_with_psd = 0
    for sf in subject_files:
        sid = _extract_subject_id(sf.stem)
        labels = subject_label_map.get(sid)
        if labels is None:
            if fallback_labels is None:
                fallback_labels = load_trial_labels(data_root, n_trials)
            labels = fallback_labels
        rows = load_subject_windows(sf, labels, normalize_subject_zscore,
                                    label_scheme=label_scheme, de_key=de_key, use_psd=use_psd)
        n_with_psd += sum(1 for r in rows if r.get("psd") is not None)
        all_samples.extend(rows)
    if all_samples:
        frac = n_with_psd / len(all_samples)
        if use_psd:
            print(f"PSD available for {n_with_psd}/{len(all_samples)} windows "
                  f"({frac*100:.0f}%) -> DE+PSD dual-stream input"
                  + ("" if frac > 0.99 else "  [WARN] some windows fall back to DE-only"))
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
        # PSD if every row has it (DE+PSD dual-stream); else None (DE-only).
        if all(r.get("psd") is not None for r in rows) and len(rows) > 0:
            self.psd = torch.from_numpy(
                np.ascontiguousarray(np.stack([r["psd"] for r in rows], axis=0))
            ).to(torch.float32)  # (N, 5, 62)
        else:
            self.psd = None

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        de = self.de[idx].numpy()  # (5,62)
        psd = self.psd[idx].numpy() if self.psd is not None else None
        img = feat_window_to_4d(de, self.image_frames, self.image_channels,
                                self.image_height, self.image_width,
                                psd_window=psd, use_psd=(psd is not None))  # (frames,C,H,W)
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


# --------------------------------------------------------------------------- #
# Balanced Batch Sampler (NEW: for Supervised Contrastive Learning)
# --------------------------------------------------------------------------- #
class BalancedBatchSampler:
    """Strictly balanced batch sampler for supervised contrastive learning.

    Guarantees every batch contains exactly `samples_per_class` samples from
    each class. This ensures:
    1. Every batch has a balanced class distribution matching the active
       label scheme (fine=7 or valence=3).
    2. Each sample has enough positive pairs for stable SupCon gradients.
    3. Eliminates random batch-to-batch class fluctuations that cause
       gradient dominance by majority classes.

    Requirements:
    - batch_size MUST be divisible by num_classes
    - Each class MUST have >= samples_per_class training examples
    """

    def __init__(self, labels, batch_size, num_classes):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes

        assert batch_size % num_classes == 0, (
            f"batch_size ({batch_size}) must be divisible by num_classes ({num_classes}). "
            f"For 7-class use multiples of 7 (e.g. 504=72*7). "
            f"For 3-class valence use multiples of 3 (e.g. 510=170*3)."
        )
        self.samples_per_class = batch_size // num_classes

        # Verify each class has enough samples for at least one batch
        for c in range(num_classes):
            count = int(np.sum(self.labels == c))
            if count < self.samples_per_class:
                raise ValueError(
                    f"Class {c} has only {count} samples, "
                    f"need at least samples_per_class={self.samples_per_class}. "
                    f"Reduce batch_size or check label distribution."
                )

        # Number of complete batches we can form
        min_per_class = min(int(np.sum(self.labels == c)) for c in range(num_classes))
        self.n_batches = min_per_class // self.samples_per_class

    def __iter__(self):
        # Build per-class index pools
        indices = {}
        for c in range(self.num_classes):
            indices[c] = np.where(self.labels == c)[0].tolist()

        rng = np.random.default_rng()

        for _ in range(self.n_batches):
            batch = []
            for c in range(self.num_classes):
                # Shuffle this class's pool and take samples_per_class
                rng.shuffle(indices[c])
                batch.extend(indices[c][:self.samples_per_class])
                # Remove taken indices
                indices[c] = indices[c][self.samples_per_class:]

            # Shuffle the entire batch so the model doesn't learn positional bias
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
