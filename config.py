"""
Central configuration for the EmotionCLIP (SST-LegoViT + frozen CLIP) pipeline
on SEED-VII.

Mirrors the reference configs/SEED_train.yaml structure, but adapted to:
  * SEED-VII (7 emotion classes by default),
  * the migrated SEED-VII .mat / Saveinfo / two-level-text CSV protocols,
  * HuggingFace CLIP (offline-cacheable),
  * Kaggle resume + time-budget support (ported from the original repo).
"""

import os
from copy import deepcopy


def get_config(num_classes: int = 7) -> dict:
    cfg = {
        "random_seed": 42,
        "multi_gpu": False,

        # ----- data / paths (SEED-VII protocol, ported) -----
        "data": {
            "dataset": "SEED-VII",
            "num_classes": num_classes,          # 7 (SEED-VII) / 4 (SEED-IV) / 3 (SEED)
            "n_trials": 80,
            "num_shot": 0,                       # few-shot from target subject (0 = zero-shot)
            "data_root": "/kaggle/input/datasets/primocosmos/seed-vii-kaggle/EEG_features",
            "saveinfo_dir": "/kaggle/input/datasets/primocosmos/seed-vii-kaggle/save_info",
            "text_csv_path": "/kaggle/input/datasets/primocosmos/seed-vii-kaggle/Emotion2text/text_protocol_template.csv",
            "val_ratio": 0.1,
            "val_split_mode": "subject",         # "subject" | "window"
            "normalize_subject_zscore": True,
            "use_l2_extra_prompts": True,        # migrate CSV L2 trial text as extra prompts
            "workers": 2,
        },

        # ----- 4D topographic input + SST-LegoViT network -----
        "network": {
            "image_frames": 4,
            "image_channels": 10,                # 5 DE bands duplicated to 10 (DE/PSD slots)
            "image_height": 64,
            "image_width": 64,
            "tubelet_frames": 1,
            "tubelet_channels": 1,
            "tubelet_height": 16,
            "tubelet_width": 16,
            "num_transformer_layers": [2, 2, 2], # [spatial, spectral, temporal]
            "embed_dims": 128,
            "num_heads": 4,
            "multi_conv2d_hidden_dims": 128,
            "spatial_type": "Multi_Conv2D",
            "spectral_type": "Legoformer",
            "temporal_type": "Transformer",
            "attn_dropout": 0.0,
            "attn_proj_dropout": 0.0,
            "ffn_proj_dropout": 0.1,
            "multi_conv_dropout": 0.1,
            "drop_path_rate": 0.1,
            "dropout_after_pos_embed": 0.3,
            "conv_type": "Conv_Stem",
            "use_spectral_pos_embedding": False,
            "clip_embed_dim": 512,               # must match CLIP text projection dim
        },

        # ----- text tower -----
        "text": {
            "clip_name": "openai/clip-vit-base-patch16",
            "max_text_len": 64,
        },

        # ----- optimization -----
        "solver": {
            "train_batch_size": 64,
            "val_batch_size": 64,
            "num_epochs": 100,
            "eval_every_n_epochs": 1,            # run val+test every N epochs (speed)
            "is_early_patience": True,
            "early_patience": 30,
            "start_epoch": 0,
            "lr_type": "Cosine",
            "lr": 1e-4,
            "lr_warmup_step": 5,
            "momentum": 0.9,
            "weight_decay": 0.003,
            "optim": "AdamW",
            "loss_type": "KL",                   # "KL" | "CE"
            "aux_ce_weight": 0.0,                # optional supervised aux head weight
        },

        # ----- runtime / Kaggle (ported) -----
        "runtime": {
            "work_dir": "/kaggle/working/emotionclip_ckpt",
            "resume": True,
            "save_every_n_steps": 100,
            "max_train_hours": 8.8,
            "time_buffer_minutes": 8,
            "amp": False,
            "log_every_n_steps": 20,
            "run_all_folds": False,              # False = only first LOSO fold
        },
    }
    return cfg


def discover_data_paths(cfg: dict) -> dict:
    """Kaggle path auto-discovery (ported from the original repo)."""
    from pathlib import Path
    d = cfg["data"]
    data_root = d["data_root"]
    saveinfo_dir = d["saveinfo_dir"]
    text_csv_path = d["text_csv_path"]

    input_root = Path("/kaggle/input")
    if input_root.exists():
        feature_dirs = sorted(input_root.rglob("EEG_features"))
        if (not os.path.exists(data_root)) and feature_dirs:
            data_root = str(feature_dirs[0])
        if (not saveinfo_dir or not os.path.exists(saveinfo_dir)) and os.path.exists(data_root):
            parent = Path(data_root).parent
            for c in [parent / "save_info", parent / "Saveinfo", parent / "saveinfo"]:
                if c.exists():
                    saveinfo_dir = str(c)
                    break
        if not text_csv_path or not os.path.exists(text_csv_path):
            cands = []
            # 1) Prefer the conventional location next to the data.
            if os.path.exists(data_root):
                parent = Path(data_root).parent
                emo_dir = parent / "Emotion2text"
                if emo_dir.exists():
                    cands.extend(sorted(emo_dir.glob("*.csv")))
                cands.extend(sorted(parent.rglob("text_protocol*.csv")))
            # 2) Global fallback: search ALL mounted datasets. This covers the
            #    case where the code + CSV live in a SEPARATE Kaggle dataset from
            #    the EEG data. We match the known protocol filename pattern.
            if not cands:
                cands.extend(sorted(input_root.rglob("text_protocol*.csv")))
            if not cands:
                # last resort: any csv with the expected columns header.
                for p in sorted(input_root.rglob("*.csv")):
                    try:
                        head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
                        if head and "l1_text" in head[0] and "l2_text" in head[0]:
                            cands.append(p)
                            break
                    except Exception:
                        continue
            if cands:
                text_csv_path = str(cands[0])

    out = deepcopy(cfg)
    out["data"]["data_root"] = data_root
    out["data"]["saveinfo_dir"] = saveinfo_dir
    out["data"]["text_csv_path"] = text_csv_path
    return out
