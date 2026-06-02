"""
EmotionCLIP training / evaluation on SEED-VII.

Technical path (Yan et al., 2025 / EmotionCLIP, official repo Departure2021/EmotionCLIP):
  * EEG tower = SST-LegoViT (trainable), produces CLIP-space embeddings.
  * Text tower = frozen CLIP text encoder with prompt ensemble (precomputed).
  * Contrastive/class-prototype alignment: normalized EEG embedding is compared
    with frozen class text features. Similarities are reshaped to
    (B, num_text_aug, num_classes), averaged across templates as RAW LOGITS,
    temperature-scaled, and trained with standard mean CrossEntropyLoss.
    Only the EEG tower is optimized.
  * Cross-subject evaluation (LOSO) over SEED-VII subjects.

Protocols MIGRATED from the original repo: SEED-VII .mat loading, Saveinfo
label parsing, two-level text CSV (L2 trial text reused as extra prompts),
per-subject z-score, subject-disjoint train/val split, Kaggle resume +
time-budget checkpointing.
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config, discover_data_paths
from data_seedvii import (
    EMOTION_NAMES, class_prompt_words, make_label_scheme,
    load_all_samples_with_saveinfo, load_two_level_texts_from_csv,
    build_subject_label_map_from_saveinfo, build_default_l2_texts,
    EEGTopoDataset, collate_fn, split_train_val, split_train_val_by_subject,
    BalancedBatchSampler,
)
from sst_legovit import create_eeg_encoder
from text_tower import EmotionTextTower, KLLoss, build_extra_class_prompts_from_l2


# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _extract_subject_id_from_stem(stem: str):
    """Extract integer subject id from a subject file stem (e.g. 'subject_3' -> 3)."""
    import re
    nums = re.findall(r"\d+", str(stem))
    return int(nums[0]) if nums else None


def _build_trial_label_lookup(sub_label_map, allowed_subject_ids, n_trials, label_scheme=None):
    """Majority-vote trial->label mapping using ONLY the allowed (training) subjects.

    `sub_label_map` holds raw FINE (7-class) labels parsed from Saveinfo. The
    majority vote is done in fine space, then mapped to the ACTIVE label space
    (fine or aggregated valence) via `label_scheme` so the L2 prompt assignment
    matches the classes the model is trained on.

    This keeps the L2 prompt assignment strictly leakage-free under LOSO: the
    held-out test subject's Saveinfo is excluded from the vote.
    """
    arrs = [v for sid, v in sub_label_map.items()
            if (allowed_subject_ids is None or sid in allowed_subject_ids) and len(v) >= n_trials]
    lookup = {}
    if arrs:
        stacked = np.stack([a[:n_trials] for a in arrs], axis=0)
        for t in range(n_trials):
            vals, counts = np.unique(stacked[:, t], return_counts=True)
            fine = int(vals[np.argmax(counts)])
            lookup[t + 1] = label_scheme.map_fine(fine) if label_scheme is not None else fine
    return lookup


def _save_ckpt(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_ckpt(path):
    return torch.load(path, map_location="cpu") if os.path.exists(path) else None


# --------------------------------------------------------------------------- #
# Supervised Contrastive Loss (NEW)
# --------------------------------------------------------------------------- #
def supervised_contrastive_loss(features, labels, temperature=0.07):
    """Supervised Contrastive Loss (SupCon).

    Enforces:
    - Same emotion (even from different subjects) = positive pair (attract)
    - Different emotion (even from same subject)  = negative pair (repel)

    Automatically adapts to the active label scheme (fine=7 or valence=3).

    Args:
        features: [B, D] L2-normalized embeddings from the EEG tower
        labels:   [B]    integer class labels (already mapped to active scheme)
        temperature: scaling factor (typically 0.05~0.1)
    """
    # features should already be L2-normalized by the caller
    # Compute pairwise cosine similarity matrix [B, B]
    sim_matrix = features @ features.T / temperature

    # Positive mask: same class AND not self
    labels = labels.unsqueeze(1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0.0)  # exclude self

    # Numerically stable denominator: log-sum-exp over all negatives+positives
    logits_mask = sim_matrix.clone()
    logits_mask.fill_diagonal_(float('-inf'))
    log_denom = torch.logsumexp(logits_mask, dim=1)

    # Numerator: sum of positive similarities
    pos_logits_sum = (sim_matrix * pos_mask).sum(dim=1)
    pos_counts = pos_mask.sum(dim=1)
    pos_counts = torch.clamp(pos_counts, min=1.0)  # prevent division by zero

    # Loss = - mean(log( exp(sim_pos) / sum(exp(sim_all)) ))
    loss_per_sample = (pos_logits_sum / pos_counts) - log_denom
    return -loss_per_sample.mean()


# --------------------------------------------------------------------------- #
def build_loaders(train_rows, val_rows, test_rows, cfg):
    n = cfg["network"]
    args = dict(image_frames=n["image_frames"], image_channels=n["image_channels"],
                image_height=n["image_height"], image_width=n["image_width"])
    train_ds = EEGTopoDataset(train_rows, **args)
    val_ds = EEGTopoDataset(val_rows, **args) if val_rows else None
    test_ds = EEGTopoDataset(test_rows, **args)

    pin = torch.cuda.is_available()
    nw = cfg["data"]["workers"]
    extra = {"prefetch_factor": 4, "persistent_workers": True} if nw > 0 else {}

    use_contrastive = cfg["solver"].get("use_contrastive_loss", False)
    batch_size = cfg["solver"]["train_batch_size"]

    if use_contrastive:
        # BalancedBatchSampler guarantees strict per-class balance in every batch.
        train_labels = [int(r["label"]) for r in train_rows]
        num_classes = cfg["data"]["num_classes"]
        sampler = BalancedBatchSampler(train_labels, batch_size, num_classes)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=nw, pin_memory=pin, collate_fn=collate_fn, **extra)
        print(f"[LOADER] BalancedBatchSampler: batch_size={batch_size}, "
              f"samples_per_class={batch_size // num_classes}, "
              f"n_batches={len(sampler)}")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=pin, collate_fn=collate_fn,
                                  drop_last=True, **extra)

    val_loader = (DataLoader(val_ds, batch_size=cfg["solver"]["val_batch_size"], shuffle=False,
                             num_workers=nw, pin_memory=pin, collate_fn=collate_fn, **extra)
                  if val_ds else None)
    test_loader = DataLoader(test_ds, batch_size=cfg["solver"]["val_batch_size"], shuffle=False,
                             num_workers=nw, pin_memory=pin, collate_fn=collate_fn, **extra)
    return train_loader, val_loader, test_loader


def compute_class_logits(image_emb, text_features, num_text_aug, num_classes, temperature: float = 0.1):
    """Compute class-level raw logits for EEG -> frozen text-prototype alignment.

    IMPORTANT: this returns RAW LOGITS suitable for nn.CrossEntropyLoss.  The old
    implementation returned per-template softmax probabilities and then fed them
    to CrossEntropyLoss, which caused a double-softmax/probability-as-logit bug.

    Steps:
      1) L2-normalize EEG and text embeddings -> cosine similarities.
      2) Reshape [B, aug*num_classes] -> [B, aug, num_classes].
      3) Average prompt-template logits.
      4) Divide by a temperature tau to control logit sharpness.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    image_emb = F.normalize(image_emb, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sim = image_emb @ text_features.t()                              # [B, aug*cls]
    logits = sim.view(image_emb.shape[0], num_text_aug, num_classes).mean(dim=1)
    return logits / temperature                                      # [B, num_classes]


def compute_class_probs(image_emb, text_features, num_text_aug, num_classes, temperature: float = 0.1):
    """Return calibrated class probabilities from raw logits.

    Kept as a convenience wrapper for evaluation/diagnostics. Training should
    use compute_class_logits(...) + CrossEntropyLoss(labels).
    """
    return compute_class_logits(image_emb, text_features, num_text_aug, num_classes,
                                temperature=temperature).softmax(dim=-1)


@torch.no_grad()
def evaluate(eeg_model, text_features, num_text_aug, loader, device, num_classes, temperature: float = 0.1):
    """Inference: raw temperature-scaled cosine logits, then argmax."""
    eeg_model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct, total = 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        emb = eeg_model.encode_image(images)
        logits = compute_class_logits(emb, text_features, num_text_aug, num_classes,
                                      temperature=temperature)
        preds = logits.argmax(dim=1)
        for p, l in zip(preds.tolist(), labels.tolist()):
            confusion[l][p] += 1
            correct += int(p == l)
            total += 1
    acc = correct / max(total, 1)
    return acc, confusion


def metrics_from_confusion(confusion):
    nc = confusion.shape[0]
    precision = np.zeros(nc); recall = np.zeros(nc); f1 = np.zeros(nc)
    for i in range(nc):
        if confusion[:, i].sum() > 0:
            precision[i] = confusion[i, i] / confusion[:, i].sum()
        if confusion[i, :].sum() > 0:
            recall[i] = confusion[i, i] / confusion[i, :].sum()
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    return precision, recall, f1


# --------------------------------------------------------------------------- #
def train_one_fold(train_rows, val_rows, test_rows, fold_name, cfg, device,
                   text_features, num_text_aug):
    runtime = cfg["runtime"]
    solver = cfg["solver"]
    num_classes = cfg["data"]["num_classes"]
    ensure_dir(runtime["work_dir"])

    train_loader, val_loader, test_loader = build_loaders(train_rows, val_rows, test_rows, cfg)
    print(f"[{fold_name}] steps: train={len(train_loader)}, "
          f"val={len(val_loader) if val_loader else 0}, test={len(test_loader)}")

    eeg_model = create_eeg_encoder(cfg).to(device)
    n_params = sum(p.numel() for p in eeg_model.parameters() if p.requires_grad) / 1e6
    print(f"[{fold_name}] EEG tower trainable params: {n_params:.3f}M")

    # --- Loss function: Joint CE + Supervised Contrastive ---
    use_contrastive = solver.get("use_contrastive_loss", False)
    alpha = float(solver.get("ce_supcon_alpha", 0.5))  # CE weight; (1-alpha) = SupCon weight
    ce_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    eval_temperature = float(solver.get("temperature", 0.1))
    supcon_temperature = float(solver.get("contrastive_temperature", 0.07))

    if use_contrastive:
        print(f"[*] Joint Loss: L = {alpha:.2f} * CE + {1 - alpha:.2f} * SupCon")
        print(f"[*] CE temperature = {eval_temperature}, SupCon temperature = {supcon_temperature}")
        if alpha == 1.0:
            print("[*] Warning: alpha=1.0 => pure CE, SupCon disabled despite use_contrastive_loss=True")
        elif alpha == 0.0:
            print("[*] Warning: alpha=0.0 => pure SupCon, CE disabled")
    else:
        print(f"[*] Using CrossEntropy Loss only (tau={eval_temperature})")

    # Only the EEG image tower is optimized; the text tower is frozen (official).
    optimizer = torch.optim.AdamW(eeg_model.parameters(),
                                  betas=(0.9, 0.98), eps=1e-8,
                                  lr=solver["lr"], weight_decay=solver["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=solver["num_epochs"], eta_min=0)
    scaler = torch.amp.GradScaler("cuda", enabled=(runtime["amp"] and device.type == "cuda"))

    latest_ckpt = os.path.join(runtime["work_dir"], f"latest_{fold_name}.pt")
    best_ckpt = os.path.join(runtime["work_dir"], f"best_{fold_name}.pt")
    history_path = os.path.join(runtime["work_dir"], f"history_{fold_name}.json")

    start_epoch = solver["start_epoch"]
    global_step = 0
    best_acc = 0.0
    history: List[Dict] = []

    loaded = _load_ckpt(latest_ckpt) if runtime["resume"] else None
    if loaded is not None:
        eeg_model.load_state_dict(loaded["eeg_model"])
        try:
            optimizer.load_state_dict(loaded["optimizer"])
        except Exception as e:
            print("[Resume] optimizer state incompatible:", e)
        if loaded.get("scheduler"):
            try:
                scheduler.load_state_dict(loaded["scheduler"])
            except Exception:
                pass
        if loaded.get("scaler"):
            scaler.load_state_dict(loaded["scaler"])
        start_epoch = int(loaded.get("epoch_completed", 0)) + 1
        global_step = int(loaded.get("global_step", 0))
        best_acc = float(loaded.get("best_acc", 0.0))
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        print(f"[Resume] {fold_name}: epoch={start_epoch}, step={global_step}, best_acc={best_acc:.4f}")

    text_features = text_features.to(device)

    start_ts = time.monotonic()
    stop_ts = start_ts + runtime["max_train_hours"] * 3600.0 - runtime["time_buffer_minutes"] * 60
    timed_out = False
    patience = 0

    for epoch in range(start_epoch, solver["num_epochs"]):
        eeg_model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0

        for batch in train_loader:
            if time.monotonic() >= stop_ts:
                timed_out = True
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(runtime["amp"] and device.type == "cuda")):
                emb = eeg_model.encode_image(images)

                if use_contrastive:
                    # Joint Loss: L = alpha * L_ce + (1 - alpha) * L_supcon
                    features_norm = F.normalize(emb, dim=-1)
                    logits = compute_class_logits(emb, text_features, num_text_aug, num_classes,
                                                  temperature=eval_temperature)
                    loss_ce = ce_loss_fn(logits, labels)
                    loss_supcon = supervised_contrastive_loss(features_norm, labels,
                                                               temperature=supcon_temperature)
                    loss = alpha * loss_ce + (1 - alpha) * loss_supcon
                else:
                    # Standard CE only
                    logits = compute_class_logits(emb, text_features, num_text_aug, num_classes,
                                                  temperature=eval_temperature)
                    loss = ce_loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            # --- Metrics tracking (compatible with both loss modes) ---
            run_loss += loss.item() * labels.shape[0]
            if use_contrastive:
                # In joint mode, also log per-component losses for diagnostics
                run_loss_ce = loss_ce.item() * labels.shape[0]
                run_loss_supcon = loss_supcon.item() * labels.shape[0]
            preds = logits.argmax(dim=1)
            run_correct += (preds == labels).sum().item()
            run_total += labels.shape[0]

            if global_step % runtime["save_every_n_steps"] == 0:
                _save_ckpt(latest_ckpt, {
                    "epoch_completed": epoch - 1, "global_step": global_step,
                    "eeg_model": eeg_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "best_acc": best_acc})

        scheduler.step()
        train_acc = run_correct / max(run_total, 1)
        train_loss = run_loss / max(run_total, 1)

        # Log per-component losses in joint mode for monitoring
        if use_contrastive:
            avg_loss_ce = run_loss_ce / max(run_total, 1)
            avg_loss_supcon = run_loss_supcon / max(run_total, 1)
        else:
            avg_loss_ce = train_loss
            avg_loss_supcon = float('nan')

        # Evaluation (val+test) roughly doubles per-epoch cost, so run it every
        # `eval_every_n_epochs` epochs (and always on the final epoch / timeout).
        eval_every = max(1, int(solver.get("eval_every_n_epochs", 1)))
        is_last = (epoch + 1 == solver["num_epochs"]) or timed_out
        do_eval = ((epoch + 1) % eval_every == 0) or is_last

        if do_eval:
            val_acc, _ = (evaluate(eeg_model, text_features, num_text_aug, val_loader, device, num_classes,
                                   temperature=eval_temperature)
                          if val_loader else (float("nan"), None))
            test_acc, test_cm = evaluate(eeg_model, text_features, num_text_aug, test_loader, device, num_classes,
                                         temperature=eval_temperature)
        else:
            val_acc, test_acc, test_cm = float("nan"), float("nan"), None

        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
               "val_acc": val_acc, "test_acc": test_acc, "global_step": global_step,
               "train_loss_ce": avg_loss_ce, "train_loss_supcon": avg_loss_supcon}
        history.append(row)
        if use_contrastive:
            print(f"[{fold_name}] E{epoch+1:03d} | loss={train_loss:.4f} "
                  f"ce={avg_loss_ce:.4f} supcon={avg_loss_supcon:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")
        else:
            print(f"[{fold_name}] E{epoch+1:03d} | loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # Only update best / patience on epochs where evaluation actually ran.
        monitor = val_acc if val_loader else test_acc
        if do_eval and not np.isnan(monitor):
            if monitor > best_acc:
                best_acc = monitor
                patience = 0
                p, r, f1 = metrics_from_confusion(test_cm)
                print(f"[{fold_name}]  * new best ({'val' if val_loader else 'test'}_acc={best_acc:.4f}) "
                      f"| test mF1={np.mean(f1):.4f}")
                _save_ckpt(best_ckpt, {"epoch": epoch + 1, "best_acc": best_acc,
                                       "eeg_model": eeg_model.state_dict(),
                                       "test_confusion": test_cm.tolist(), "cfg": cfg})
            else:
                patience += 1

        _save_ckpt(latest_ckpt, {
            "epoch_completed": epoch, "global_step": global_step,
            "eeg_model": eeg_model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_acc": best_acc})

        if timed_out:
            print(f"[{fold_name}] time budget reached, saved checkpoint, exiting.")
            break
        if solver["is_early_patience"] and patience >= solver["early_patience"]:
            print(f"[{fold_name}] early stopping (best_acc={best_acc:.4f}).")
            break

    return {"fold": fold_name, "best_acc": best_acc, "latest_ckpt": latest_ckpt,
            "best_ckpt": best_ckpt, "history_path": history_path, "timed_out": timed_out}


# --------------------------------------------------------------------------- #
def run_loso(cfg: dict):
    seed_everything(cfg["random_seed"])
    cfg = discover_data_paths(cfg)
    runtime = cfg["runtime"]
    ensure_dir(runtime["work_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    d = cfg["data"]
    # Build the (configurable) label scheme: fine 7-class or aggregated valence 3-class.
    label_scheme = make_label_scheme(d)
    num_classes = label_scheme.num_classes
    d["num_classes"] = num_classes  # keep cfg consistent with the active scheme
    print("resolved data_root   :", d["data_root"])
    print("resolved saveinfo_dir:", d["saveinfo_dir"])
    print("resolved text_csv    :", d["text_csv_path"])
    print(label_scheme.describe())

    # ---- text protocol (CSV migrated): L2 trial descriptions ----
    # NOTE on data-leakage: the trial->emotion mapping used to assign L2 text to
    # classes is rebuilt PER FOLD from TRAINING subjects' Saveinfo only. We never
    # touch the held-out test subject's files when constructing the prompts, so
    # the frozen-text prompt ensemble (and thus text_features) stays leakage-free.
    l2_texts = None
    sub_label_map = None
    if d["text_csv_path"] and os.path.exists(d["text_csv_path"]):
        _, l2_texts = load_two_level_texts_from_csv(d["text_csv_path"], d["n_trials"])
        sub_label_map = build_subject_label_map_from_saveinfo(
            d["data_root"], d["saveinfo_dir"], d["n_trials"])
    else:
        print("text CSV not found; using template prompts only.")

    # ---- text tower (frozen CLIP) ----
    text_tower = EmotionTextTower(cfg["text"]["clip_name"], cfg["text"]["max_text_len"]).to(device)
    assert text_tower.clip_embed_dim == cfg["network"]["clip_embed_dim"], (
        f"clip_embed_dim mismatch: CLIP={text_tower.clip_embed_dim}, "
        f"cfg={cfg['network']['clip_embed_dim']}")
    words = label_scheme.class_prompt_words()

    # ---- load EEG samples (labels already mapped to the active scheme) ----
    all_rows = load_all_samples_with_saveinfo(
        d["data_root"], d["saveinfo_dir"], d["n_trials"], d["normalize_subject_zscore"],
        label_scheme=label_scheme, de_key=d.get("de_key", "de"), use_psd=d.get("use_psd", True))
    subjects = sorted({r["subject"] for r in all_rows})
    print("subjects =", len(subjects), "| total windows =", len(all_rows))

    targets = subjects if runtime["run_all_folds"] else subjects[:1]
    fold_results = []
    for test_sub in targets:
        train_pool = [r for r in all_rows if r["subject"] != test_sub]
        test_rows = [r for r in all_rows if r["subject"] == test_sub]
        if d["val_split_mode"] == "subject":
            train_rows, val_rows = split_train_val_by_subject(train_pool, d["val_ratio"], cfg["random_seed"])
        else:
            train_rows, val_rows = split_train_val(train_pool, d["val_ratio"], cfg["random_seed"])

        leak = {r["subject"] for r in train_rows} & {r["subject"] for r in val_rows}
        fold_name = f"loso_test_{test_sub}"
        print(f"\n=== {fold_name} ===")
        print("train/val/test =", len(train_rows), len(val_rows), len(test_rows),
              "| subject_overlap(train,val) =", len(leak))

        # -- per-fold, leakage-free prompt ensemble --
        # trial->label majority vote computed ONLY from training subjects.
        train_subject_ids = {_extract_subject_id_from_stem(s) for s in
                             {r["subject"] for r in train_pool}}
        extra_prompts = None
        if d["use_l2_extra_prompts"] and l2_texts and sub_label_map:
            trial_label_lookup = _build_trial_label_lookup(
                sub_label_map, train_subject_ids, d["n_trials"], label_scheme=label_scheme)
            extra_prompts = build_extra_class_prompts_from_l2(
                l2_texts, trial_label_lookup=trial_label_lookup, num_classes=num_classes)
            print("L2 extra prompt classes (train-only):",
                  None if extra_prompts is None else {k: len(v) for k, v in extra_prompts.items()})

        text_features, num_text_aug = text_tower.build_class_text_features(
            words, device, normalize=True, extra_class_prompts=extra_prompts)
        print(f"[{fold_name}] num_text_aug={num_text_aug} | "
              f"text_features={tuple(text_features.shape)}")

        res = train_one_fold(train_rows, val_rows, test_rows, fold_name, cfg, device,
                             text_features, num_text_aug)
        fold_results.append(res)

    result_path = os.path.join(runtime["work_dir"], "loso_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)
    print("\nSaved:", result_path)
    if fold_results:
        accs = [r["best_acc"] for r in fold_results]
        print(f"Mean best acc over {len(accs)} fold(s): {np.mean(accs):.4f}")
    return fold_results


if __name__ == "__main__":
    # label_mode="fine" -> 7-class; label_mode="valence" -> 3-class (neg/neu/pos)
    config = get_config(label_mode="fine")
    run_loso(config)
