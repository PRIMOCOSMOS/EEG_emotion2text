import json
import os
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from eeg2text_core import (
    CFG,
    EEGEncoder,
    EEGTextWindowDataset,
    TextTower,
    asymmetric_contrastive_loss,
    build_trial_texts,
    collate_fn,
    ensure_work_dir,
    load_all_samples_with_saveinfo,
    load_trial_texts_from_csv,
    seed_everything,
    split_train_val,
)


def build_loaders(
    train_rows: List[Dict], val_rows: List[Dict], cfg: CFG, trial_texts: Dict[int, Dict[str, str]]
) -> Tuple[DataLoader, DataLoader]:
    train_ds = EEGTextWindowDataset(train_rows, trial_texts=trial_texts)
    val_ds = EEGTextWindowDataset(val_rows, trial_texts=trial_texts)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


def _save_ckpt(path: str, state: Dict) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_ckpt_if_exists(path: str):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def run_epoch(
    train_mode: bool,
    eeg_model: EEGEncoder,
    text_model: TextTower,
    loader: DataLoader,
    optimizer,
    scaler,
    cfg: CFG,
    device: torch.device,
    start_ts: float,
    stop_ts: float,
    epoch: int,
    global_step: int,
    latest_ckpt_path: str,
):
    eeg_model.train(mode=train_mode)
    text_model.train(mode=train_mode)

    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0
    timed_out = False

    for batch in loader:
        now = time.monotonic()
        if train_mode and now >= stop_ts:
            timed_out = True
            break

        eeg = batch["eeg"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                eeg_z = eeg_model(eeg)
                txt_z = text_model(batch["text_l1"], batch["text_l2"], batch["text_l3"], device)
                loss, metrics = asymmetric_contrastive_loss(eeg_z, txt_z, cfg.temperature, cfg.alpha)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                if global_step % cfg.save_every_n_steps == 0:
                    _save_ckpt(
                        latest_ckpt_path,
                        {
                            "epoch_completed": epoch - 1,
                            "global_step": global_step,
                            "eeg_model": eeg_model.state_dict(),
                            "text_proj": text_model.proj.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "elapsed_seconds": time.monotonic() - start_ts,
                        },
                    )

        total_loss += loss.item()
        total_acc += metrics["batch_retrieval_acc"]
        n_steps += 1

    if n_steps == 0:
        stats = {"loss": float("nan"), "retrieval_acc": float("nan")}
    else:
        stats = {"loss": total_loss / n_steps, "retrieval_acc": total_acc / n_steps}

    return stats, timed_out, global_step


def train_one_fold(
    train_rows: List[Dict],
    val_rows: List[Dict],
    fold_name: str,
    cfg: CFG,
    device: torch.device,
    trial_texts: Dict[int, Dict[str, str]],
):
    ensure_work_dir(cfg.work_dir)
    train_loader, val_loader = build_loaders(train_rows, val_rows, cfg, trial_texts=trial_texts)

    eeg_model = EEGEncoder(
        f1=cfg.f1,
        depth_mult=cfg.depth_mult,
        f2=cfg.f2,
        embed_dim=cfg.embed_dim,
        heads=cfg.attn_heads,
        drop_conv=cfg.drop_conv,
        drop_attn=cfg.drop_attn,
    ).to(device)
    text_model = TextTower(cfg.clip_name, embed_dim=cfg.embed_dim, max_len=cfg.max_text_len).to(device)

    params = list(eeg_model.parameters()) + list(text_model.proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    latest_ckpt = os.path.join(cfg.work_dir, f"latest_{fold_name}.pt")
    best_ckpt = os.path.join(cfg.work_dir, f"best_{fold_name}.pt")
    history_path = os.path.join(cfg.work_dir, f"history_{fold_name}.json")

    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    history = []

    loaded = _load_ckpt_if_exists(latest_ckpt) if cfg.resume else None
    if loaded is not None:
        eeg_model.load_state_dict(loaded["eeg_model"])
        text_model.proj.load_state_dict(loaded["text_proj"])
        optimizer.load_state_dict(loaded["optimizer"])
        if "scaler" in loaded and loaded["scaler"]:
            scaler.load_state_dict(loaded["scaler"])
        start_epoch = int(loaded.get("epoch_completed", 0)) + 1
        global_step = int(loaded.get("global_step", 0))

        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if len(history) > 0:
                best_val = min(float(r["val_loss"]) for r in history if "val_loss" in r)

        print(f"[Resume] {fold_name}: epoch={start_epoch}, global_step={global_step}")

    start_ts = time.monotonic()
    max_seconds = cfg.max_train_hours * 3600.0
    stop_ts = start_ts + max_seconds - cfg.time_buffer_minutes * 60

    timed_out = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr, train_timeout, global_step = run_epoch(
            True,
            eeg_model,
            text_model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            start_ts,
            stop_ts,
            epoch,
            global_step,
            latest_ckpt,
        )

        if train_timeout:
            timed_out = True
            print(f"[{fold_name}] 时间预算触发，训练中断并已保存 latest checkpoint。")
            break

        va, _, global_step = run_epoch(
            False,
            eeg_model,
            text_model,
            val_loader,
            optimizer,
            scaler,
            cfg,
            device,
            start_ts,
            stop_ts,
            epoch,
            global_step,
            latest_ckpt,
        )

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ret_acc": tr["retrieval_acc"],
            "val_loss": va["loss"],
            "val_ret_acc": va["retrieval_acc"],
            "global_step": global_step,
        }
        history.append(row)
        print(
            f"[{fold_name}] E{epoch:02d} | train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"| train_acc={tr['retrieval_acc']:.4f} val_acc={va['retrieval_acc']:.4f}"
        )

        _save_ckpt(
            latest_ckpt,
            {
                "epoch_completed": epoch,
                "global_step": global_step,
                "eeg_model": eeg_model.state_dict(),
                "text_proj": text_model.proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "elapsed_seconds": time.monotonic() - start_ts,
            },
        )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if va["loss"] < best_val:
            best_val = va["loss"]
            _save_ckpt(
                best_ckpt,
                {
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "eeg_model": eeg_model.state_dict(),
                    "text_proj": text_model.proj.state_dict(),
                    "cfg": cfg.__dict__,
                },
            )

        if time.monotonic() >= stop_ts:
            timed_out = True
            print(f"[{fold_name}] 到达最大训练时间，提前退出并保留断点。")
            break

    return {
        "fold": fold_name,
        "best_val_loss": best_val,
        "latest_ckpt": latest_ckpt,
        "best_ckpt": best_ckpt,
        "history_path": history_path,
        "timed_out": timed_out,
        "last_epoch": history[-1]["epoch"] if len(history) > 0 else 0,
    }


def run_loso(cfg: CFG, run_all_folds: bool = False):
    seed_everything(cfg.seed)
    ensure_work_dir(cfg.work_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.text_csv_path and os.path.exists(cfg.text_csv_path):
        trial_texts = load_trial_texts_from_csv(cfg.text_csv_path, n_trials=80)
        print("loaded text protocol csv:", cfg.text_csv_path)
    else:
        trial_texts = build_trial_texts(80)
        print("text protocol csv not found, fallback to template text.")

    all_rows = load_all_samples_with_saveinfo(cfg.data_root, saveinfo_dir=cfg.saveinfo_dir)
    subjects = sorted(list({r["subject"] for r in all_rows}))

    target_subjects = subjects if run_all_folds else subjects[:1]
    print("device=", device)
    print("subjects=", len(subjects), "run_folds=", len(target_subjects))
    print("total_windows=", len(all_rows))

    fold_results = []
    for test_sub in target_subjects:
        train_pool = [r for r in all_rows if r["subject"] != test_sub]
        test_rows = [r for r in all_rows if r["subject"] == test_sub]
        train_rows, val_rows = split_train_val(train_pool, cfg.val_ratio)

        fold_name = f"loso_test_{test_sub}"
        print("\n===", fold_name, "===")
        print("train/val/test =", len(train_rows), len(val_rows), len(test_rows))

        fold_result = train_one_fold(train_rows, val_rows, fold_name, cfg, device, trial_texts)
        fold_results.append(fold_result)

    result_path = os.path.join(cfg.work_dir, "loso_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)

    print("\nSaved:", result_path)
    return fold_results