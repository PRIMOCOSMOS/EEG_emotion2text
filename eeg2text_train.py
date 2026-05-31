import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from eeg2text_core import (
    CFG,
    EEGEncoder,
    EEGTextWindowDataset,
    TextTower,
    asymmetric_soft_contrastive_loss,
    build_default_l1_texts,
    build_default_l2_texts,
    collate_fn,
    ensure_work_dir,
    load_all_samples_with_saveinfo,
    load_two_level_texts_from_csv,
    seed_everything,
    split_train_val,
)


def build_loaders(
    train_rows: List[Dict],
    val_rows: List[Dict],
    cfg: CFG,
    l1_texts: Dict[str, str],
    l2_texts: Dict[int, str],
) -> Tuple[DataLoader, DataLoader]:
    use_pin_memory = torch.cuda.is_available()
    train_ds = EEGTextWindowDataset(train_rows, l1_texts=l1_texts, l2_texts=l2_texts)
    val_ds = EEGTextWindowDataset(val_rows, l1_texts=l1_texts, l2_texts=l2_texts)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
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
    epoch_start = time.monotonic()
    samples_processed = 0

    for batch in loader:
        now = time.monotonic()
        if train_mode and now >= stop_ts:
            timed_out = True
            break

        # Move data to device
        eeg = batch["eeg"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if not torch.isfinite(eeg).all():
            raise RuntimeError("Found non-finite EEG values in batch.")
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= 7:
            raise RuntimeError(f"Label out of range [0,6], got min={int(labels.min().item())}, max={int(labels.max().item())}")

        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                eeg_z = eeg_model(eeg)
                txt_z = text_model.forward_from_ids(
                    labels,
                    batch["trial"],
                    device,
                    l1_weight=cfg.l1_weight,
                    l2_weight=cfg.l2_weight,
                )
                loss, metrics = asymmetric_soft_contrastive_loss(
                    eeg_z,
                    txt_z,
                    cfg.temperature,
                    cfg.alpha,
                    labels,
                    same_emotion_weight=cfg.same_emotion_weight,
                    pos_neg_margin=cfg.pos_neg_margin,
                    margin_loss_weight=cfg.margin_loss_weight,
                )

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
        samples_processed += eeg.size(0)

        if train_mode and (n_steps == 1 or n_steps % max(1, cfg.log_every_n_steps) == 0):
            elapsed = time.monotonic() - epoch_start
            throughput = samples_processed / elapsed if elapsed > 0 else 0
            gpu_mem = ""
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                mem_res = torch.cuda.memory_reserved(device) / 1e9
                gpu_mem = f" GPUmem={mem_alloc:.2f}/{mem_res:.2f}GB"
            print(
                f"step={n_steps} global_step={global_step} "
                f"loss={loss.item():.4f} acc={metrics['batch_retrieval_acc']:.4f} "
                f"samples/s={throughput:.1f}{gpu_mem}"
            )

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
    l1_texts: Dict[str, str],
    l2_texts: Dict[int, str],
):
    ensure_work_dir(cfg.work_dir)
    print(f"[Init] building dataloaders, num_workers={cfg.num_workers}, batch_size={cfg.batch_size}")
    train_loader, val_loader = build_loaders(train_rows, val_rows, cfg, l1_texts=l1_texts, l2_texts=l2_texts)
    print(f"[Init] dataloaders ready, train_steps={len(train_loader)}, val_steps={len(val_loader)}")

    eeg_model = EEGEncoder(
        f1=cfg.f1,
        depth_mult=cfg.depth_mult,
        f2=cfg.f2,
        embed_dim=cfg.embed_dim,
        heads=cfg.attn_heads,
        drop_conv=cfg.drop_conv,
        drop_attn=cfg.drop_attn,
    ).to(device)
    print("[Init] EEG model ready")
    text_model = TextTower(cfg.clip_name, embed_dim=cfg.embed_dim, max_len=cfg.max_text_len).to(device)
    print("[Init] Text model ready")
    text_model.build_level_caches(l1_texts=l1_texts, l2_texts=l2_texts, dev=device)
    print(f"[Init] Text caches ready: L1={len(l1_texts)} emotions, L2={len(l2_texts)} trials")

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
    if cfg.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    seed_everything(cfg.seed)
    ensure_work_dir(cfg.work_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name}, Capability: sm_{cap[0]}{cap[1]}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Some Kaggle GPU/runtime combos may fail on flash/mem-efficient SDP kernels.
        # Force math kernel for stability.
        if cfg.force_math_sdp and hasattr(torch.backends, "cuda"):
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
                print("SDP backend: flash=False, mem_efficient=False, math=True")
            except Exception as e:
                print("SDP backend setup skipped:", e)

        # Quick CUDA smoke test to fail early with a clearer message.
        try:
            x = torch.randn(512, 512, device=device)
            y = torch.randn(512, 512, device=device)
            _ = (x @ y).mean().item()
            print("CUDA smoke test: PASSED")
        except Exception as e:
            msg = str(e)
            if "no kernel image is available" in msg:
                raise RuntimeError(
                    "Detected CUDA kernel-image mismatch. Try another Kaggle GPU type (T4), "
                    "or run on CPU, or use a Torch build compatible with this GPU architecture."
                ) from e
            raise

    def _discover_data_paths():
        # Support Kaggle layout like:
        # /kaggle/input/<dataset>/{EEG_features, save_info, Emotion2text}
        data_root = cfg.data_root
        saveinfo_dir = cfg.saveinfo_dir
        text_csv_path = cfg.l1_l2_text_csv_path or cfg.text_csv_path

        input_root = Path("/kaggle/input")
        if input_root.exists():
            feature_dirs = sorted(input_root.rglob("EEG_features"))
            if (not os.path.exists(data_root)) and len(feature_dirs) > 0:
                data_root = str(feature_dirs[0])

            if (not saveinfo_dir or not os.path.exists(saveinfo_dir)) and os.path.exists(data_root):
                parent = Path(data_root).parent
                cands = [parent / "save_info", parent / "Saveinfo", parent / "saveinfo"]
                for c in cands:
                    if c.exists():
                        saveinfo_dir = str(c)
                        break

            if (not text_csv_path or not os.path.exists(text_csv_path)) and os.path.exists(data_root):
                parent = Path(data_root).parent
                cands = []
                emo_dir = parent / "Emotion2text"
                if emo_dir.exists():
                    cands.extend(sorted(emo_dir.glob("*.csv")))
                cands.extend(sorted(parent.glob("*text*protocol*.csv")))
                cands.extend(sorted(parent.rglob("text_protocol*.csv")))
                if len(cands) > 0:
                    text_csv_path = str(cands[0])

        return data_root, saveinfo_dir, text_csv_path

    data_root, saveinfo_dir, text_csv_path = _discover_data_paths()
    print("resolved data_root:", data_root)
    print("resolved saveinfo_dir:", saveinfo_dir)
    print("resolved text_csv_path:", text_csv_path)

    if text_csv_path and os.path.exists(text_csv_path):
        l1_texts, l2_texts = load_two_level_texts_from_csv(text_csv_path, n_trials=80)
        print("loaded two-level text protocol csv:", text_csv_path)
    else:
        l1_texts = build_default_l1_texts()
        l2_texts = build_default_l2_texts(80)
        print("text protocol csv not found, fallback to template text.")

    all_rows = load_all_samples_with_saveinfo(data_root, saveinfo_dir=saveinfo_dir)
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

        fold_result = train_one_fold(train_rows, val_rows, fold_name, cfg, device, l1_texts, l2_texts)
        fold_results.append(fold_result)

    result_path = os.path.join(cfg.work_dir, "loso_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, ensure_ascii=False, indent=2)

    print("\nSaved:", result_path)
    return fold_results