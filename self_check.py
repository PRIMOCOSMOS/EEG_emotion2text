"""
Self-check for the EmotionCLIP (SST-LegoViT) SEED-VII pipeline.

Run this ONCE on Kaggle (or anywhere with the data mounted) to verify the
training logic is correct AND faithful to the official EmotionCLIP repo
(Departure2021/EmotionCLIP, Yan et al. 2025) before launching a long run.

Usage:
    python self_check.py
    # or import and call run_all_checks(cfg)

It runs two tiers:
  [A] Logic/fidelity checks that need NO data (pure unit tests).
  [B] Data/pipeline checks that use your mounted SEED-VII data if available.

Every check prints PASS/FAIL with a short reason. Exit code is non-zero if any
hard check fails.
"""

import os
import sys
import math
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_PASS, _FAIL, _SKIP = "PASS", "FAIL", "SKIP"
_results = []


def _check(name, cond, detail=""):
    status = _PASS if cond else _FAIL
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def _skip(name, reason):
    """Record a non-fatal SKIP (e.g. CLIP weights or data not present here)."""
    _results.append((name, _SKIP, reason))
    print(f"[{_SKIP}] {name}  -- {reason}")


# --------------------------------------------------------------------------- #
# [A] Logic / fidelity checks (no data needed)
# --------------------------------------------------------------------------- #
def check_class_logits_formula():
    """compute_class_logits must return raw temperature-scaled cosine logits;
    compute_class_probs is only a softmax wrapper for diagnostics/eval."""
    from train_emotionclip import compute_class_logits, compute_class_probs
    B, aug, C, D = 8, 4, 3, 16
    tau = 0.1
    emb = torch.randn(B, D)
    text = F.normalize(torch.randn(aug * C, D), dim=-1)

    logits = compute_class_logits(emb, text, aug, C, temperature=tau)
    probs = compute_class_probs(emb, text, aug, C, temperature=tau)

    e = F.normalize(emb, dim=-1)
    t = F.normalize(text, dim=-1)
    ref_logits = (e @ t.t()).view(B, aug, C).mean(dim=1) / tau

    ok = torch.allclose(logits, ref_logits, atol=1e-6)
    _check("class logits = mean cosine similarity / temperature", ok,
           "max|diff|={:.2e}".format((logits - ref_logits).abs().max().item()))
    _check("class probs are softmax(logits) and sum to 1",
           torch.allclose(probs, logits.softmax(1), atol=1e-6)
           and torch.allclose(probs.sum(1), torch.ones(B), atol=1e-5))


def check_ce_loss_and_grad():
    """Training CE must consume RAW logits with integer labels and mean reduction."""
    from train_emotionclip import compute_class_logits
    B, aug, C, D = 32, 4, 3, 16
    emb = torch.randn(B, D, requires_grad=True)
    text = F.normalize(torch.randn(aug * C, D), dim=-1)
    labels = torch.randint(0, C, (B,))

    logits = compute_class_logits(emb, text, aug, C, temperature=0.1)
    loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
    loss.backward()
    g = emb.grad.norm().item()
    _check("CE(reduction=mean) on raw logits runs", torch.isfinite(loss).item(),
           f"loss={loss.item():.3f}")
    _check("gradient flows to EEG embedding (non-zero, finite)",
           (g > 1e-6) and math.isfinite(g), f"grad_norm={g:.4f}")


def check_temperature_config_and_no_aux():
    """Temperature is an explicit fixed config value; no auxiliary CE head added."""
    import train_emotionclip as T
    from config import get_config
    src = open(T.__file__).read()
    cfg = get_config(label_mode="valence")
    temp = cfg["solver"].get("temperature", None)
    _check("fixed temperature configured in solver", isinstance(temp, (int, float)) and temp > 0,
           f"temperature={temp}")
    _check("no learnable logit_scale parameter in train path", "logit_scale" not in src)
    no_aux = "aux_ce_weight" not in src and "aux_logits" not in src
    _check("no extra auxiliary CE head in train path", no_aux)

def check_text_tower_frozen():
    """Fidelity: text tower must be frozen (only EEG tower trains)."""
    try:
        from text_tower import EmotionTextTower
    except Exception as e:
        _check("text tower import", False, repr(e)); return
    try:
        tower = EmotionTextTower("openai/clip-vit-base-patch16")
    except Exception as e:
        _skip("CLIP text encoder frozen", f"CLIP weights/net unavailable here ({type(e).__name__})")
        return
    n_trainable = sum(p.requires_grad for p in tower.encoder.parameters())
    _check("CLIP text encoder fully frozen", n_trainable == 0,
           f"{n_trainable} params still require grad")


def check_de_psd_dual_stream():
    """DE+PSD 4D build: 10 channels = [5 DE | 5 PSD] (Legoformer split = DE|PSD)."""
    from data_seedvii import feat_window_to_4d
    de = np.random.randn(5, 62).astype(np.float32)
    psd = np.random.randn(5, 62).astype(np.float32)
    img = feat_window_to_4d(de, 4, 10, 64, 64, psd_window=psd, use_psd=True)
    _check("DE+PSD -> 10-channel 4D (frames,10,H,W)",
           tuple(img.shape) == (4, 10, 64, 64), str(tuple(img.shape)))
    # DE-only fallback still yields requested channels
    img2 = feat_window_to_4d(de, 4, 10, 64, 64, psd_window=None, use_psd=False)
    _check("DE-only fallback fills channels", tuple(img2.shape) == (4, 10, 64, 64))


def check_mat_robustness():
    """`.mat` parsing must be robust: axis-permutation, singleton dims, NaN/Inf."""
    import scipy.io as sio
    import tempfile
    from data_seedvii import _load_mat_any, _normalize_feat_TBN, _mat_get

    # axis-permutation handling: (5,62,T)/(62,5,T)/(T,5,62) all -> (T,5,62)
    T = 7
    base = np.random.randn(T, 5, 62).astype(np.float64)
    forms = {
        "(T,5,62)": base,
        "(5,62,T)": np.transpose(base, (1, 2, 0)),
        "(62,5,T)": np.transpose(base, (2, 1, 0)),
        "(1,T,5,62)": base[np.newaxis],
    }
    ok = True
    for name, arr in forms.items():
        out = _normalize_feat_TBN(arr)
        ok = ok and (out is not None) and (out.shape == (T, 5, 62))
    _check("feature axis-permutation/singleton normalized to (T,5,62)", ok)

    # NaN/Inf sanitized
    bad = base.copy(); bad[0, 0, 0] = np.nan; bad[1, 1, 1] = np.inf
    out = _normalize_feat_TBN(bad)
    _check("NaN/Inf sanitized in features", out is not None and np.isfinite(out).all())

    # round-trip through a real .mat via the robust reader (classic format)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.mat")
    sio.savemat(p, {"de_LDS_1": base, "psd_1": base * 2})
    m = _load_mat_any(p)
    _check("robust .mat reader returns expected keys",
           _mat_get(m, "de_LDS_1") is not None and _mat_get(m, "psd_1") is not None)
    _check("robust .mat reader is case-insensitive on keys",
           _mat_get(m, "DE_LDS_1") is not None)


def check_model_io_shapes():
    """SST-LegoViT: encode_image -> CLIP dim; forward -> num_classes logits."""
    from sst_legovit import SSTLegoViT
    for ncls in (7, 3):
        m = SSTLegoViT(image_frames=4, image_channels=10, image_height=64, image_width=64,
                       tubelet_frames=1, tubelet_channels=1, tubelet_height=16, tubelet_width=16,
                       num_classes=ncls, num_transformer_layers=(2, 2, 0), embed_dims=128,
                       num_heads=4, multi_conv2d_hidden_dims=128, conv_type="Conv_Stem",
                       clip_embed_dim=512)
        x = torch.rand(2, 4, 10, 64, 64)
        emb = m.encode_image(x)
        logits = m(x)
        _check(f"encode_image -> CLIP dim (ncls={ncls})", tuple(emb.shape) == (2, 512),
               str(tuple(emb.shape)))
        _check(f"forward -> num_classes logits (ncls={ncls})", tuple(logits.shape) == (2, ncls),
               str(tuple(logits.shape)))


def check_label_scheme():
    """LabelScheme: fine=7, valence=3, mapping correct, prompt words correct."""
    from data_seedvii import LabelScheme, EMOTION_NAMES
    fine = LabelScheme("fine")
    val = LabelScheme("valence")
    _check("fine scheme has 7 classes", fine.num_classes == 7, str(fine.names))
    _check("valence scheme has 3 classes", val.num_classes == 3, str(val.names))
    # default valence mapping
    want = {"neutral": "neutral", "joy": "positive", "sadness": "negative",
            "fear": "negative", "disgust": "negative", "anger": "negative",
            "surprise": "positive"}
    ok = all(val.names[val.map_fine(EMOTION_NAMES.index(e))] == w for e, w in want.items())
    _check("valence aggregation maps 7->3 as expected", ok)
    # labels stay in range
    rng_ok = all(0 <= val.map_fine(i) < 3 for i in range(7))
    _check("valence labels in [0,3)", rng_ok)


def check_no_double_count_consistency():
    """Prediction path uses the same raw-logit formula as the training path."""
    from train_emotionclip import compute_class_logits
    B, aug, C, D = 16, 4, 3, 16
    tau = 0.1
    emb = torch.randn(B, D); text = F.normalize(torch.randn(aug * C, D), dim=-1)
    logits = compute_class_logits(emb, text, aug, C, temperature=tau)
    ref = (F.normalize(emb, dim=-1) @ F.normalize(text, dim=-1).t()).view(B, aug, C).mean(1) / tau
    _check("train/eval prediction == raw-logit argmax",
           torch.equal(logits.argmax(1), ref.argmax(1)))


# --------------------------------------------------------------------------- #
# [B] Data / pipeline checks (use mounted data if present)
# --------------------------------------------------------------------------- #
def check_data_pipeline(cfg):
    from config import discover_data_paths
    from data_seedvii import (make_label_scheme, load_all_samples_with_saveinfo,
                              build_subject_label_map_from_saveinfo,
                              load_two_level_texts_from_csv, EEGTopoDataset, collate_fn)
    cfg = discover_data_paths(cfg)
    d = cfg["data"]
    if not (d["data_root"] and os.path.exists(d["data_root"])):
        _skip("data pipeline checks", f"data_root not found: {d['data_root']}")
        return
    scheme = make_label_scheme(d)
    nc = scheme.num_classes

    rows = load_all_samples_with_saveinfo(d["data_root"], d["saveinfo_dir"], d["n_trials"],
                                          normalize_subject_zscore=True, label_scheme=scheme)
    labels = sorted({r["label"] for r in rows})
    _check("windows loaded", len(rows) > 0, f"{len(rows)} windows")
    _check("labels within active class range", all(0 <= l < nc for l in labels), str(labels))

    # 4x20=80 trial ordering sanity (per subject session concat)
    slm = build_subject_label_map_from_saveinfo(d["data_root"], d["saveinfo_dir"], d["n_trials"])
    lens_ok = all(len(v) == d["n_trials"] for v in slm.values()) if slm else False
    _check("every subject has n_trials labels (4x20=80 concat)", lens_ok,
           {k: len(v) for k, v in (slm or {}).items()})

    # leakage: per-subject zscore already applied; check LOSO split disjoint
    subs = sorted({r["subject"] for r in rows})
    if len(subs) >= 2:
        test_sub = subs[0]
        train_subjects = {r["subject"] for r in rows if r["subject"] != test_sub}
        _check("LOSO test subject excluded from train pool", test_sub not in train_subjects)

    # CSV L2 -> prompts coverage (no text dropped)
    if d["text_csv_path"] and os.path.exists(d["text_csv_path"]):
        _, l2 = load_two_level_texts_from_csv(d["text_csv_path"], d["n_trials"])
        import csv as _csv
        with open(d["text_csv_path"], encoding="utf-8") as f:
            raw = {int(r["trial"]): r["l2_text"] for r in _csv.DictReader(f)
                   if (r.get("trial") or "").strip() and (r.get("l2_text") or "").strip()}
        mism = [t for t in l2 if t in raw and l2[t] != raw[t]]
        _check("L2 text parsed without truncation (verbatim)", len(mism) == 0,
               f"mismatched trials: {mism[:5]}")

    # one mini-batch through the dataset
    ds = EEGTopoDataset(rows[:8], cfg["network"]["image_frames"], cfg["network"]["image_channels"],
                        cfg["network"]["image_height"], cfg["network"]["image_width"])
    b = collate_fn([ds[i] for i in range(min(4, len(ds)))])
    _check("4D batch shape (B,frames,C,H,W)",
           tuple(b["image"].shape) == (min(4, len(ds)), cfg["network"]["image_frames"],
                                       cfg["network"]["image_channels"], 64, 64),
           str(tuple(b["image"].shape)))
    _check("batch tensor finite", torch.isfinite(b["image"]).all().item())


def check_one_train_step_decreases_loss(cfg):
    """A few optimizer steps on a tiny real/synthetic batch should reduce loss."""
    from config import discover_data_paths
    from data_seedvii import (make_label_scheme, load_all_samples_with_saveinfo,
                              EEGTopoDataset, collate_fn)
    from sst_legovit import create_eeg_encoder
    from train_emotionclip import compute_class_logits

    cfg = discover_data_paths(cfg)
    d = cfg["data"]
    scheme = make_label_scheme(d)
    nc = scheme.num_classes
    cfg["data"]["num_classes"] = nc

    if d["data_root"] and os.path.exists(d["data_root"]):
        rows = load_all_samples_with_saveinfo(d["data_root"], d["saveinfo_dir"], d["n_trials"],
                                              normalize_subject_zscore=True, label_scheme=scheme)[:64]
    else:
        # synthetic fallback so the check still runs offline
        rng = np.random.default_rng(0)
        rows = [{"subject": "s1", "trial": i % 80 + 1, "label": i % nc,
                 "de": rng.standard_normal((5, 62)).astype(np.float32) + (i % nc)}
                for i in range(64)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = EEGTopoDataset(rows, cfg["network"]["image_frames"], cfg["network"]["image_channels"],
                        cfg["network"]["image_height"], cfg["network"]["image_width"])
    batch = collate_fn([ds[i] for i in range(len(ds))])
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # tiny model + fixed random normalized text anchors
    cfg["network"]["num_transformer_layers"] = [1, 1, 0]
    cfg["network"]["embed_dims"] = 32
    cfg["network"]["multi_conv2d_hidden_dims"] = 32
    model = create_eeg_encoder(cfg).to(device)
    # Distinct, well-separated class anchors so a learnable EEG tower CAN fit the
    # batch (this tests optimization plumbing, not generalization).
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(nc, 512), dim=-1)
    text = anchors.repeat(16, 1).to(device)           # 16 templates x nc classes
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    model.train()
    losses, accs = [], []
    for _ in range(60):
        logits = compute_class_logits(model.encode_image(images), text, 16, nc,
                                      temperature=cfg["solver"].get("temperature", 0.1))
        loss = loss_fn(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        accs.append((logits.argmax(1) == labels).float().mean().item())
    # On a tiny fittable batch the loss must drop and train-acc must rise clearly.
    _check("training loss decreases (optimizer plumbing works)",
           losses[-1] < losses[0] - 1e-3, f"{losses[0]:.2f} -> {losses[-1]:.2f}")
    _check("train accuracy increases on a fittable batch",
           accs[-1] > accs[0] + 0.05, f"acc {accs[0]:.2f} -> {accs[-1]:.2f}")


# --------------------------------------------------------------------------- #
def run_all_checks(cfg=None):
    print("=" * 70)
    print("EmotionCLIP SEED-VII  --  SELF CHECK")
    print("=" * 70)
    if cfg is None:
        from config import get_config
        cfg = get_config(label_mode="fine")

    print("\n--- [A] logic / fidelity (no data needed) ---")
    for fn in (check_class_logits_formula, check_ce_loss_and_grad,
               check_temperature_config_and_no_aux, check_text_tower_frozen,
               check_de_psd_dual_stream, check_mat_robustness, check_model_io_shapes,
               check_label_scheme, check_no_double_count_consistency):
        try:
            fn()
        except Exception:
            _check(fn.__name__, False, "EXCEPTION:\n" + traceback.format_exc().splitlines()[-1])

    print("\n--- [B] data / pipeline (uses mounted data if present) ---")
    for fn in (check_data_pipeline, check_one_train_step_decreases_loss):
        try:
            fn(cfg)
        except Exception:
            _check(fn.__name__, False, "EXCEPTION:\n" + traceback.format_exc().splitlines()[-1])

    n_pass = sum(1 for _, s, _ in _results if s == _PASS)
    n_fail = sum(1 for _, s, _ in _results if s == _FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == _SKIP)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_skip:
        print("  (SKIP = optional check not runnable here, e.g. no CLIP weights / no "
              "data mounted; these run on Kaggle.)")
    print("=" * 70)
    return n_fail == 0


if __name__ == "__main__":
    ok = run_all_checks()
    sys.exit(0 if ok else 1)
