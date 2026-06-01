"""
EmotionCLIP text tower + multi-template prompt ensemble.

Technical path (Yan et al., 2025 / EmotionCLIP):
  * A FROZEN CLIP text encoder produces class text features.
  * Each emotion class is rendered through many prompt templates
    (prompt ensemble, ported from utils/Text_Prompt.py). Similarity between an
    EEG embedding and the text features is averaged across templates.
  * Only the EEG tower is trained; the text features are precomputed.

Difference vs. the reference repo: the reference uses the original OpenAI `clip`
package (needs ViT-B-16.pt). Here we use HuggingFace `transformers` CLIP so the
weights can be cached offline on Kaggle (chosen by the user). The contrastive
math is preserved.

We ALSO keep the original repo's L2 (trial-level) text descriptions and append
them as additional per-class prompts (mapped to each class via the trial->label
mapping), satisfying the user's request to migrate the CSV protocol.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from data_seedvii import EMOTION_NAMES


# Prompt templates ported from utils/Text_Prompt.py (eeg_text_prompt).
PROMPT_TEMPLATES = [
    "{}",
    "The human is {}",
    "The video makes the human feel {}",
    "A video of {} emotion",
    "Look, the human is {}",
    "Playing a kind of emotion, {}",
    "Doing a kind of emotion, {}",
    "Does this video convey {} emotion?",
    "What emotion does this video convey: {}?",
    "Identify the emotion in this video: {}",
    "Categorize this video into {} emotion",
    "Can you recognize the emotion of {}?",
    "The human feels {} now",
    "The human looks {} about the video",
    "{}, a kind of emotion",
    "{} this is an emotion",
]


class EmotionTextTower(nn.Module):
    """Frozen CLIP text encoder with prompt-ensemble class feature builder."""

    def __init__(self, clip_name: str = "openai/clip-vit-base-patch16", max_len: int = 64):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.encoder = CLIPTextModelWithProjection.from_pretrained(clip_name)
        self.max_len = max_len
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.clip_embed_dim = self.encoder.config.projection_dim  # 512 for ViT-B/16

    @torch.no_grad()
    def _encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.encoder(**tok).text_embeds  # already projected to CLIP dim
        return out.float()

    @torch.no_grad()
    def build_class_text_features(
        self,
        class_words: List[str],
        device: torch.device,
        normalize: bool = True,
        extra_class_prompts: Optional[Dict[int, List[str]]] = None,
    ):
        """Build text features for prompt ensemble.

        Returns:
            text_features: [num_text_aug * num_classes, dim]
            num_text_aug:  number of prompt groups
        Layout matches the reference: group-major, i.e. for each template the
        block of `num_classes` class features is concatenated.
        extra_class_prompts: {class_idx: [free-form prompt strings]} -> each
        unique extra prompt becomes one additional template group.
        """
        num_classes = len(class_words)
        feats_per_group = []

        # Standard template groups.
        for tmpl in PROMPT_TEMPLATES:
            texts = [tmpl.format(c) for c in class_words]
            feats_per_group.append(self._encode(texts, device))

        # Extra groups from migrated CSV L2 trial descriptions.
        # We turn them into uniform-width groups: for each "slot" we need one
        # prompt per class. We build as many extra groups as the MAX number of
        # extra prompts available across classes, so NO L2 text is dropped.
        # Classes that have fewer prompts are CYCLE-FILLED with their own
        # prompts (round-robin) instead of falling back to the bare emotion
        # word, keeping every group well-formed without losing information.
        if extra_class_prompts:
            max_extra = max((len(v) for v in extra_class_prompts.values()), default=0)
            for j in range(max_extra):
                texts = []
                for ci in range(num_classes):
                    plist = extra_class_prompts.get(ci, [])
                    if plist:
                        texts.append(plist[j % len(plist)])   # round-robin fill
                    else:
                        texts.append(class_words[ci])          # only if class has no L2 text
                feats_per_group.append(self._encode(texts, device))

        num_text_aug = len(feats_per_group)
        text_features = torch.cat(feats_per_group, dim=0)  # [aug*cls, dim]
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features, num_text_aug

    def forward(self, texts, device):
        return self._encode(texts, device)


def build_extra_class_prompts_from_l2(l2_texts, subject_label_map=None,
                                      trial_label_lookup=None, num_classes=7,
                                      max_per_class=None):
    """Map L2 trial descriptions to per-class prompt lists.

    trial_label_lookup: dict trial_id -> label index. If not provided we cannot
    assign trial text to a class and return None.

    max_per_class: cap on prompts kept per class. Default None = keep ALL L2
    texts (no truncation). Set an int only if you intentionally want to limit
    the prompt ensemble size for speed/memory.
    """
    if max_per_class is None:
        max_per_class = len(l2_texts) + 1  # effectively unlimited
    if trial_label_lookup is None:
        return None
    per_class: Dict[int, List[str]] = {c: [] for c in range(num_classes)}
    for trial_id, text in sorted(l2_texts.items()):
        lbl = trial_label_lookup.get(int(trial_id))
        if lbl is None or not (0 <= lbl < num_classes):
            continue
        if len(per_class[lbl]) < max_per_class:
            per_class[lbl].append(text)
    if all(len(v) == 0 for v in per_class.values()):
        return None
    return per_class


class KLLoss(nn.Module):
    """KL divergence loss with temperature, ported from utils/KLLoss.py."""

    def __init__(self):
        super().__init__()
        self.error_metric = nn.KLDivLoss(reduction="batchmean")

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, dim=1)
        probs2 = F.softmax(label * 10, dim=1)
        return self.error_metric(probs1, probs2) * batch_size
