"""
SST-LegoViT: Spatial-Spectral-Temporal LegoViT EEG backbone.

Faithful re-implementation (ported) of the EEG image encoder from
"Cross-domain EEG-based Emotion Recognition with Contrastive Learning"
(Yan et al., 2025) -- official repo: https://github.com/Departure2021/EmotionCLIP

This module is the EEG (image) tower of EmotionCLIP. It maps a 4D EEG
topographic representation (frames, bands, H, W) into the CLIP joint
embedding space (default 512-d to match CLIP ViT-B/16).

Pipeline (single-stream):
    Tubelet Embedding (Conv stem)
        -> Spatial Transformer (Multi-scale Conv2D + MHSA)
        -> Spectral Transformer (Legoformer: DE/PSD dual-stream + cross-attn)
        -> Temporal Transformer
        -> image projection to CLIP dim

The code is adapted to be runnable on both CPU and CUDA (the original used
hard-coded `.cuda()` on positional embeddings; here we use buffers / device
inference so it also works on CPU).
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce


# --------------------------------------------------------------------------- #
# Basic blocks
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    def __init__(self, dropout_p=None):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, x):
        return self._drop_path(x, self.dropout_p, self.training)

    @staticmethod
    def _drop_path(x, dropout_p=0.0, training=False):
        if dropout_p == 0.0 or not training:
            return x
        keep_prob = 1 - dropout_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _make_layer_drop(layer_drop):
    if layer_drop:
        dropout_p = layer_drop.get("dropout_p", 0.0)
        drop_cls = layer_drop.get("type", DropPath)
        return drop_cls(dropout_p)
    return nn.Identity()


class PatchEmbed(nn.Module):
    """EEG signals to Tubelet Embedding.

    Input:  x = [B, frames, bands, H, W]
    Output: x = [B, n_t, n_c, (h*w), embed_dims]
    """

    def __init__(self, img_size, tube_size, embed_dims=128, conv_type="Conv_Stem"):
        super().__init__()
        self.img_size = img_size      # [frames, bands, H, W]
        self.tube_size = tube_size    # [t, c, th, tw]
        self.conv_type = conv_type

        if conv_type == "Conv2d":
            self.projection = nn.Conv2d(
                in_channels=tube_size[1],
                out_channels=embed_dims,
                kernel_size=(tube_size[2], tube_size[3]),
                stride=(tube_size[2], tube_size[3]),
            )
        elif conv_type == "Conv_Stem":
            # 64 -> 32 -> 16 -> 4 -> 4 (channel reshaping stem)
            self.projection = nn.Sequential(
                nn.Conv2d(tube_size[1], embed_dims // 8, kernel_size=(2, 2), stride=(2, 2), padding=0),
                nn.BatchNorm2d(embed_dims // 8),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 8, embed_dims // 2, kernel_size=(2, 2), stride=(2, 2), padding=0),
                nn.BatchNorm2d(embed_dims // 2),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims * 2, kernel_size=(4, 4), stride=(4, 4), padding=0),
                nn.BatchNorm2d(embed_dims * 2),
                nn.ReLU(),
                nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=(1, 1), stride=(1, 1), padding=0),
            )
        else:
            raise TypeError(f"Unsupported conv layer type {conv_type}")

    def forward(self, x):
        # x: [B, t, (c tc), h, w] -> per (b t c) Conv2d -> tokens
        tc = self.tube_size[1]
        n_t = self.img_size[0] // self.tube_size[0]
        n_c = self.img_size[1] // self.tube_size[1]
        x = rearrange(x, "b t (c tc) h w -> (b t c) tc h w", tc=tc)
        x = self.projection(x)
        x = rearrange(x, "(b t c) p h w -> b t c (h w) p", t=n_t, c=n_c)
        return x


class MultiheadAttentionWithPreNorm(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_dropout=0.0, attn_proj_dropout=0.0,
                 norm_layer=nn.LayerNorm, layer_drop=None, **kwargs):
        super().__init__()
        self.norm_layer = norm_layer(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_dropout, batch_first=True)
        self.proj_drop = nn.Dropout(attn_proj_dropout)
        self.layer_drop = _make_layer_drop(layer_drop)

    def forward(self, x, **kwargs):
        residual = x
        x = self.norm_layer(x)
        attn_out = self.attn(x, x, x)[0]
        return residual + self.layer_drop(self.proj_drop(attn_out))


class MultiheadCrossAttentionWithPreNorm(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_dropout=0.0, attn_proj_dropout=0.0,
                 norm_layer=nn.LayerNorm, layer_drop=None, **kwargs):
        super().__init__()
        self.norm_query = norm_layer(embed_dims)
        self.norm_key = norm_layer(embed_dims)
        self.norm_value = norm_layer(embed_dims)
        self.cross_attention = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_dropout, batch_first=True)
        self.proj_drop = nn.Dropout(attn_proj_dropout)
        self.layer_drop = _make_layer_drop(layer_drop)

    def forward(self, query, key=None, value=None, **kwargs):
        q = self.norm_query(query)
        k = self.norm_key(key)
        v = self.norm_value(value)
        residual = q
        attn_out = self.cross_attention(q, k, v)[0]
        return residual + self.layer_drop(self.proj_drop(attn_out))


class FFNWithPreNorm(nn.Module):
    def __init__(self, embed_dims=256, hidden_dims=1024, num_layers=2, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, ffn_proj_dropout=0.0, layer_drop=None, **kwargs):
        super().__init__()
        assert num_layers >= 2
        self.norm_layer = norm_layer(embed_dims)
        layers = []
        in_channels = embed_dims
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_channels, hidden_dims))
            layers.append(act_layer())
            layers.append(nn.Dropout(ffn_proj_dropout))
            in_channels = hidden_dims
        layers.append(nn.Linear(hidden_dims, embed_dims))
        layers.append(nn.Dropout(ffn_proj_dropout))
        self.layers = nn.ModuleList(layers)
        self.layer_drop = _make_layer_drop(layer_drop)

    def forward(self, x):
        residual = x
        x = self.norm_layer(x)
        for layer in self.layers:
            x = layer(x)
        return residual + self.layer_drop(x)


class MultiscaleConv2DWithPreNorm(nn.Module):
    """Multi-scale 2D conv applied on spatial tokens (reshaped to h x w)."""

    def __init__(self, embed_dims=128, multi_conv2d_hidden_dims=256, norm_layer=nn.LayerNorm,
                 multi_conv_dropout=0.0, layer_drop=None,
                 filter_sizes=((1, 1), (3, 3), (5, 5)), height=4, width=4):
        super().__init__()
        self.height = height
        self.width = width
        self.num_multi_conv = len(filter_sizes)
        self.norm_layer = norm_layer(embed_dims)

        multi_conv_layers = []
        for fs in filter_sizes:
            multi_conv_layers.append(nn.Sequential(
                nn.Conv2d(embed_dims, multi_conv2d_hidden_dims, kernel_size=(fs[0], fs[1]),
                          stride=(1, 1), padding=((fs[0] - 1) // 2, (fs[1] - 1) // 2)),
                nn.BatchNorm2d(multi_conv2d_hidden_dims),
                nn.Dropout(multi_conv_dropout),
                nn.ReLU(),
                nn.Conv2d(multi_conv2d_hidden_dims, embed_dims, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(embed_dims),
                nn.Dropout(multi_conv_dropout),
                nn.ReLU(),
            ))
        self.multi_conv_layers = nn.ModuleList(multi_conv_layers)
        self.layer_drop = _make_layer_drop(layer_drop)

    def forward(self, x):
        residual = x
        x = self.norm_layer(x)
        x = rearrange(x, "b (h w) d -> b d h w", h=self.height, w=self.width)
        conv_outs = [layer(x).unsqueeze(1) for layer in self.multi_conv_layers]
        x = torch.sum(torch.cat(conv_outs, dim=1), dim=1) / self.num_multi_conv
        x = rearrange(x, "b d h w -> b (h w) d", h=self.height, w=self.width)
        return residual + self.layer_drop(x)


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
class TransformerContainer(nn.Module):
    def __init__(self, embed_dims, num_transformer_layers, num_heads, hidden_dims,
                 attn_dropout=0.0, attn_proj_dropout=0.0, ffn_proj_dropout=0.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_layers=2, drop_path_rate=0.1):
        super().__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(nn.ModuleList([
                MultiheadAttentionWithPreNorm(
                    embed_dims=embed_dims, num_heads=num_heads, attn_dropout=attn_dropout,
                    attn_proj_dropout=attn_proj_dropout, norm_layer=norm_layer,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)),
                FFNWithPreNorm(
                    embed_dims=embed_dims, hidden_dims=hidden_dims, num_layers=num_layers,
                    act_layer=act_layer, norm_layer=norm_layer, ffn_proj_dropout=ffn_proj_dropout,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)),
            ]))

    def forward(self, x):
        for attn, ff in self.transformer_layers:
            x = attn(x)
            x = ff(x)
        return x


class SpatialTransformerContainer(nn.Module):
    def __init__(self, embed_dims, num_transformer_layers, num_heads, multi_conv2d_hidden_dims=256,
                 attn_dropout=0.0, attn_proj_dropout=0.0, multi_conv_dropout=0.0,
                 norm_layer=nn.LayerNorm, drop_path_rate=0.1,
                 filter_sizes=((1, 1), (3, 3), (5, 5)), height=4, width=4):
        super().__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(nn.ModuleList([
                MultiheadAttentionWithPreNorm(
                    embed_dims=embed_dims, num_heads=num_heads, attn_dropout=attn_dropout,
                    attn_proj_dropout=attn_proj_dropout, norm_layer=norm_layer,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)),
                MultiscaleConv2DWithPreNorm(
                    embed_dims=embed_dims, multi_conv2d_hidden_dims=multi_conv2d_hidden_dims,
                    multi_conv_dropout=multi_conv_dropout, norm_layer=norm_layer,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate),
                    filter_sizes=filter_sizes, height=height, width=width),
            ]))

    def forward(self, x):
        for attn, multi_conv in self.transformer_layers:
            x = attn(x)
            x = multi_conv(x)
        return x


class DecoderContainer(nn.Module):
    def __init__(self, embed_dims, num_transformer_layers, num_heads, hidden_dims,
                 attn_dropout=0.0, attn_proj_dropout=0.0, ffn_proj_dropout=0.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_layers=2, drop_path_rate=0.1):
        super().__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(nn.ModuleList([
                MultiheadAttentionWithPreNorm(
                    embed_dims=embed_dims, num_heads=num_heads, attn_dropout=attn_dropout,
                    attn_proj_dropout=attn_proj_dropout, norm_layer=norm_layer,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)),
                MultiheadCrossAttentionWithPreNorm(
                    embed_dims=embed_dims, num_heads=num_heads, attn_dropout=attn_dropout,
                    attn_proj_dropout=attn_proj_dropout, norm_layer=norm_layer,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)),
                FFNWithPreNorm(
                    embed_dims=embed_dims, hidden_dims=hidden_dims, num_layers=num_layers,
                    act_layer=act_layer, norm_layer=norm_layer, ffn_proj_dropout=ffn_proj_dropout,
                    layer_drop=dict(type=DropPath, dropout_p=drop_path_rate)),
            ]))

    def forward(self, query, key=None, value=None):
        for self_attn, cross_attn, ff in self.transformer_layers:
            query = self_attn(query)
            query = cross_attn(query=query, key=key, value=value)
            query = ff(query)
        return query


class Legoformer(nn.Module):
    """Spectral Legoformer: parallel DE/PSD transformers + cross-attention bridge.

    Input is the spectral token sequence [B, n_c, d] where the first half of
    channels are DE bands and the second half are PSD bands. If n_c is odd or
    PSD is absent, the input is duplicated to form a symmetric DE/PSD pair.
    """

    def __init__(self, embed_dims, num_transformer_layers, num_heads,
                 attn_dropout=0.0, attn_proj_dropout=0.0, ffn_proj_dropout=0.0, drop_path_rate=0.1):
        super().__init__()
        self.bridge_layers = nn.ModuleList([])
        for _ in range(num_transformer_layers):
            self.bridge_layers.append(nn.ModuleList([
                TransformerContainer(num_transformer_layers=1, embed_dims=embed_dims, num_heads=num_heads,
                                     hidden_dims=embed_dims * 4, attn_dropout=attn_dropout,
                                     attn_proj_dropout=attn_proj_dropout, ffn_proj_dropout=ffn_proj_dropout,
                                     drop_path_rate=drop_path_rate),
                TransformerContainer(num_transformer_layers=1, embed_dims=embed_dims, num_heads=num_heads,
                                     hidden_dims=embed_dims * 4, attn_dropout=attn_dropout,
                                     attn_proj_dropout=attn_proj_dropout, ffn_proj_dropout=ffn_proj_dropout,
                                     drop_path_rate=drop_path_rate),
                DecoderContainer(num_transformer_layers=1, embed_dims=embed_dims, num_heads=num_heads,
                                 hidden_dims=embed_dims * 4, attn_dropout=attn_dropout,
                                 attn_proj_dropout=attn_proj_dropout, ffn_proj_dropout=ffn_proj_dropout,
                                 drop_path_rate=drop_path_rate),
            ]))

    def forward(self, x):
        nc = x.shape[1]
        if nc % 2 != 0:
            # symmetric DE/PSD split is impossible -> duplicate to keep dual-stream design
            x = torch.cat([x, x], dim=1)
            nc = x.shape[1]
        x_de, x_psd = x[:, : nc // 2, :], x[:, nc // 2:, :]
        encoder_de, encoder_psd, decoder_de = x_de, x_psd, x_de
        for transformer_de, transformer_psd, cross_transformer_de in self.bridge_layers:
            encoder_de, encoder_psd = transformer_de(encoder_de), transformer_psd(encoder_psd)
            decoder_de = encoder_de + encoder_psd + decoder_de
            decoder_de = cross_transformer_de(query=decoder_de, key=encoder_de, value=encoder_de)
        return decoder_de


class TemporalEncoderTransformer(nn.Module):
    def __init__(self, num_frames, embed_dims, num_transformer_layers, num_heads, hidden_dims,
                 attn_dropout=0.0, attn_proj_dropout=0.0, ffn_proj_dropout=0.0,
                 drop_path_rate=0.1, temporal_type="Transformer"):
        super().__init__()
        self.num_temporal_transformer_layers = num_transformer_layers
        self.temporal_type = temporal_type

        if num_transformer_layers != 0 and temporal_type == "Transformer":
            self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dims))
            nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

        if temporal_type == "MeanPooling":
            self.temporal_transformer = nn.Identity()
        elif temporal_type == "Transformer":
            self.temporal_transformer = TransformerContainer(
                num_transformer_layers=num_transformer_layers, embed_dims=embed_dims, num_heads=num_heads,
                hidden_dims=embed_dims * 4, attn_dropout=attn_dropout, attn_proj_dropout=attn_proj_dropout,
                ffn_proj_dropout=ffn_proj_dropout, drop_path_rate=drop_path_rate)
        elif temporal_type == "LSTM":
            self.temporal_transformer = nn.LSTM(input_size=embed_dims, hidden_size=embed_dims,
                                                batch_first=True, bidirectional=False, num_layers=1)
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

    def forward(self, x):
        if self.temporal_type == "MeanPooling":
            return x.mean(dim=1)
        if self.temporal_type == "Transformer":
            if self.num_temporal_transformer_layers != 0:
                x = x + self.temporal_embedding.to(x.device)
            x = self.temporal_transformer(x)
            return x.mean(dim=1)
        if self.temporal_type == "LSTM":
            x_original = x
            x, _ = self.temporal_transformer(x)
            x = x.type(x_original.dtype) + x_original
            return x.mean(dim=1)
        raise ValueError(f"Unknown temporal_type: {self.temporal_type}")


class FactorisedEncoderTransformerEncoder(nn.Module):
    """Factorised Spatial-Spectral-Temporal encoder (SST core)."""

    def __init__(self, num_frames, num_channels, num_spatial, embed_dims, multi_conv2d_hidden_dims,
                 num_spatial_transformer_layers, num_spectral_transformer_layers,
                 num_temporal_transformer_layers, num_heads, spectral_type="Legoformer",
                 spatial_type="Multi_Conv2D", temporal_type="Transformer",
                 attn_dropout=0.0, attn_proj_dropout=0.0, ffn_proj_dropout=0.0,
                 multi_conv_dropout=0.0, drop_path_rate=0.1, use_spectral_pos_embedding=False,
                 spatial_height=4, spatial_width=4):
        super().__init__()
        self.spectral_type = spectral_type
        self.temporal_type = temporal_type
        self.num_spatial_transformer_layers = num_spatial_transformer_layers
        self.num_spectral_transformer_layers = num_spectral_transformer_layers
        self.num_temporal_transformer_layers = num_temporal_transformer_layers
        self.use_spectral_pos_embedding = use_spectral_pos_embedding

        if num_spatial_transformer_layers != 0:
            self.spatial_embedding = nn.Parameter(torch.zeros(1, num_spatial, embed_dims))
            nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        if num_spectral_transformer_layers != 0 and use_spectral_pos_embedding:
            self.spectral_embedding = nn.Parameter(torch.zeros(1, num_channels, embed_dims))
            nn.init.trunc_normal_(self.spectral_embedding, std=0.02)

        # Spatial
        if num_spatial_transformer_layers == 0:
            spatial_transformer = nn.Identity()
        elif spatial_type == "Transformer":
            spatial_transformer = TransformerContainer(
                num_transformer_layers=num_spatial_transformer_layers, embed_dims=embed_dims,
                num_heads=num_heads, hidden_dims=embed_dims * 4, attn_dropout=attn_dropout,
                attn_proj_dropout=attn_proj_dropout, ffn_proj_dropout=ffn_proj_dropout,
                drop_path_rate=drop_path_rate)
        else:
            spatial_transformer = SpatialTransformerContainer(
                num_transformer_layers=num_spatial_transformer_layers, embed_dims=embed_dims,
                num_heads=num_heads, multi_conv2d_hidden_dims=multi_conv2d_hidden_dims,
                attn_dropout=attn_dropout, attn_proj_dropout=attn_proj_dropout,
                multi_conv_dropout=multi_conv_dropout, drop_path_rate=drop_path_rate,
                filter_sizes=((1, 1), (3, 3), (5, 5)), height=spatial_height, width=spatial_width)

        # Spectral
        if num_spectral_transformer_layers == 0:
            spectral_transformer = nn.Identity()
        elif spectral_type == "Transformer":
            spectral_transformer = TransformerContainer(
                num_transformer_layers=num_spectral_transformer_layers, embed_dims=embed_dims,
                num_heads=num_heads, hidden_dims=embed_dims * 4, attn_dropout=attn_dropout,
                attn_proj_dropout=attn_proj_dropout, ffn_proj_dropout=ffn_proj_dropout,
                drop_path_rate=drop_path_rate)
        elif spectral_type == "Legoformer":
            spectral_transformer = Legoformer(
                embed_dims=embed_dims, num_transformer_layers=num_spectral_transformer_layers,
                num_heads=num_heads, attn_dropout=attn_dropout, attn_proj_dropout=attn_proj_dropout,
                ffn_proj_dropout=ffn_proj_dropout, drop_path_rate=drop_path_rate)
        else:
            spectral_transformer = nn.Identity()

        temporal_transformer = TemporalEncoderTransformer(
            embed_dims=embed_dims, num_frames=num_frames,
            num_transformer_layers=num_temporal_transformer_layers, num_heads=num_heads,
            hidden_dims=embed_dims * 4, attn_dropout=attn_dropout, attn_proj_dropout=attn_proj_dropout,
            ffn_proj_dropout=ffn_proj_dropout, drop_path_rate=drop_path_rate, temporal_type=temporal_type)

        self.spatial_transformer = spatial_transformer
        self.spectral_transformer = spectral_transformer
        self.temporal_transformer = temporal_transformer

    def forward(self, x):
        """x = [B, nt, nc, (h w), d]"""
        B, nt, nc, npx = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        x = rearrange(x, "b t c p d -> (b t c) p d", b=B, t=nt, c=nc)
        if self.num_spatial_transformer_layers != 0:
            x = x + self.spatial_embedding.to(x.device)
        x = self.spatial_transformer(x)
        x = rearrange(x, "(b t c) p d -> b t c p d", b=B, t=nt, c=nc)
        x = reduce(x, "b t c p d -> b t c d", "mean", b=B, t=nt, c=nc)

        x = rearrange(x, "b t c d -> (b t) c d", b=B, t=nt)
        if self.num_spectral_transformer_layers != 0 and self.use_spectral_pos_embedding:
            x = x + self.spectral_embedding.to(x.device)
        x = self.spectral_transformer(x)
        x = rearrange(x, "(b t) c d -> b t c d", b=B, t=nt)
        x = reduce(x, "b t c d -> b t d", "mean")

        x = self.temporal_transformer(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes, in_channels, init_std=0.02):
        super().__init__()
        self.init_std = init_std
        self.norm_layer = nn.LayerNorm(in_channels)
        self.cls_head = nn.Linear(in_channels, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.cls_head(self.norm_layer(x))


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class SSTLegoViT(nn.Module):
    """SST-LegoViT EEG image encoder (EmotionCLIP image tower).

    forward(x) returns class logits (for optional supervised probing).
    encode_image(x) returns the CLIP-space embedding (used for contrastive).
    """

    def __init__(self, image_frames, image_channels, image_height, image_width,
                 tubelet_frames, tubelet_channels, tubelet_height, tubelet_width,
                 num_classes=3, num_transformer_layers=(2, 2, 0), embed_dims=128,
                 num_heads=4, multi_conv2d_hidden_dims=128,
                 spatial_type="Multi_Conv2D", spectral_type="Legoformer",
                 temporal_type="Transformer", attn_dropout=0.0, attn_proj_dropout=0.0,
                 ffn_proj_dropout=0.1, multi_conv_dropout=0.1, drop_path_rate=0.1,
                 dropout_after_pos_embed=0.3, conv_type="Conv_Stem",
                 use_spectral_pos_embedding=False, clip_embed_dim=512):
        super().__init__()
        assert image_frames % tubelet_frames == 0 and image_channels % tubelet_channels == 0
        assert image_height % tubelet_height == 0 and image_width % tubelet_width == 0

        self.embed_dims = embed_dims
        self.nHight = image_height // tubelet_height
        self.nWidth = image_width // tubelet_width

        self.tubelet_embedding = PatchEmbed(
            img_size=[image_frames, image_channels, image_height, image_width],
            tube_size=[tubelet_frames, tubelet_channels, tubelet_height, tubelet_width],
            embed_dims=embed_dims, conv_type=conv_type)

        self.drop_after_pos = nn.Dropout(dropout_after_pos_embed)

        # After Conv_Stem the spatial map collapses to 4x4 -> 16 tokens.
        spatial_h = 4 if conv_type == "Conv_Stem" else self.nHight
        spatial_w = 4 if conv_type == "Conv_Stem" else self.nWidth

        self.transformer = FactorisedEncoderTransformerEncoder(
            num_frames=image_frames // tubelet_frames,
            num_channels=image_channels // tubelet_channels,
            num_spatial=spatial_h * spatial_w,
            embed_dims=embed_dims,
            multi_conv2d_hidden_dims=multi_conv2d_hidden_dims,
            num_spatial_transformer_layers=num_transformer_layers[0],
            num_spectral_transformer_layers=num_transformer_layers[1],
            num_temporal_transformer_layers=num_transformer_layers[2],
            num_heads=num_heads, spectral_type=spectral_type, spatial_type=spatial_type,
            temporal_type=temporal_type, attn_dropout=attn_dropout,
            attn_proj_dropout=attn_proj_dropout, ffn_proj_dropout=ffn_proj_dropout,
            multi_conv_dropout=multi_conv_dropout, drop_path_rate=drop_path_rate,
            use_spectral_pos_embedding=use_spectral_pos_embedding,
            spatial_height=spatial_h, spatial_width=spatial_w)

        # Projection to CLIP joint space.
        self.image_proj = nn.Parameter(torch.randn(embed_dims, clip_embed_dim))
        nn.init.normal_(self.image_proj, std=embed_dims ** -0.5)

        self.ClassificationHead = ClassificationHead(num_classes=num_classes, in_channels=embed_dims)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def encode_image(self, image):
        x = self.tubelet_embedding(image)
        x = self.transformer(x)
        return x @ self.image_proj

    def forward(self, x):
        x = self.tubelet_embedding(x)
        x = self.transformer(x)
        return self.ClassificationHead(x)


def create_eeg_encoder(cfg):
    """Build the SST-LegoViT from a config dict (see config.py)."""
    n = cfg["network"]
    return SSTLegoViT(
        image_frames=n["image_frames"], image_channels=n["image_channels"],
        image_height=n["image_height"], image_width=n["image_width"],
        tubelet_frames=n["tubelet_frames"], tubelet_channels=n["tubelet_channels"],
        tubelet_height=n["tubelet_height"], tubelet_width=n["tubelet_width"],
        num_classes=cfg["data"]["num_classes"],
        num_transformer_layers=tuple(n["num_transformer_layers"]),
        embed_dims=n["embed_dims"], num_heads=n["num_heads"],
        multi_conv2d_hidden_dims=n["multi_conv2d_hidden_dims"],
        spatial_type=n["spatial_type"], spectral_type=n["spectral_type"],
        temporal_type=n["temporal_type"], attn_dropout=n["attn_dropout"],
        attn_proj_dropout=n["attn_proj_dropout"], ffn_proj_dropout=n["ffn_proj_dropout"],
        multi_conv_dropout=n["multi_conv_dropout"], drop_path_rate=n["drop_path_rate"],
        dropout_after_pos_embed=n["dropout_after_pos_embed"], conv_type=n["conv_type"],
        use_spectral_pos_embedding=n["use_spectral_pos_embedding"],
        clip_embed_dim=n["clip_embed_dim"])


if __name__ == "__main__":
    # smoke test on CPU
    model = SSTLegoViT(
        image_frames=4, image_channels=10, image_height=64, image_width=64,
        tubelet_frames=1, tubelet_channels=1, tubelet_height=16, tubelet_width=16,
        num_classes=7, num_transformer_layers=(2, 2, 2), embed_dims=128, num_heads=4,
        multi_conv2d_hidden_dims=128, spectral_type="Legoformer", temporal_type="Transformer",
        conv_type="Conv_Stem", clip_embed_dim=512)
    x = torch.rand(8, 4, 10, 64, 64)
    emb = model.encode_image(x)
    logits = model(x)
    print("embedding:", emb.shape, "logits:", logits.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable params: {n_params:.3f}M")
