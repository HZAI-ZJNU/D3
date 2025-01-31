import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import MultiheadAttention

__all__ = ['nlp2d']


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(256, num_pos_feats)
        self.col_embed = nn.Embedding(256, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


class NLP_BASE(nn.Module):
    def __init__(self, in_channels, idx, patch_size=2, dim_reduced_ratio=1.0, num_heads=4, position_embedding='learned',
                 conv2d=nn.Conv2d):
        super(NLP_BASE, self).__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size

        reduced_dim = round(dim_reduced_ratio * in_channels / num_heads)
        if reduced_dim == 0:
            reduced_dim = 1
        embed_dim = reduced_dim * num_heads
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.downsample = nn.Sequential(
            conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.multihead_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.restore = nn.Sequential(
            conv2d(embed_dim, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        self.pos_embed = build_position_encoding(hidden_dim=embed_dim, position_embedding=position_embedding)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size

        downsampled = self.downsample(x)

        nested_downsampled = NestedTensor(downsampled, None)
        pos_embed = self.pos_embed(nested_downsampled)
        pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')

        embed = rearrange(downsampled, 'b c h w -> b (h w) c') + pos_embed

        attn_seq, attn_weights = self.multihead_attn(embed, embed, embed)
        attn = rearrange(attn_seq, 'b (h w) c -> b c h w', h=h // p, w=w // p)
        attn = self.restore(F.interpolate(attn, size=(h, w), mode='nearest'))

        weight = torch.exp(attn)
        return weight


class Pool2d(nn.Module):
    def __init__(self, pool_module, in_channels, idx, kernel_size=2, stride=2, padding=0, win_norm=True, **kwargs):
        super(Pool2d, self).__init__()
        self.win_norm = win_norm
        self.logit = nn.Sequential(OrderedDict([
            ('pool_weight', pool_module(in_channels, idx, **kwargs))
        ]))
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        weight = self.logit(x)

        if self.win_norm == True:
            y = self.pool(x * weight) / self.pool(weight)
        else:
            y = self.pool(x * weight)
        return y


def nlp2d(inc, idx, kernel_size, stride, padding, win_norm=True, **kwargs):
    return Pool2d(NLP_BASE, inc, idx, kernel_size, stride, padding, win_norm, **kwargs)

