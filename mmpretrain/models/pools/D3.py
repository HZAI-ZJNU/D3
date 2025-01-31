

import torch
import torch.nn as nn
import typing as t

from einops import rearrange
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, build_activation_layer
import torch.nn.functional as F

__all__ = ['D3']


class D3(BaseModule):
    """
    Dual-Domain Downsampling (Pooling)
    """
    def __init__(
            self,
            in_chans: int,
            kernel_size: int = 2,
            stride: t.Optional[int] = None,
            padding: int = 0,
            gate_mode: str = 'sigmoid',
            hidden_factor: int = 16,
            weight_k_size: int = 3,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='GELU'),
            aggregate_mode: str = 'learned',
            split_ratio: float = 0.5,
    ) -> None:
        super(D3, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.aggregate_mode = aggregate_mode
        self.split_ratio = split_ratio
        self.weight_k_size = weight_k_size

        spatial_scale_factor = in_chans

        self.s_conv2 = ConvModule(
            in_channels=spatial_scale_factor,
            out_channels=spatial_scale_factor,
            kernel_size=weight_k_size,
            stride=1,
            groups=spatial_scale_factor,
            padding=weight_k_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.s_conv3 = ConvModule(
            in_channels=spatial_scale_factor,
            out_channels=in_chans,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        if aggregate_mode == 'learned':
            freq_expand_factor = int(kernel_size * kernel_size * hidden_factor)
            assert freq_expand_factor % 4 == 0, 'signle_chans should be a multiple of 4'
            single_chans = freq_expand_factor // 4
            self.f_conv1 = ConvModule(
                in_channels=int(in_chans * self.split_ratio),
                out_channels=freq_expand_factor,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

            self.f_conv2_1 = ConvModule(
                in_channels=single_chans,
                out_channels=single_chans,
                kernel_size=weight_k_size,
                stride=1,
                groups=single_chans,
                padding=weight_k_size // 2,
            )

            self.f_conv2_2 = ConvModule(
                in_channels=single_chans,
                out_channels=single_chans,
                kernel_size=weight_k_size,
                stride=1,
                groups=single_chans,
                padding=(weight_k_size // 2) * 2,
                dilation=2,
            )
            self.f_conv2_3 = ConvModule(
                in_channels=single_chans,
                out_channels=single_chans,
                kernel_size=(1, weight_k_size * 2 - 1),
                stride=1,
                groups=single_chans,
                padding=(0, weight_k_size - 1),
            )
            self.f_conv2_4 = ConvModule(
                in_channels=single_chans,
                out_channels=single_chans,
                kernel_size=(weight_k_size * 2 - 1, 1),
                stride=1,
                groups=single_chans,
                padding=(weight_k_size - 1, 0),
            )

            self.f2_norm = nn.GroupNorm(num_groups=4, num_channels=freq_expand_factor)
            self.f2_act = build_activation_layer(act_cfg)

            self.f_conv3 = ConvModule(
                in_channels=freq_expand_factor,
                out_channels=kernel_size * kernel_size,
                kernel_size=1,
                bias=True,
            )

        if gate_mode == 'softmax':
            self.gate = nn.Softmax(dim=-1)
        elif gate_mode == 'sigmoid':
            self.gate = nn.Sigmoid()
        else:
            assert False, f'Unsupported gate mode: {gate_mode}'

        self.register_buffer('dct_2d', self.build_2d_dct(kernel_size))

    @staticmethod
    def build_filter(pos: int, freq: int, size: int) -> torch.Tensor:
        dct_filter = torch.cos(torch.tensor(torch.pi * freq * (pos + 0.5) / size))
        if freq == 0:
            return dct_filter * torch.sqrt(torch.tensor(1 / size))
        else:
            return dct_filter * torch.sqrt(torch.tensor(2 / size))

    def build_1d_dct(self, size: int):
        """
        For each position, construct a frequency representation of size
        """
        dct_filter = torch.zeros((size, size))
        for freq in range(size):
            for pos in range(size):
                dct_filter[freq][pos] = self.build_filter(pos, freq, size)
        return dct_filter

    def build_2d_dct(self, size: int):
        """
        apply Kronecker product to construct 2D DCT
        """
        t_1d = self.build_1d_dct(size)
        t_2d = torch.kron(t_1d, t_1d)
        return t_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W)
        x = x + self.s_conv2(x)
        x = x + self.s_conv3(x)

        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        b, c, h, w = x.size()
        left_chans = int(c * self.split_ratio)
        right_chans = c - left_chans
        x_left, x_right = torch.split(x, [left_chans, right_chans], dim=1)

        # (B, C1, H, W) -> (B, C1, new_h, new_w, kernel_h, kernel_w)
        unfolded = x_left.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # (B, C1, new_h, new_w, kernel_h * kernel_w)
        unfolded = rearrange(unfolded, 'b c new_h new_k k_h k_w -> b c new_h new_k (k_h k_w)')
        # (B, C1, new_h, new_w, kernel_h * kernel_w)
        dct_trans = torch.matmul(unfolded, self.dct_2d.transpose(0, 1))

        # (B, C1, H, W) -> (B, C2, new_h, new_w)
        avg_unfolded = F.avg_pool2d(x_right, self.kernel_size, self.stride)
        # (B, C2, new_h, new_w) -> (B, 1, new_h, new_w, kernel_h * kernel_w)
        f_attn = self.f_conv1(unfolded.mean(dim=-1, keepdim=False))

        x_0, x_1, x_2, x_3 = f_attn.chunk(4, dim=1)
        x_0 = self.f_conv2_1(x_0)
        x_1 = self.f_conv2_2(x_1)
        x_2 = self.f_conv2_3(x_2)
        x_3 = self.f_conv2_4(x_3)
        f_attn = f_attn + torch.cat((x_0, x_1, x_2, x_3), dim=1)
        f_attn = self.f_conv3(self.f2_act(self.f2_norm(f_attn)))
        f_attn = self.gate(f_attn)

        f_attn = f_attn.permute(0, 2, 3, 1).contiguous().unsqueeze(1)
        # (B, C1, new_h, new_w, kernel_h * kernel_w)
        dct_trans = dct_trans * f_attn
        # (B, C1, new_h, new_w, 1)
        dct_trans = dct_trans.sum(dim=-1, keepdim=True)

        # (B, C1, new_h, new_w, kernel_h * kernel_w)
        dct_trans = dct_trans.expand(b, left_chans, unfolded.size(2), unfolded.size(3),
                                     self.kernel_size * self.kernel_size)
        dct_trans = dct_trans.clone()
        dct_trans[..., 1:] = 0

        # (B, C1, new_h, new_w, kernel_h * kernel_w)
        idct_trans = torch.matmul(dct_trans, self.dct_2d)

        return torch.cat((idct_trans.mean(dim=-1, keepdim=False), avg_unfolded), dim=1)
