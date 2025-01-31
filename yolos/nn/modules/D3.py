
import torch.nn as nn
import torch
import typing as t
import torch.nn.functional as F

from einops import rearrange
from .conv import Conv
from yolos.utils.act import build_act


class D3(nn.Module):
    """
    Dual-Domain Downsampling
    """

    def __init__(
            self,
            in_chans: int,
            kernel_size: int = 2,
            stride: t.Optional[int] = None,
            weight_k_size: int = 3,
            padding: int = 0,
            gate_mode: str = 'sigmoid',
            hidden_factor: int = 16,
            act: str = 'gelu',
            aggregate_mode: str = 'learned',
            split_ratio: float = 0.5,
    ) -> None:
        super(D3, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.aggregate_mode = aggregate_mode
        self.split_ratio = split_ratio

        self.s_conv2 = Conv(
            c1=in_chans,
            c2=in_chans,
            k=weight_k_size,
            d=1,
            g=in_chans,
            p=weight_k_size // 2,
            act=build_act(act)(),
        )

        self.s_conv3 = Conv(
            c1=in_chans,
            c2=in_chans,
            k=1,
            act=build_act(act)(),
        )

        if aggregate_mode == 'learned':
            freq_expand_factor = int(kernel_size * kernel_size * hidden_factor)
            assert freq_expand_factor % 4 == 0, 'signle_chans should be a multiple of 4'
            single_chans = freq_expand_factor // 4
            self.f_conv1 = Conv(
                c1=int(in_chans * self.split_ratio),
                c2=freq_expand_factor,
                k=1,
                act=build_act(act)(),
            )

            self.f_conv2_1 = Conv(
                c1=single_chans,
                c2=single_chans,
                k=weight_k_size,
                s=1,
                g=single_chans,
                p=weight_k_size // 2,
                act=build_act('relu')()
            )

            self.f_conv2_2 = Conv(
                c1=single_chans,
                c2=single_chans,
                k=weight_k_size,
                s=1,
                g=single_chans,
                p=(weight_k_size // 2) * 2,
                d=2,
                act=build_act('relu')()
            )

            self.f_conv2_3 = Conv(
                c1=single_chans,
                c2=single_chans,
                k=(1, weight_k_size * 2 - 1),
                s=1,
                g=single_chans,
                p=(0, weight_k_size - 1),
                act=build_act('relu')()
            )

            self.f_conv2_4 = Conv(
                c1=single_chans,
                c2=single_chans,
                k=(weight_k_size * 2 - 1, 1),
                s=1,
                g=single_chans,
                p=(weight_k_size - 1, 0),
                act=build_act('relu')()
            )

            self.f2_bn = nn.GroupNorm(num_groups=4, num_channels=freq_expand_factor)
            self.f2_act = build_act(act)()

            self.f_conv3 = nn.Conv2d(
                in_channels=freq_expand_factor,
                out_channels=kernel_size * kernel_size,
                kernel_size=1,
                bias=True
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
        # (B, C, H, W) -> (B, C, new_h, new_w, kernel_h, kernel_w)
        unfolded = x_left.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # (B, C, new_h, new_w, kernel_h * kernel_w)
        unfolded = rearrange(unfolded, 'b c new_h new_k k_h k_w -> b c new_h new_k (k_h k_w)')
        # (B, C, new_h, new_w, kernel_h * kernel_w)
        dct_trans = torch.matmul(unfolded, self.dct_2d.transpose(0, 1))

        # (B, C, new_h, new_w) -> (B, 1, new_h, new_w, kernel_h * kernel_w)
        # (B, C1, H, W) -> (B, C2, new_h, new_w)
        avg_unfolded = F.avg_pool2d(x_right, self.kernel_size, self.stride)

        f_attn = self.f_conv1(unfolded.mean(dim=-1, keepdim=False))
        x_0, x_1, x_2, x_3 = f_attn.chunk(4, dim=1)
        x_0 = self.f_conv2_1(x_0)
        x_1 = self.f_conv2_2(x_1)
        x_2 = self.f_conv2_3(x_2)
        x_3 = self.f_conv2_4(x_3)
        f_attn = f_attn + torch.cat((x_0, x_1, x_2, x_3), dim=1)
        f_attn = self.f_conv3(self.f2_act(self.f2_bn(f_attn)))
        f_attn = self.gate(f_attn)

        f_attn = f_attn.permute(0, 2, 3, 1).contiguous()
        # (B, C, new_h, new_w, kernel_h * kernel_w)

        dct_trans = dct_trans * f_attn.unsqueeze(1)
        # (B, C, new_h, new_w, 1)
        dct_trans = dct_trans.sum(dim=-1, keepdim=True)

        # (B, C, new_h, new_w, kernel_h * kernel_w)
        dct_trans = dct_trans.expand(b, left_chans, unfolded.size(2), unfolded.size(3),
                                     self.kernel_size * self.kernel_size)
        dct_trans = dct_trans.clone()
        dct_trans[..., 1:] = 0

        # (B, C, new_h, new_w, kernel_h * kernel_w)
        idct_trans = torch.matmul(dct_trans, self.dct_2d)

        return torch.cat((idct_trans.mean(dim=-1, keepdim=False), avg_unfolded), dim=1)
