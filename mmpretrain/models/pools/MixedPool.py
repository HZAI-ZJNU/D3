import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MixedPool']


class MixedPool(nn.Module):
    def __init__(self, kernel_size: int, stride: int, alpha: float = 0.5, padding: int = 0, dilation: int = 1):
        # nn.Module.__init__(self)
        super(MixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)  # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation) + (
                1 - self.alpha) * F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        # print(self.alpha)
        # print(x.size())
        return x
