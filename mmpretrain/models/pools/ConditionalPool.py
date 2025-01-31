"""
This code was created by Ertugrul Bayraktar and Cihat Bora Yigit for their paper titled "Conditional-Pooling for Improved Data Transmission,"
which has been accepted as a journal paper in Pattern Recognition. The copyright of the code belongs to Ertugrul Bayraktar and Cihat Bora Yigit,
and it is licensed under the MIT License.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConditionalPoolingLayer']


class ConditionalPoolingLayer(nn.Module):
    """
    The class has an __init__ method that initializes the pooling layer with the specified size, stride, and padding.
    If no stride is provided, it defaults to the size of the pooling layer.

    For each channel, it extracts patches of size size with stride stride.
    It returns the pooled tensor following the concatenation of pooled channels.
    """

    def __init__(self, size, stride=None, padding=0):
        super(ConditionalPoolingLayer, self).__init__()
        self.size = size
        self.stride = stride if stride is not None else size
        self.padding = padding

    def forward(self, x):
        N = self.size
        S = self.stride

        batch = x.shape[0]  # Batch size
        chan = x.shape[1]  # Number of channels

        W = int((x.shape[2] - N) / S) + 1  # Width of output
        H = int((x.shape[3] - N) / S) + 1  # Height of output

        # Extract overlapping patches from the input image
        x_kernels = F.unfold(x, kernel_size=self.size, stride=self.stride, padding=self.padding).reshape(batch, chan,
                                                                                                         N * N, -1)

        # Permute dimensions for further processing
        x_kernels = x_kernels.permute(0, 1, 3, 2)

        # Calculate the mean of each patch
        mean = x_kernels.mean(dim=3, keepdim=True)

        # Count the number of values greater and less than the mean
        H_num = torch.sum(torch.greater(x_kernels, mean), dim=3, keepdim=True).float()
        L_num = torch.sum(torch.less(x_kernels, mean), dim=3, keepdim=True).float()

        # Perform Conditional-Pooling based on the number of values greater and less than the mean
        x_pooled = torch.where(torch.greater(H_num, L_num),
                               torch.divide(torch.sum(torch.where(torch.greater(x_kernels, mean), x_kernels, 0), dim=3,
                                                      keepdim=True), H_num),
                               torch.where(torch.less(H_num, L_num),
                                           torch.divide(
                                               torch.sum(torch.where(torch.less(x_kernels, mean), x_kernels, 0), dim=3,
                                                         keepdim=True), L_num), mean))

        # Reshape the output to the desired dimensions
        x_pooled = x_pooled.reshape(batch, chan, W, H)
        return x_pooled