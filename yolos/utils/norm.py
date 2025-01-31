import torch.nn as nn
import typing as t
from functools import partial

__all__ = ['build_norm', ]

REGISTERED_NORM_DICT: dict[str, type] = {
    'bn': nn.BatchNorm2d,
    'gn': partial(nn.GroupNorm, 32),
    'ln': nn.LayerNorm,
    'in': nn.InstanceNorm2d
}
def build_norm(name: str) -> t.Union[nn.Module, t.Callable]:
    """
    build normalization function based on param name
    """
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        return norm_cls
    else:
        raise TypeError(f'no found norm {name}')