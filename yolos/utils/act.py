import torch.nn as nn
import typing as t
from functools import partial

__all__ = ['build_act', ]

REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hardswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def build_act(name: str) -> t.Union[nn.Module, t.Callable]:
    """
    build activation function based on param name
    """
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        return act_cls
    else:
        raise TypeError(f'no found act {name}')
