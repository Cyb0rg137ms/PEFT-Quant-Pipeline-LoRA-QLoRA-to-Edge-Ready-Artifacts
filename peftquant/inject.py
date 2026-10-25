"""Inject adapters into arbitrary models, count parameters, and merge for deployment."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

from torch import nn

from peftquant.dora import DoRALinear
from peftquant.lora import LoRALinear

Kind = Literal["lora", "dora"]


def _dora_from_linear(layer: nn.Linear, rank: int, alpha: float, dropout: float) -> DoRALinear:
    m = DoRALinear(layer.in_features, layer.out_features, rank, alpha, dropout)
    m = m.to(device=layer.weight.device, dtype=layer.weight.dtype)
    m.weight.data.copy_(layer.weight.data)
    if layer.bias is not None:
        m.bias.data.copy_(layer.bias.data)
    else:
        m.bias.data.zero_()
    m.magnitude.data.copy_(m.weight.norm(dim=1).clamp(min=1e-6))
    return m


def inject_adapters(
    model: nn.Module,
    target: str | Callable[[str, nn.Linear], bool] = r".*",
    kind: Kind = "lora",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> list[str]:
    """Replace matching ``nn.Linear`` sub-modules in place; returns replaced names.

    ``target`` is a regex matched (``re.fullmatch``) against the dotted module
    name, or a predicate ``(name, module) -> bool``. With ``freeze_base`` every
    non-adapter parameter of the model is frozen.
    """
    pred = target if callable(target) else (lambda n, _m, rx=re.compile(target): bool(rx.fullmatch(n)))
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if type(child) is nn.Linear and pred(full, child):
                new: nn.Module = (
                    LoRALinear.from_linear(child, rank, alpha, dropout)
                    if kind == "lora"
                    else _dora_from_linear(child, rank, alpha, dropout)
                )
                setattr(module, child_name, new)
                replaced.append(full)
    if not replaced:
        raise ValueError("no nn.Linear matched the target")
    return replaced


def merge_adapters(model: nn.Module) -> int:
    """Replace every LoRA/DoRA module with a plain ``nn.Linear``; returns the count."""
    n = 0
    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.to_linear())
                n += 1
            elif isinstance(child, DoRALinear):
                lin = nn.Linear(child.in_features, child.out_features).to(child.weight.device, child.weight.dtype)
                lin.weight.data.copy_(child.merge_weights().detach())
                lin.bias.data.copy_(child.bias.data)
                setattr(module, child_name, lin)
                n += 1
    return n


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[nn.Module]:
    """Temporarily run every LoRA layer as its frozen base (e.g. the GRPO/KL reference policy)."""
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in layers]
    for m in layers:
        m.enabled = False
    try:
        yield model
    finally:
        for m, p in zip(layers, prev, strict=True):
            m.enabled = p
