"""Freivalds-style randomised verification of matrix products and merges.

Checking ``C == A @ B`` directly costs a full matrix multiply. Freivalds'
test multiplies by random vectors instead: ``C r == A (B r)`` costs
``O(n^2)`` per trial, and a wrong ``C`` passes one trial with probability
at most 1/2 for {0,1} vectors (far less for Gaussian ``r``, up to floating
point tolerance). Used here to certify that exported artefacts compute what
they claim, i.e. that a merged weight equals ``W0 + s * B A`` without forming ``B A``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FreivaldsReport:
    """Outcome of a randomised check. ``max_rel_residual`` is the worst trial."""

    passed: bool
    trials: int
    max_rel_residual: float


def freivalds(A: Tensor, B: Tensor, C: Tensor, trials: int = 8, rtol: float = 1e-4, seed: int = 0) -> FreivaldsReport:
    """Probabilistically check ``C == A @ B``."""
    if A.shape[1] != B.shape[0] or C.shape != (A.shape[0], B.shape[1]):
        raise ValueError("shape mismatch")
    gen = torch.Generator().manual_seed(seed)
    # all trials at once: R is (n, trials); one mat-mat with a thin matrix per operand
    R = torch.randn(B.shape[1], trials, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        lhs = C.detach().double() @ R
        rhs = A.detach().double() @ (B.detach().double() @ R)
    worst = float(((lhs - rhs).norm(dim=0) / rhs.norm(dim=0).clamp_min(1e-30)).max())
    return FreivaldsReport(worst <= rtol, trials, worst)


def verify_lora_merge(
    merged: Tensor,
    base: Tensor,
    lora_A: Tensor,
    lora_B: Tensor,
    scaling: float,
    trials: int = 8,
    rtol: float = 1e-4,
    seed: int = 0,
) -> FreivaldsReport:
    """Check ``merged == base + scaling * B @ A`` in ``O((out + in) * rank)`` per trial (plus two mat-vecs)."""
    gen = torch.Generator().manual_seed(seed)
    R = torch.randn(merged.shape[1], trials, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        lhs = merged.detach().double() @ R
        rhs = base.detach().double() @ R + scaling * (lora_B.detach().double() @ (lora_A.detach().double() @ R))
    worst = float(((lhs - rhs).norm(dim=0) / rhs.norm(dim=0).clamp_min(1e-30)).max())
    return FreivaldsReport(worst <= rtol, trials, worst)
