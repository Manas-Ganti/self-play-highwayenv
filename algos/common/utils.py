"""Shared training utilities: seeding, devices, and gradient diagnostics."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and torch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device string, falling back to CPU when the accelerator is absent.

    The target machine is a single A100 (CLAUDE.md §1), but this must not hard-fail
    on a laptop -- the same code runs both places.
    """

    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if req == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(req)


def global_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of the current gradient, across all parameters."""

    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


class GradientVarianceTracker:
    """Tracks per-update variance of the policy gradient.

    This is how RQ1 gets an explanation rather than just a verdict: GRPO replaces
    GAE's learned baseline with a group mean, and the theory says that costs
    variance. CLAUDE.md §4.2 requires logging it for *both* algorithms so the
    group-baseline vs GAE gap is measured rather than assumed.

    Uses Welford's algorithm over the flattened policy gradient of each minibatch,
    then reports the mean per-coordinate variance.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def update(self, params) -> None:
        """Record the gradient currently sitting on ``params``.

        Accumulated on CPU in float64: MPS has no float64 at all, and Welford's
        update on a near-zero-mean gradient loses too much precision in float32 to
        trust the PPO-vs-GRPO variance comparison this exists to make. The policy
        is a small MLP, so the copy is cheap.
        """

        grads = [p.grad.detach().flatten().cpu() for p in params if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads).double()

        self._n += 1
        if self._mean is None:
            self._mean = torch.zeros_like(flat)
            self._m2 = torch.zeros_like(flat)
        delta = flat - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (flat - self._mean)

    @property
    def variance(self) -> float:
        """Mean per-coordinate gradient variance across the update's minibatches."""

        if self._m2 is None or self._n < 2:
            return 0.0
        return float((self._m2 / (self._n - 1)).mean().item())

    @property
    def gradient_norm(self) -> float:
        """Norm of the mean gradient across the update's minibatches."""

        if self._mean is None:
            return 0.0
        return float(self._mean.norm().item())

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio ||E[g]||^2 / mean Var[g] -- the headline comparison."""

        var = self.variance
        if var <= 0.0:
            return 0.0
        return float(self.gradient_norm**2 / var)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - Var(y - y_pred) / Var(y). PPO critic health; undefined (nan) for GRPO."""

    var_y = float(np.var(y_true))
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)
