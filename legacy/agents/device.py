"""Safe torch device resolution.

Per CLAUDE.md: never assume MPS is available. Always check and fall back to CPU
gracefully. Centralised here so every agent and script resolves devices the same
way.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("racetrack_rl")


def resolve_device(requested: str) -> str:
    """Resolve a requested torch device to one that is actually available.

    Parameters
    ----------
    requested : str
        ``"mps"``, ``"cuda"``, or ``"cpu"``.

    Returns
    -------
    str
        A usable device string, falling back to ``"cpu"`` when the requested
        accelerator is unavailable.
    """

    import torch

    req = requested.lower()
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but unavailable; falling back to cpu.")
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but unavailable; falling back to cpu.")
        return "cpu"
    return "cpu"
