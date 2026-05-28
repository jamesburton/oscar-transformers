"""Debug-only online-rotation mode.

This module exists so the baked-rotation path can be cross-validated
numerically. Where :func:`bake_rotations` permanently modifies the model's
weights, the online mode here leaves weights alone and instead:

1. Rotates ``K`` and ``V`` on the fly inside :class:`OnlineOSCARCache.update`,
   before quantization.
2. Patches the model's attention modules so ``Q`` is rotated on the right at
   query time, and the attention output is de-rotated before ``o_proj``.

After both modes are run on the same prompt with the same rotations, dequant
of the cached middle region and the final logits should agree to within
quantization noise. If they diverge by more than that, one of the two paths
has a bug.

Not for production. Slower and more invasive than the baked path.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .cache import OSCARCache
from .rotation import RotationSet


class OnlineOSCARCache(OSCARCache):
    """Variant of :class:`OSCARCache` that rotates K and V inline before
    quantizing. Pair with :func:`patch_attention_for_online_rotation` to
    rotate Q at query time.
    """

    def __init__(
        self,
        *,
        config: Any,
        k_rotations: RotationSet,
        v_rotations: RotationSet,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        if k_rotations.head_dim != v_rotations.head_dim:
            raise ValueError(
                f"K head_dim={k_rotations.head_dim} != V head_dim={v_rotations.head_dim}"
            )
        self._k_rotations = k_rotations
        self._v_rotations = v_rotations

    def _rotate_per_head(
        self, x: torch.Tensor, rotation: torch.Tensor
    ) -> torch.Tensor:
        rot = rotation.to(x.device, x.dtype)
        return torch.einsum("bhtd,de->bhte", x, rot)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r_k = self._k_rotations.layers[layer_idx].rotation
        r_v = self._v_rotations.layers[layer_idx].rotation
        k_rot = self._rotate_per_head(key_states, r_k)
        v_rot = self._rotate_per_head(value_states, r_v)
        return super().update(k_rot, v_rot, layer_idx, cache_kwargs)


@contextmanager
def patch_attention_for_online_rotation(
    model: nn.Module, *, k_rotations: RotationSet, v_rotations: RotationSet
):
    """Context manager that monkey-patches each attention block so Q is
    rotated by R_k before the attention dot product and the attention output
    is rotated by R_v.T (cancelling the V rotation we apply at cache write).

    Use only with :class:`OnlineOSCARCache`. Restores original forwards on
    exit.

    .. warning::

       The exact attention-forward signature varies by HF version and model.
       This is a reference patch for Qwen3-style attention modules; if it
       does not match, fall back to the baked path.
    """
    raise NotImplementedError(
        "Online attention-module patching is intentionally not implemented "
        "in this commit. The OnlineOSCARCache class exists so the K/V "
        "rotation path can be cross-checked in isolation; once it agrees "
        "with the baked path on a short prompt, this patcher can be filled "
        "in to support full online runs."
    )
    yield  # pragma: no cover
