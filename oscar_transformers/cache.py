"""`transformers.cache_utils.Cache` subclass implementing OSCAR's mixed
sink + INT2 middle + recent KV cache.

The cache assumes the model has already had K/V projections rotated via
:func:`oscar_transformers.rotation.bake_rotations` (the default "baked" mode).
A debug-only "online" mode is available via :mod:`oscar_transformers._online`.

Layout per layer
----------------

Each :class:`OSCARCacheLayer` keeps three regions:

- **sink** (first ``sink_tokens`` tokens, default 64): full-precision K, V.
- **recent** (last ``recent_tokens`` tokens, default 256): full-precision K, V.
- **middle** (everything in between): per-token group-128 asymmetric INT2,
  with separate K/V clip ratios.

As new tokens stream in via :meth:`update`, the recent window fills FIFO; once
it exceeds ``recent_tokens`` the oldest recent tokens are quantized and
appended to the middle region. The sink region never changes after the first
``sink_tokens`` tokens are written.

The cache is *not* compatible with :meth:`Cache.crop` (used by the parent
delta-mem-tests runner for cross-question reuse). Crop is disabled for
non-bf16 backends in ``run/_chunked_eval_runner.py``; OSCAR is no exception.

Dequant shadow cache
--------------------

By default an incrementally-maintained dequantized middle is held in bf16
alongside the packed codes (see ``_middle_k_dq`` / ``_middle_v_dq``) so
:meth:`_assemble` does not re-dequantize the entire middle each decode step.
At 17 k context this shadow adds ~2.45 GB persistent VRAM. Set the env var
``OSCAR_DISABLE_DEQUANT_SHADOW=1`` to skip it — :meth:`_assemble` will
dequantize the middle on demand each call (~40 ms/layer extra, ~24 min added
on a conv-0/10q eval). Useful when context approaches the VRAM ceiling.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .quantize import QuantizedBlock, concat_blocks, dequantize, quantize_per_token


_SHADOW_DISABLED = os.environ.get("OSCAR_DISABLE_DEQUANT_SHADOW", "0") not in ("", "0", "false", "False")


class OSCARCacheLayer(CacheLayerMixin):
    """Per-layer OSCAR cache. See :class:`OSCARCache` for context."""

    is_sliding = False

    def __init__(
        self,
        *,
        sink_tokens: int = 64,
        recent_tokens: int = 256,
        bits: int = 2,
        group_size: int = 128,
        k_clip: float = 0.96,
        v_clip: float = 0.92,
    ) -> None:
        super().__init__()
        self.sink_tokens = int(sink_tokens)
        self.recent_tokens = int(recent_tokens)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.k_clip = float(k_clip)
        self.v_clip = float(v_clip)
        self.is_initialized = False
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None
        self.sink_k: Optional[torch.Tensor] = None
        self.sink_v: Optional[torch.Tensor] = None
        self.middle_k: Optional[QuantizedBlock] = None
        self.middle_v: Optional[QuantizedBlock] = None
        self.recent_k: Optional[torch.Tensor] = None
        self.recent_v: Optional[torch.Tensor] = None
        # Decode-time fast path: maintain the dequantized middle region
        # incrementally so :meth:`_assemble` does not re-dequantize the entire
        # ~17 k-token INT2 middle on every decoded token. Each spill from
        # ``recent`` into ``middle`` dequantizes only the spilled chunk and
        # appends it to ``_middle_k_dq`` / ``_middle_v_dq``. This caches the
        # bf16 cost in exchange for ~2× the middle's VRAM (codes + dequant).
        self._middle_k_dq: Optional[torch.Tensor] = None
        self._middle_v_dq: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return (
            (0 if self.sink_k is None else self.sink_k.shape[2])
            + (0 if self.middle_k is None else self.middle_k.codes.shape[2])
            + (0 if self.recent_k is None else self.recent_k.shape[2])
        )

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0

    def reset(self) -> None:
        self.is_initialized = False
        self.sink_k = self.sink_v = None
        self.middle_k = self.middle_v = None
        self.recent_k = self.recent_v = None
        self._middle_k_dq = None
        self._middle_v_dq = None

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        raise NotImplementedError("OSCARCacheLayer does not support beam search yet.")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append ``key_states, value_states`` and return the assembled
        (sink + middle dequant + recent) K, V slabs.

        ``key_states`` and ``value_states`` arrive shaped
        ``(batch, num_kv_heads, new_tokens, head_dim)`` already in the rotated
        basis (because the model was passed through :func:`bake_rotations`).
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        n_new = key_states.shape[2]
        offset = 0

        sink_have = 0 if self.sink_k is None else self.sink_k.shape[2]
        sink_remaining = self.sink_tokens - sink_have
        if sink_remaining > 0 and n_new > 0:
            take = min(sink_remaining, n_new)
            new_sink_k = key_states[:, :, offset:offset + take, :]
            new_sink_v = value_states[:, :, offset:offset + take, :]
            self.sink_k = new_sink_k if self.sink_k is None else torch.cat([self.sink_k, new_sink_k], dim=2)
            self.sink_v = new_sink_v if self.sink_v is None else torch.cat([self.sink_v, new_sink_v], dim=2)
            offset += take

        if offset < n_new:
            new_recent_k = key_states[:, :, offset:, :]
            new_recent_v = value_states[:, :, offset:, :]
            self.recent_k = new_recent_k if self.recent_k is None else torch.cat([self.recent_k, new_recent_k], dim=2)
            self.recent_v = new_recent_v if self.recent_v is None else torch.cat([self.recent_v, new_recent_v], dim=2)

        if self.recent_k is not None and self.recent_k.shape[2] > self.recent_tokens:
            overflow = self.recent_k.shape[2] - self.recent_tokens
            spill_k = self.recent_k[:, :, :overflow, :]
            spill_v = self.recent_v[:, :, :overflow, :]
            self.recent_k = self.recent_k[:, :, overflow:, :]
            self.recent_v = self.recent_v[:, :, overflow:, :]

            quant_k = quantize_per_token(
                spill_k, bits=self.bits, group_size=self.group_size, clip_ratio=self.k_clip,
            )
            quant_v = quantize_per_token(
                spill_v, bits=self.bits, group_size=self.group_size, clip_ratio=self.v_clip,
            )
            self.middle_k = quant_k if self.middle_k is None else concat_blocks((self.middle_k, quant_k))
            self.middle_v = quant_v if self.middle_v is None else concat_blocks((self.middle_v, quant_v))

            # Maintain the dequantized-middle cache incrementally so :meth:`_assemble`
            # does not re-dequantize the entire INT2 middle on every decoded token.
            # Dequant cost goes from O(total_middle) per step to O(spill_size) per spill.
            # Set OSCAR_DISABLE_DEQUANT_SHADOW=1 to skip this (trades ~2.45 GB at 17k
            # for ~40 ms/layer extra per decode step).
            if not _SHADOW_DISABLED:
                dtype = self.dtype or torch.bfloat16
                dq_k = dequantize(quant_k, dtype=dtype)
                dq_v = dequantize(quant_v, dtype=dtype)
                if self._middle_k_dq is None:
                    self._middle_k_dq = dq_k
                    self._middle_v_dq = dq_v
                else:
                    self._middle_k_dq = torch.cat([self._middle_k_dq, dq_k], dim=2)
                    self._middle_v_dq = torch.cat([self._middle_v_dq, dq_v], dim=2)

        return self._assemble()

    def _assemble(self) -> Tuple[torch.Tensor, torch.Tensor]:
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []
        if self.sink_k is not None:
            parts_k.append(self.sink_k)
            parts_v.append(self.sink_v)
        if self._middle_k_dq is not None:
            parts_k.append(self._middle_k_dq)
            parts_v.append(self._middle_v_dq)
        elif self.middle_k is not None:
            # Shadow disabled — dequantize on demand. Transient bf16 tensor lives
            # only for this _assemble call; the subsequent torch.cat consumes it.
            dtype = self.dtype or torch.bfloat16
            parts_k.append(dequantize(self.middle_k, dtype=dtype))
            parts_v.append(dequantize(self.middle_v, dtype=dtype))
        if self.recent_k is not None:
            parts_k.append(self.recent_k)
            parts_v.append(self.recent_v)
        if not parts_k:
            empty = torch.empty(0, device=self.device or "cpu")
            return empty, empty
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    def snapshot(self) -> dict:
        """Capture the full layer state into a CPU-resident dict that
        :meth:`restore_from` can later replay.

        Cross-question cache reuse: ``Cache.crop(history_len)`` is
        fundamentally hard for OSCAR because the INT2 middle stores per-
        token-per-group scale/zero with group-128 boundaries that don't
        align with arbitrary truncation lengths. Snapshot/restore at a
        single checkpoint (typically end-of-history prefill) sidesteps the
        boundary problem entirely: we capture every per-region tensor as
        it stands and replay it verbatim before each subsequent question.

        Tensors are moved to CPU so the snapshot survives independent of
        GPU lifecycle and doesn't compete for VRAM with the live decode.
        """
        def _clone_cpu(t):
            return None if t is None else t.detach().to("cpu").clone()

        def _clone_block_cpu(b):
            if b is None:
                return None
            return QuantizedBlock(
                codes=b.codes.detach().to("cpu").clone(),
                scale=b.scale.detach().to("cpu").clone(),
                zero=b.zero.detach().to("cpu").clone(),
                bits=b.bits,
                group_size=b.group_size,
                unpacked_d=b.unpacked_d,
            )

        return {
            "is_initialized": self.is_initialized,
            "dtype": self.dtype,
            "device": str(self.device) if self.device is not None else None,
            "sink_k": _clone_cpu(self.sink_k),
            "sink_v": _clone_cpu(self.sink_v),
            "middle_k": _clone_block_cpu(self.middle_k),
            "middle_v": _clone_block_cpu(self.middle_v),
            "recent_k": _clone_cpu(self.recent_k),
            "recent_v": _clone_cpu(self.recent_v),
            "middle_k_dq": _clone_cpu(self._middle_k_dq),
            "middle_v_dq": _clone_cpu(self._middle_v_dq),
        }

    def restore_from(self, state: dict, device: Optional[torch.device] = None) -> None:
        """Replace this layer's state with a snapshot. Tensors move back to
        the requested device (or the snapshot's recorded device).
        """
        target_device = device or torch.device(state["device"] or "cpu")
        def _to_dev(t):
            return None if t is None else t.to(target_device)

        def _block_to_dev(b):
            if b is None:
                return None
            return QuantizedBlock(
                codes=b.codes.to(target_device),
                scale=b.scale.to(target_device),
                zero=b.zero.to(target_device),
                bits=b.bits,
                group_size=b.group_size,
                unpacked_d=b.unpacked_d,
            )

        self.is_initialized = bool(state["is_initialized"])
        self.dtype = state["dtype"]
        self.device = target_device
        self.sink_k = _to_dev(state["sink_k"])
        self.sink_v = _to_dev(state["sink_v"])
        self.middle_k = _block_to_dev(state["middle_k"])
        self.middle_v = _block_to_dev(state["middle_v"])
        self.recent_k = _to_dev(state["recent_k"])
        self.recent_v = _to_dev(state["recent_v"])
        self._middle_k_dq = _to_dev(state["middle_k_dq"])
        self._middle_v_dq = _to_dev(state["middle_v_dq"])


class OSCARCache(Cache):
    """OSCAR mixed-precision KV cache.

    Parameters
    ----------
    config:
        The model's config (any object with ``num_hidden_layers``).
    sink_tokens:
        Number of leading tokens kept in full precision. Default 64 matches
        RotationZoo / sglang ``SGLANG_MIXED_KV_PREFIX_TOKENS=64``.
    recent_tokens:
        Number of trailing tokens kept in full precision. Default 256 matches
        ``SGLANG_MIXED_KV_RECENT_TOKENS=256``.
    bits:
        INT bits for the middle region. Default 2.
    group_size:
        Quantization group size along head_dim. Default 128.
    k_clip:
        Clip ratio for K quantization. Default 0.96 (RotationZoo).
    v_clip:
        Clip ratio for V quantization. Default 0.92 (RotationZoo).
    """

    def __init__(
        self,
        *,
        config: Any,
        sink_tokens: int = 64,
        recent_tokens: int = 256,
        bits: int = 2,
        group_size: int = 128,
        k_clip: float = 0.96,
        v_clip: float = 0.92,
    ) -> None:
        n_layers = int(getattr(config, "num_hidden_layers"))
        layers = [
            OSCARCacheLayer(
                sink_tokens=sink_tokens,
                recent_tokens=recent_tokens,
                bits=bits,
                group_size=group_size,
                k_clip=k_clip,
                v_clip=v_clip,
            )
            for _ in range(n_layers)
        ]
        super().__init__(layers=layers)

    def snapshot(self) -> list:
        """Return a list of per-layer snapshots (see
        :meth:`OSCARCacheLayer.snapshot`). The list is the durable artifact
        — pass it back to :meth:`restore_from` to roll the whole cache back
        to this point.
        """
        return [layer.snapshot() for layer in self.layers]

    def restore_from(self, snapshots: list, device: Optional[torch.device] = None) -> None:
        if len(snapshots) != len(self.layers):
            raise ValueError(
                f"snapshot has {len(snapshots)} layers; cache has {len(self.layers)}"
            )
        for layer, snap in zip(self.layers, snapshots):
            layer.restore_from(snap, device=device)
