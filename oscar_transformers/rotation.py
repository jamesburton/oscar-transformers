"""Rotation loading and attention-forward patching.

OSCAR's rotation tensors are per-layer fp32 orthogonal matrices of shape
``(head_dim, head_dim)``. Two separate files exist per model — one for K
(``k_rotation_qqt_r_h_pbr.pt``) and one for V (``v_rotation_sst_r_h_pbr.pt``).

Format (per upstream ``compute_kv_rotation.py``)::

    {
        "format_version": 1,
        "objective": "<method>_<composition>",
        "source_grouping": "layer",
        "layers": {
            layer_id: {
                "layer_id": int,
                "rotation": Tensor (head_dim, head_dim) fp32,
                "eigenvalues": Tensor (head_dim,) fp32,
            },
            ...
        },
    }

Why we don't bake rotations into projection weights on Qwen3
-------------------------------------------------------------

Modern Qwen3 attention applies QK-Norm and rotary positional embeddings
between the q/k/v projections and the attention dot product::

    query_states = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
    key_states   = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

Neither the per-channel learnable scale γ inside ``q_norm`` / ``k_norm`` nor
the RoPE block-diagonal rotation commute with an arbitrary orthogonal R, so
pre-multiplying R into ``q_proj.weight`` / ``k_proj.weight`` produces wrong
outputs (verified empirically on Qwen3-4B-Instruct-2507: rotation-only output
collapses to repeated '?' tokens). V has no such preprocessing, so V-side
baking is mathematically valid, but for API simplicity we apply *all*
rotations inline.

:func:`apply_rotations` therefore monkey-patches the attention class's
``forward`` to inject the rotation at the right point:

* Rotate Q and K by ``R_k`` **after** q_norm/k_norm and RoPE, **before**
  ``past_key_values.update``.
* Rotate V by ``R_v`` before ``past_key_values.update``.
* Un-rotate the attention output by ``R_v.T`` before ``o_proj`` so the block
  output is unchanged in infinite precision.

The rotation is end-to-end mathematically invariant (R is orthogonal); the
benefit is that the K, V tensors that flow into the cache are in a basis
chosen offline to flatten per-channel outliers for INT2 quantization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn


@dataclass
class LayerRotation:
    layer_id: int
    rotation: torch.Tensor
    eigenvalues: torch.Tensor


@dataclass
class RotationSet:
    objective: str
    layers: Dict[int, LayerRotation]

    @property
    def head_dim(self) -> int:
        first = next(iter(self.layers.values()))
        return int(first.rotation.shape[0])

    def __len__(self) -> int:
        return len(self.layers)


def load_rotation_file(path: str | Path) -> RotationSet:
    """Load a ``k_rotation_*.pt`` or ``v_rotation_*.pt`` file produced by
    OSCAR's ``compute_kv_rotation.py``.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if "layers" not in blob:
        raise ValueError(
            f"{path}: missing 'layers' key; got top-level keys {list(blob.keys())}. "
            f"Expected an OSCAR rotation file (format_version=1)."
        )
    layers: Dict[int, LayerRotation] = {}
    for lid, entry in blob["layers"].items():
        rot = entry["rotation"].to(torch.float32)
        if rot.dim() != 2 or rot.shape[0] != rot.shape[1]:
            raise ValueError(
                f"layer {lid}: rotation has shape {tuple(rot.shape)}; "
                f"expected square (head_dim, head_dim)."
            )
        eig = entry.get("eigenvalues")
        eig_t = eig.to(torch.float32) if eig is not None else torch.empty(0)
        layers[int(lid)] = LayerRotation(
            layer_id=int(lid), rotation=rot, eigenvalues=eig_t,
        )
    return RotationSet(objective=str(blob.get("objective", "")), layers=layers)


def _iter_attention_blocks(model: nn.Module):
    """Yield (layer_id, attention_module) for each attention block in the model."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError(
            f"Could not find decoder layers on {type(model).__name__}; "
            f"expected `model.model.layers` (Qwen2/Qwen3 layout)."
        )
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise AttributeError(f"Layer {idx} has no `self_attn`.")
        yield idx, attn


_PATCHED_CLASSES: set[type] = set()


def _build_patched_forward():
    """Construct the patched forward function. Imported at patch time so
    transformers is not an import-time hard dependency of this module.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb, eager_attention_forward,
    )

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        r_k = self._oscar_k_rot.to(query_states.dtype)
        r_v = self._oscar_v_rot.to(value_states.dtype)
        query_states = torch.einsum("bhtd,de->bhte", query_states, r_k)
        key_states = torch.einsum("bhtd,de->bhte", key_states, r_k)
        value_states = torch.einsum("bhtd,de->bhte", value_states, r_v)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Runtime un-rotation by R_v.T before o_proj. Originally we tried
        # baking R_v.T into o_proj.weight to save one einsum per layer, but
        # empirical v3c/v3d evals showed quality regression on conv-0/10q
        # (likely because delta-mem's delta_o LoRA was trained against the
        # un-baked base.o_proj). The inline einsum is preserved as the
        # production path; see report/tier1-summary.md Appendix D.
        b, t = attn_output.shape[0], attn_output.shape[1]
        attn_output = attn_output.reshape(b, t, -1, self.head_dim)
        r_v_T = r_v.transpose(-1, -2).to(attn_output.dtype)
        attn_output = torch.einsum("bthd,de->bthe", attn_output, r_v_T)
        attn_output = attn_output.reshape(b, t, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return patched_forward


def _looks_like_delta_mem_attention(attn: nn.Module) -> bool:
    """Duck-type check for delta-Mem's DeltaMemAttention wrapper.

    delta-Mem wraps each ``Qwen3Attention`` (or ``SmolLM3Attention``) inside a
    ``DeltaMemAttention`` that re-implements attention internally to thread
    the delta-state through, instead of delegating to ``self.base.forward``.
    That means our class-level patch on ``Qwen3Attention.forward`` is bypassed
    entirely on the delta arm. We detect the wrapper by structure rather than
    importing it, to keep oscar-transformers free of a delta-Mem dependency.
    """
    return (
        hasattr(attn, "base")
        and hasattr(attn.base, "q_proj")
        and hasattr(attn, "_apply_standard_rotary")
        and hasattr(attn, "_normalize_value_states")
    )


def _bake_r_v_T_into_o_proj(o_proj: nn.Module, r_v: torch.Tensor) -> None:
    """Absorb the R_v.T per-head un-rotation into ``o_proj.weight`` in place.

    Math: at runtime the rotated attention output ``attn_rot`` (B, T, H, D)
    needs ``R_v.T`` applied per head before the linear ``W_o``. That is:

        attn_unrot[..., h, :] = attn_rot[..., h, :] @ R_v.T
        out = W_o @ attn_unrot.flatten()

    Equivalent baked form: ``W_o' = W_o @ block_diag(R_v.T, ..., R_v.T)`` with
    one ``R_v.T`` block per head. Then ``out = W_o' @ attn_rot.flatten()``
    skips the un-rotation entirely.

    Delta-mem compatibility: in ``DeltaMemAttention.forward``
    (``delta-Mem/deltamem/core/delta_impl.py:2280-2294``), ``delta_o`` is
    added *after* ``base.o_proj(attn_output)`` and so already lives in the
    un-rotated basis — no further adjustment needed.

    Idempotent: the original weight is saved as
    ``o_proj._oscar_original_weight`` on first call. Subsequent calls
    re-bake from the saved original, so changing rotations is safe.
    """
    if not hasattr(o_proj, "weight"):
        raise TypeError(
            f"_bake_r_v_T_into_o_proj expects a linear-like module with .weight; "
            f"got {type(o_proj).__name__}"
        )
    head_dim = int(r_v.shape[0])
    # Save original on first bake; restore from it on subsequent re-bakes.
    if not hasattr(o_proj, "_oscar_original_weight"):
        o_proj._oscar_original_weight = o_proj.weight.detach().clone()  # type: ignore[attr-defined]
    w = o_proj._oscar_original_weight  # type: ignore[attr-defined]
    out_features, in_features = w.shape
    if in_features % head_dim != 0:
        raise ValueError(
            f"o_proj in_features={in_features} not divisible by head_dim={head_dim}"
        )
    n_heads = in_features // head_dim
    # Cast rotation to match weight dtype; einsum stays in that dtype.
    # Bake in fp32 then cast back to weight dtype. At long context (e.g. 17k
    # tokens x 36 layers) doing the bake einsum directly in bf16 accumulates
    # enough precision loss in the baked weights to degrade end-to-end
    # quality on LoCoMo conv-0/10q (v3c showed base 0.1333 vs v2's 0.2379).
    # The runtime numerics still happen in the model's native dtype; only
    # the offline bake math is upgraded to fp32.
    target_dtype = w.dtype
    w_fp32 = w.to(torch.float32)
    r_v_fp32 = r_v.to(device=w.device, dtype=torch.float32)
    w_per_head = w_fp32.view(out_features, n_heads, head_dim)
    # The runtime un-rotation that we are absorbing is
    # ``O_orig[h, e] = sum_d O_rot[h, d] * R_v[e, d]``  (== O_rot @ R_v.T).
    # Bake: ``W_baked[o, h, d_new] = sum_e_old W[o, h, e_old] * R_v[e_old, d_new]``
    # i.e. ``W @ R_v`` per head. einsum "ohd,de->ohe" gives that exactly.
    w_baked_fp32 = torch.einsum("ohd,de->ohe", w_per_head, r_v_fp32).reshape(
        out_features, in_features,
    )
    w_baked = w_baked_fp32.to(target_dtype)
    with torch.no_grad():
        o_proj.weight.data.copy_(w_baked)


def _patch_delta_mem_instance(attn: nn.Module, r_k: torch.Tensor, r_v: torch.Tensor) -> None:
    """Per-instance rotation injection for a single DeltaMemAttention block.

    DeltaMemAttention.forward (delta-Mem/deltamem/core/delta_impl.py) does:

        query_states, key_states, value_states = self._apply_delta_qkv(...)
        query_states = self._normalize_query_states(query_states).transpose(1, 2)
        key_states   = self._normalize_key_states(key_states).transpose(1, 2)
        value_states = self._normalize_value_states(value_states).transpose(1, 2)
        query_states, key_states = self._apply_standard_rotary(...)
        # *** rotation injected HERE — before past_key_values.update ***
        past_key_values.update(key_states, value_states, ...)
        attn_output = attention_interface(self.base, query_states, key_states, value_states, ...)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        base_o_output = self.base.o_proj(attn_output)   # un-rotation BAKED into o_proj.weight
        attn_output = base_o_output + delta_o
        return attn_output, attn_weights

    Three instance-level patches:

    1. ``_apply_standard_rotary`` → also rotate Q and K by R_k after RoPE.
    2. ``_normalize_value_states`` → also rotate V by R_v after the norm.
       V is in shape (B, T, H, D) here (still pre-transpose); we rotate
       along the last (head_dim) axis, which is shape-agnostic.
    3. ``self.base.o_proj`` is wrapped with a Module that un-rotates by
       R_v.T along head_dim before applying the original o_proj. This is
       the un-rotation site because delta-Mem reshapes ``attn_output`` from
       (B, T, H, D) to (B, T, H*D) before calling base.o_proj. We previously
       tried baking R_v.T directly into o_proj.weight to save the wrapper
       overhead, but that degraded delta_o + base_o_output additive balance
       on long-context evals; see report/tier1-summary.md Appendix D.

    All three patches are on the INSTANCE (not the class). This keeps the
    patch local to one block and avoids any class-level side effects.
    """
    head_dim = attn.head_dim
    device = next(attn.parameters()).device
    dtype = next(attn.parameters()).dtype
    r_k_t = r_k.to(device=device, dtype=dtype)
    r_v_t = r_v.to(device=device, dtype=dtype)
    r_v_T = r_v_t.transpose(-1, -2).contiguous()

    if getattr(attn, "_oscar_delta_patched", False):
        # Re-apply: refresh buffers; wrapped o_proj reads attn._oscar_v_rot_T
        # at call time so the new value flows through automatically.
        attn._oscar_k_rot = r_k_t  # type: ignore[attr-defined]
        attn._oscar_v_rot = r_v_t  # type: ignore[attr-defined]
        attn._oscar_v_rot_T = r_v_T  # type: ignore[attr-defined]
        return

    attn._oscar_k_rot = r_k_t  # type: ignore[attr-defined]
    attn._oscar_v_rot = r_v_t  # type: ignore[attr-defined]
    attn._oscar_v_rot_T = r_v_T  # type: ignore[attr-defined]

    orig_apply_rotary = attn._apply_standard_rotary

    def patched_apply_standard_rotary(query_states, key_states, cos, sin):
        q, k = orig_apply_rotary(query_states, key_states, cos, sin)
        rk = attn._oscar_k_rot.to(q.dtype)
        q = torch.einsum("bhtd,de->bhte", q, rk)
        k = torch.einsum("bhtd,de->bhte", k, rk)
        return q, k

    attn._apply_standard_rotary = patched_apply_standard_rotary  # type: ignore[assignment]

    orig_normalize_v = attn._normalize_value_states

    def patched_normalize_value_states(states):
        out = orig_normalize_v(states)
        # ``out`` is (B, T, H, D) at this point — _normalize_value_states is
        # called before .transpose(1, 2) in DeltaMemAttention.forward. We
        # rotate the trailing head_dim axis which is shape-invariant.
        rv = attn._oscar_v_rot.to(out.dtype)
        return torch.einsum("...d,de->...e", out, rv)

    attn._normalize_value_states = patched_normalize_value_states  # type: ignore[assignment]

    orig_o_proj = attn.base.o_proj

    class _OSCARUnrotatingOProj(nn.Module):
        """Wraps the original o_proj. Un-rotates attn_output by R_v.T along
        head_dim before applying the original linear projection. The
        wrapper preserves the original o_proj weight unchanged so
        delta-mem's ``delta_o`` LoRA stays in its trained additive balance.
        """

        def __init__(self, orig, head_dim, host_attn):
            super().__init__()
            self._orig = orig
            self._head_dim = head_dim
            self._host_attn = host_attn

        def forward(self, x):
            b, t = x.shape[0], x.shape[1]
            x_rs = x.reshape(b, t, -1, self._head_dim)
            r_v_T = self._host_attn._oscar_v_rot_T.to(x_rs.dtype)
            x_unrot = torch.einsum("bthd,de->bthe", x_rs, r_v_T)
            x_back = x_unrot.reshape(b, t, -1).contiguous()
            return self._orig(x_back)

    attn.base.o_proj = _OSCARUnrotatingOProj(orig_o_proj, head_dim, attn)
    attn._oscar_delta_patched = True  # type: ignore[attr-defined]


def apply_rotations(
    model: nn.Module,
    *,
    k_rotations: RotationSet,
    v_rotations: RotationSet,
) -> None:
    """Wire OSCAR K and V rotations into ``model``.

    Two code paths depending on what wraps each layer's ``self_attn``:

    1. **Plain ``Qwen3Attention``** (the typical case, also the base arm of
       delta-mem-tests): attach ``_oscar_k_rot`` / ``_oscar_v_rot`` buffers
       to each attention module, replace the class's ``forward`` with a
       patched version that injects the rotation between
       q_norm/k_norm + RoPE and ``past_key_values.update``, and bake the
       R_v.T un-rotation directly into ``o_proj.weight`` so the runtime
       does not pay an extra einsum.

    2. **``DeltaMemAttention`` wrapper** (delta-Mem's delta arm): the wrapper
       re-implements attention internally and does not call
       ``self.base.forward``, so the class-level patch from (1) is bypassed.
       For each instance we monkey-patch two methods at instance level
       (``_apply_standard_rotary``, ``_normalize_value_states``) and bake
       R_v.T into ``self.base.o_proj.weight``; see
       :func:`_patch_delta_mem_instance`.

    Calling :func:`apply_rotations` repeatedly on the same model is safe:
    buffers are refreshed and patches are guarded by sentinel flags.
    """
    if k_rotations.head_dim != v_rotations.head_dim:
        raise ValueError(
            f"K head_dim={k_rotations.head_dim} != V head_dim={v_rotations.head_dim}"
        )

    seen_qwen3_classes: set[type] = set()
    attached_layers = 0
    for layer_idx, attn in _iter_attention_blocks(model):
        if layer_idx not in k_rotations.layers:
            raise KeyError(f"K rotations missing for layer {layer_idx}")
        if layer_idx not in v_rotations.layers:
            raise KeyError(f"V rotations missing for layer {layer_idx}")
        r_k = k_rotations.layers[layer_idx].rotation
        r_v = v_rotations.layers[layer_idx].rotation

        if _looks_like_delta_mem_attention(attn):
            _patch_delta_mem_instance(attn, r_k, r_v)
        else:
            device = next(attn.parameters()).device
            dtype = next(attn.parameters()).dtype
            attn.register_buffer(
                "_oscar_k_rot", r_k.to(device=device, dtype=dtype), persistent=False,
            )
            attn.register_buffer(
                "_oscar_v_rot", r_v.to(device=device, dtype=dtype), persistent=False,
            )
            # NOTE: the R_v.T un-rotation could be baked into ``o_proj.weight``
            # for a small decode-time win (one fewer einsum), but empirically
            # this breaks delta-mem's delta_o LoRA additive balance — see
            # report/tier1-summary.md Appendix D. The bake helper remains
            # in this module (_bake_r_v_T_into_o_proj) for experimentation
            # but is no longer applied here. Runtime un-rotation stays in
            # patched_forward.
            seen_qwen3_classes.add(type(attn))
        attached_layers += 1

    if attached_layers != len(k_rotations.layers):
        raise RuntimeError(
            f"Model has {attached_layers} attention layers but rotation file "
            f"has {len(k_rotations.layers)}. Refusing to leave the model "
            f"half-rotated."
        )

    if seen_qwen3_classes:
        patched_forward = _build_patched_forward()
        for cls in seen_qwen3_classes:
            if cls in _PATCHED_CLASSES:
                continue
            cls.forward = patched_forward
            _PATCHED_CLASSES.add(cls)


def bake_rotations(*args, **kwargs):
    """Deprecated. Qwen3's QK-Norm + RoPE pipeline means projection-weight
    baking is not mathematically valid; use :func:`apply_rotations` instead.
    """
    raise NotImplementedError(
        "bake_rotations was deprecated after empirical validation: Qwen3's "
        "q_norm/k_norm (per-channel γ) and RoPE both fail to commute with an "
        "arbitrary orthogonal rotation, so pre-multiplying R into the "
        "projection weights produces gibberish at inference time. Use "
        "apply_rotations(model, k_rotations=..., v_rotations=...) — it "
        "registers the rotations as buffers on each attention module and "
        "monkey-patches the attention forward to apply them after "
        "q_norm/k_norm+RoPE."
    )
