"""OSCAR 2-bit KV cache for Hugging Face transformers.

Public API:

    from oscar_transformers import OSCARCache, bake_rotations, load_rotation_file

    rot_k = load_rotation_file("k_rotation_qqt_r_h_pbr.pt")
    rot_v = load_rotation_file("v_rotation_sst_r_h_pbr.pt")
    bake_rotations(model, k_rotations=rot_k, v_rotations=rot_v)
    cache = OSCARCache(config=model.config, k_clip=0.96, v_clip=0.92)
    model.generate(..., past_key_values=cache)
"""
from __future__ import annotations

from .cache import OSCARCache
from .rotation import apply_rotations, bake_rotations, load_rotation_file

__all__ = ["OSCARCache", "apply_rotations", "bake_rotations", "load_rotation_file"]
__version__ = "0.0.2"
