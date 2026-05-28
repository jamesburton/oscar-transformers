# oscar-transformers

OSCAR 2-bit KV cache quantization, ported into the
`transformers.cache_utils.Cache` API so it can be used with stock Hugging Face
models without depending on SGLang.

The OSCAR algorithm (paper: [arXiv:2605.17757](https://arxiv.org/abs/2605.17757),
upstream code: [FutureMLS-Lab/OSCAR](https://github.com/FutureMLS-Lab/OSCAR))
runs an offline covariance-aware spectral rotation on attention K/V before
quantizing to INT2. The rotation flattens per-channel outliers so the 2-bit
grid can represent the rotated tensor without destroying attention scores.

This package implements two things:

1. **`bake_rotations(model, k_rotation_path, v_rotation_path)`** — a one-shot
   utility that pre-multiplies the rotations into `q_proj`, `k_proj`, `v_proj`,
   and `o_proj` weights in-place. After baking, the model's attention numerics
   are mathematically equivalent to the original (rotations are orthogonal) but
   the K/V tensors that reach the cache are already in the rotated basis.

2. **`OSCARCache`** — a `transformers.cache_utils.Cache` subclass that
   quantizes the (already-rotated) K/V to per-token asymmetric INT2 with
   group size 128, keeping a small FP16 sink window (first N tokens) and a
   recent-token FP16 window. K and V each have their own clip ratio
   (defaults match RotationZoo: K=0.96, V=0.92).

A debug-only `OSCARCache(mode="online")` path skips the weight baking and
applies rotations on the fly inside the cache (paired with an attention-module
patch for Q). Use it only to cross-check the baked path numerically on a small
slice; it is not production code.

## Status

PyTorch reference implementation, **not yet runnable**. Targets:

- [ ] Load `{k,v}_rotation_*.pt` files from `Zhongzhu/OSCAR-RotationZoo`
- [ ] `bake_rotations` for Qwen3-style attention (q/k/v/o `nn.Linear` projections)
- [ ] `OSCARCache.update` with group-128 asymmetric INT2 + sink + recent windows
- [ ] Smoke test: bake Thinking-2507 rotations onto Qwen3-4B-Instruct-2507 and
      verify the cache returns sensible logits on a 100-token prompt
- [ ] End-to-end run on `delta-mem-tests` LoCoMo conv-0 / first 10 questions,
      compared to the bf16, TQ4, and HQQ2 baselines committed in the parent
      repo

## Install (planned)

```bash
pip install -e .
```

The package is also intended to be vendored as a git submodule by
`delta-mem-tests` under `third_party/oscar-transformers`.

## Licensing & attribution

This is an independent re-implementation, MIT-licensed. The algorithm and the
calibrated rotation matrices are the work of the OSCAR authors; see `NOTICE`
for full attribution.
