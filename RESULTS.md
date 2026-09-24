# Measured results (source machine)

Machine: AMD Ryzen AI Max+ / Radeon 8060S (`gfx1151`), 2026-09-24.

Recipe: SnapFlow 1-NFE + W4A4 prefix LM + LM fp16 + fused quant +
`torch.compile(max-autotune-no-cudagraphs)` + batched SigLIP 776 prefix.

| Metric | Value |
|--------|--------|
| E2E action chunk, default prefix | **95.03 ms** median (p10 94.64, p90 96.20; 100 iterations) |
| E2E action chunk, language 64 + empty-camera cache | **74.48 ms** median (p10 74.30, p90 74.80; 100 iterations) |
| Embedl W4A4 `.mxr` (reference, not this tree) | ~92 ms |
| LIBERO-10, seed 0, 2 episodes × 10 tasks | **20/20 = 100%** |
| `n_action_steps` | 10 |
| `num_inference_steps` | 1 |

Eval artifacts from the development tree:
`results/libero10_maintainable_seed0_retry_20260923/`

Two episodes per task is a **smoke** for quantization/compile correctness, not a
full 500-episode LIBERO score.

The WMMA epilogue update was also checked in-process against the prior source:
**97.78 → 96.29 ms** median over an interleaved 50-cycle ABBA run
(100 samples per side), a **1.49 ms** reduction. Full-model output was bit-exact.

An experimental `PI05_LANG_TOKENS=192` run reached **93.04 ms**, but its
LIBERO-10 seed-0 smoke was 19/20 rather than the maintained path's 20/20, so it
remains opt-in and is not included in the default result above.
