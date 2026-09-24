# Measured results (source machine)

Machine: AMD Ryzen AI Max+ / Radeon 8060S (`gfx1151`), 2026-09-24.

Recipe: SnapFlow 1-NFE + W4A4 prefix LM + LM fp16 + fused quant +
`torch.compile(max-autotune-no-cudagraphs)` + batched SigLIP 776 prefix.

| Metric | Value |
|--------|--------|
| E2E action chunk, default prefix | **96.00 ms** median (p10 95.67, p90 98.00; 100 iterations) |
| E2E action chunk, language 64 + empty-camera cache | **79.88 ms** median (p10 76.31, p90 80.19; 100 iterations) |
| Embedl W4A4 `.mxr` (reference, not this tree) | ~92 ms |
| LIBERO-10, seed 0, 2 episodes × 10 tasks | **20/20 = 100%** |
| `n_action_steps` | 10 |
| `num_inference_steps` | 1 |

Eval artifacts from the development tree:
`results/libero10_maintainable_seed0_retry_20260923/`

Two episodes per task is a **smoke** for quantization/compile correctness, not a
full 500-episode LIBERO score.
