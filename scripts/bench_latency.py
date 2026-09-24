#!/usr/bin/env python3
"""Synthetic 1-NFE latency (compile warmup + median predict). Not LIBERO accuracy."""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

import torch

from pi05_fast.checkpoint_compat import sanitize_snapflow_checkpoint
from pi05_fast.rocm_pi05_optim import install_rocm_pi05_eval_hooks
from pi05_fast.snapflow_pi05 import checkpoint_has_snapflow_mlp, install_pi05_snapflow_eval_hooks


def main() -> None:
    ckpt = Path(os.environ.get("POLICY_ID", Path.home() / "model_data/pi05_snapflow_1nfe")).expanduser()
    load = Path(os.environ.get("POLICY_LOAD", ckpt)).expanduser()
    n = int(os.environ.get("PI05_BENCH_ITERS", "30"))
    w = int(os.environ.get("PI05_BENCH_WARMUP", "8"))
    teacher = Path(os.environ.get("SNAPFLOW_TEACHER_POLICY", Path.home() / "model_data/pi05_libero_v044"))
    for note in sanitize_snapflow_checkpoint(load, teacher):
        print(f"policy checkpoint: {note}", flush=True)

    if checkpoint_has_snapflow_mlp(ckpt) or checkpoint_has_snapflow_mlp(load):
        install_pi05_snapflow_eval_hooks()
    install_rocm_pi05_eval_hooks()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print("loading", load, flush=True)
    policy = PI05Policy.from_pretrained(str(load))
    model = policy.model
    device = next(model.parameters()).device
    images = [
        torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32) for _ in range(3)
    ]
    img_masks = [torch.ones(1, device=device, dtype=torch.bool) for _ in range(3)]
    tokens = torch.randint(1, 1000, (1, 200), device=device)
    # Real prompts fill only part of the padded 200-token stream (LIBERO-10 uses ~52).
    lang_len = max(1, min(int(os.environ.get("PI05_BENCH_LANG_LEN", "52")), 200))
    masks = torch.zeros(1, 200, device=device, dtype=torch.bool)
    masks[:, :lang_len] = True

    def step():
        with torch.no_grad():
            return model.sample_actions(images, img_masks, tokens, masks)

    print("warmup / compile...", flush=True)
    torch.cuda.synchronize()
    for _ in range(w):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    print(
        f"n={n}  median={statistics.median(times):.2f} ms  "
        f"p10={times[max(0, n // 10)]:.2f}  p90={times[min(n - 1, 9 * n // 10)]:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
