# Fast gfx1151 PyTorch recipe for OpenPI π0.5 SnapFlow (1-NFE)

## Measured results (2026-09-24)

Machine: AMD Ryzen AI Max+ / Radeon 8060S (`gfx1151`).

| Metric | Value |
|--------|--------|
| E2E action chunk latency | **95.03 ms** default; **91.50 ms** at 192 language tokens; **81.99 ms** at 64 tokens; **74.48 ms** at 64 tokens + empty-camera cache (100-run medians) |
| LIBERO-10 accuracy | Default: **20/20**. The 192-token mode produced **19/20** and **18/20** in two repeated seed-0 smoke runs |
| `n_action_steps` / `num_inference_steps` | 10 / 1 |
| Prefix tokens | 776 by default; **768** with `PI05_LANG_TOKENS=192`; **640** with `PI05_LANG_TOKENS=64` |

Two episodes per task is a **smoke** for quantization/compile correctness, not a full 500-episode LIBERO score.

## Software (ROCm)

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon 8060S (`gfx1151`) |
| Host ROCm (`/opt/rocm`) | **7.2.4** (70204-93) |
| `rocm-smi` | 4.0.0+97f5574fe2 |
| Python | 3.12 |
| PyTorch (venv) | **2.15.0a0+rocm10.1.0a20260822** |
| HIP in PyTorch (`torch.version.hip`) | **7.16.26332** |
| Wheel extra | `torch[device-gfx1151]` from `https://rocm.nightlies.amd.com/whl-multi-arch/` |
| Arch override | `HSA_OVERRIDE_GFX_VERSION=11.5.1` |

## What this is

Closed-loop **LeRobot PI05Policy** inference:

1. **SnapFlow 1-NFE** student (`Rylinjames/pi05-snapflow-distill-1nfe`)
2. **W4A4** packed INT4 GEMM on the **PaliGemma prefix LM only** (HIP custom op)
3. Prefix LM weights/activations in **fp16** (match W4A4; SigLIP stays bf16)
4. Shared activation quant for QKV and gate/up (`int4_gemm3` / `int4_gemm2`)
5. WMMA-fragment-major gate/up weights for coalesced static-weight loads
6. Cached row/column dequant scales in the WMMA epilogue; bounds-free full gate tiles
7. Batched SigLIP (3 cameras in one forward); empty-camera pool **256→64** → prefix **776**
8. **`torch.compile(max-autotune-no-cudagraphs, fullgraph=True)`** + gfx1151 L2 antialias
9. LM / expert attention **eager** (Inductor); SigLIP **SDPA**
10. `n_action_steps=10`, `num_inference_steps=1`

**pure PyTorch** path we have on this GPU.

## Hardware / OS

| Item | Value |
|------|--------|
| GPU | gfx1151 (e.g. Radeon 8060S / Ryzen AI Max+ 395) |
| ROCm | Host **7.2.4** (`/opt/rocm`) + `rocm[devel]` in the venv (see table above) |
| Python | 3.12 |
| Tools | `uv`, `git`, HIP compiler (for the INT4 extension) |

Set `HSA_OVERRIDE_GFX_VERSION=11.5.1` if the runtime does not detect gfx1151.

## One-time setup

```bash
cd pi05-pytorch-fast

# 1) Venv (does not touch onnx-infer / pi05-migraphx)
ROCM_ARCH=gfx1151 bash scripts/setup.sh -y
source env.sh

# 2) Weights + tokenizer + LIBERO sim (clones if missing)
# Must run after setup.sh: it needs the `hf` CLI installed into the venv.
bash scripts/download_checkpoints.sh all
# Needs HF token for gated paligemma tokenizer if google/paligemma2-3b-pt-224 is gated.
# LIBERO defaults to ~/model_data/libero; reuses ~/openpi/third_party/libero when present.
```

If you already have `~/.local/venvs/pi05-migraphx` from the parent project, you *can*
point `--venv` at it, but the intended distribute path is a **fresh**
`~/.local/venvs/pi05-pytorch-fast`.

## Pack W4A4 weights (once per checkpoint)

```bash
source env.sh
python scripts/pack_w4a4_prefix.py \
  --model ~/model_data/pi05_snapflow_1nfe \
  --out models/w4a4_prefix
```

`env.sh` sets `PI05_W4A4_PACK` to `models/w4a4_prefix`. First inference JIT-compiles
`pi05_fast/w4a4/csrc/int4_gemm.cu` (needs `ninja` + ROCm devel, already in setup).
PyTorch hipifies that source during the build; the generated `int4_gemm.hip` is
not source-controlled.

## W4A4 correctness smoke

```bash
source env.sh
python tests/test_w4a4.py -v
```

The CPU test verifies the preshuffled payload and scale layout. On ROCm, the GPU
tests also compare original versus padded weight rows, compare row-major versus
preshuffled gate weights, and reject a row-major tensor passed to the tiled op.

## LIBERO eval (accuracy)

```bash
source env.sh
export MUJOCO_GL=egl

python scripts/libero_eval.py \
  --policy-id ~/model_data/pi05_snapflow_1nfe \
  --tasks libero_10 --episodes 2 --seed 0 \
  --n-inference-steps 1 --n-action-steps 10 \
  --output-dir eval_out/libero10_ep2

# optional prefix trimming; evaluate separately for the target task set
PI05_LANG_TOKENS=192 python scripts/libero_eval.py ... --output-dir eval_out/libero10_lang192
PI05_LANG_TOKENS=64 python scripts/libero_eval.py ... --output-dir eval_out/libero10_lang64
```

Videos: `eval_out/libero10_ep2/videos/libero_10_{id}/eval_episode_{0,1}.mp4`

`--resume` evaluates one task at a time (survives crashes; recompiles per task).

When using `PI05_LANG_TOKENS`, check the log says `lang tokens trimmed to 64` and does
**not** warn `instruction needs N tokens`: on that warning the prompt did not fit, the full
200 tokens were used, and the run measured the default path rather than the trimmed one.

## Latency (synthetic, not sim)

```bash
source env.sh

python scripts/bench_latency.py                        # ~95 ms
PI05_LANG_TOKENS=192 python scripts/bench_latency.py   # 91.50 ms
PI05_LANG_TOKENS=64 python scripts/bench_latency.py    # 81.99 ms
PI05_LANG_TOKENS=64 PI05_CACHE_EMPTY_CAM=1 \
    python scripts/bench_latency.py                    # 74.48 ms
```

First call pays `torch.compile` autotune (minutes). Median after warmup is the number
to compare (**95.03 ms** on 8060S, **74.48 ms** with both knobs).

The synthetic prompt fills 52 of the 200 padded language tokens, matching LIBERO-10
(`PI05_BENCH_LANG_LEN` overrides it).

`download_checkpoints.sh` rewrites SnapFlow `config.json` once (drops `_reflex_*`
keys that stock LeRobot rejects). Later eval/bench runs are a no-op if already clean.

## Layout

```
pi05-pytorch-fast/
├── README.md
├── RESULTS.md
├── constraints-rocm.txt
├── requirements-lerobot-runtime.txt
├── env.sh                      # generated by setup.sh
├── pi05_fast/                  # runtime hooks (no ONNX)
│   ├── rocm_pi05_optim.py
│   ├── rocm_antialias.py
│   ├── snapflow_pi05.py
│   └── w4a4/               
├── scripts/
    ├── setup.sh
    ├── download_checkpoints.sh
    ├── pack_w4a4_prefix.py
    ├── libero_eval.py
    └── bench_latency.py
└── tests/
    └── test_w4a4.py
```

## Environment knobs

| Env | Default | Meaning |
|-----|---------|---------|
| `PI05_W4A4_PACK` | `$ROOT/models/w4a4_prefix` | Packed INT4 LM weights |
| `PI05_COMPILE` | `1` | `torch.compile` |
| `PI05_COMPILE_MODE` | `max-autotune-no-cudagraphs` | Inductor mode |
| `PI05_PREFIX_LM_FP16` | `1` | Cast prefix LM to fp16 after W4A4 |
| `PI05_W4A4_FUSE_QUANT` | `1` | Shared QKV / gate-up quant |
| `PI05_W4A4_ROW_PAD_WORDS` | `8` | Runtime-pad each packed weight row to a 32-byte-aligned, off-period stride. Set `1` for the original `K/8+1` layout |
| `PI05_W4A4_GATE_PRESHUFFLE` | `1` | Reorder the static 2048→16384 gate/up weights once at load time into WMMA-fragment-major order. Set `0` for the row-major fallback |
| `PYTORCH_HIP_ALLOC_CONF` | `max_split_size_mb:512` | Stable allocator setting on this gfx1151 APU. `expandable_segments:True` can hang the first allocation with the pinned nightly stack |
| `PI05_INDUCTOR_GEMM_BACKENDS` | `ATEN` | Inductor GEMM backend. ROCm 10.1 disables the origami heuristic, so Inductor mis-ranks triton over hipBLASLt; `ATEN` is ~2% faster. Set empty to restore triton autotune |
| `PI05_LANG_TOKENS` | unset (off) | Trim trailing **padded** language tokens to this bucket (prefix = 576 + N). Real tokens are never dropped: if the prompt does not fit, the full 200 are used and a warning is printed. This changes the compiled sequence shape and is not action-equivalent, so trimmed modes remain opt-in. |
| `PI05_CACHE_EMPTY_CAM` | `0` (off) | Cache the empty-camera SigLIP embedding. With `empty_cameras=1` the placeholder camera is a constant `-1` image, so its embedding never changes, yet a full vision forward runs on it every frame. Caching it batches only the real cameras (−7 ms). Not bit-exact: bf16 SigLIP output depends on batch size (rel err ~4e-3) |
| `SNAPFLOW_TEACHER_POLICY` | `~/model_data/pi05_libero_v044` | Processor json/safetensors |
| `PALIGEMMA_TOKENIZER_PATH` | `~/model_data/paligemma2-3b-pt-224` | Local tokenizer |

## What we tried and did **not** keep

- SigLIP fp16 (slower than bf16 on this GPU)
- SigLIP / LM / expert FlashAttention-2 (no win; FA2 + 4D mask crashes)
- LM SDPA (slower than eager with additive 4D masks)
- Quantizing SigLIP 4-bit (MLP `K=4304` not multiple of 64; attn projections cost ~1 LIBERO
  episode for only −1.7% latency)
- W4A4 on the action expert (at `M=50` the INT4 path is *slower* than bf16 hipBLASLt)
- Fusing GELU × gate/up into down-projection quantization (slower and not bit-exact)
- XOR-swizzled LDS and later vector/tail specializations whose sub-millisecond gains did
  not justify the extra shape-specific production code
- Making `PI05_LANG_TOKENS=192` the default: it measured 91.50 ms, but two
  repeated seed-0 LIBERO-10 smoke runs were 19/20 and 18/20 versus 20/20 on the
  maintained 200-token path. The failed episodes changed between runs, so this
  is a robustness warning rather than evidence of one deterministic task failure
- ROCm expandable allocator segments with the pinned nightly stack (first allocation can hang)
- ONNX / MIGraphX / ORT (parent `pi05-migraphx` package)

## License / provenance

- W4A4 kernel layout (Hadamard on selected K).
- WMMA epilogue scale caching/full-tile pattern adapted from
  [Comfy-Kitchen PR #174](https://github.com/Comfy-Org/comfy-kitchen/pull/174).
- SnapFlow eval class adapted from Reflex VLA (Apache-2.0).
- Antialias pass from AMD rocm-scripts gfx1151 tests.
- Policy weights: Hugging Face (see `download_checkpoints.sh`).
