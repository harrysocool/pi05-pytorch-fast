"""ROCm / gfx1151 PyTorch optimizations from rocm-scripts pi0.5 tests.

Source: rocm-scripts/test/pytorch/lerobot_pi05.py and _openpi_setup.py

  - HIP allocator + hipblaslt + AOTriton experimental flags (set before torch)
  - bfloat16 weights
  - batched SigLIP embed_prefix (one forward for all cameras)
  - eager attention (Inductor fuses better than SDPA on short seqs)
  - tf32/high matmul precision
  - gfx1151 L2 anti-alias stride pads (before torch.compile)
  - torch.compile(sample_actions, max-autotune-no-cudagraphs, fullgraph=True)
"""

from __future__ import annotations

import os
import types
from typing import Any

_FROM_PRETRAINED_PATCHED = False


def apply_rocm_env() -> None:
    """Call before importing torch."""
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")


def clone_kv_cache(past_key_values, dtype=None):
    """Tensor clone of DynamicCache (Dynamo-traceable; no copy.deepcopy)."""
    from transformers.cache_utils import DynamicCache

    if hasattr(past_key_values, "layers"):
        cloned = DynamicCache()
        for layer in past_key_values.layers:
            k, v = layer.keys.clone(), layer.values.clone()
            if dtype is not None:
                k, v = k.to(dtype), v.to(dtype)
            cloned.update(k, v, len(cloned.layers))
        return cloned
    from lerobot.policies.pi05.modeling_pi05 import clone_past_key_values

    return clone_past_key_values(past_key_values)


# embedl prefix_encoder.mxr: cam0/cam1 full SigLIP (256) + cam2 pooled (64) + lang (200) = 776
EMBEDL_CAM2_TOKENS = 64
EMBEDL_CAM2_POOL = 4  # 256 / 64


def pool_cam2_image_tokens(img_emb):
    """256 vision tokens → 64 (4:1 sum-pool), matching embedl .mxr / ONNX layout."""
    bsize, num_tokens, dim = img_emb.shape
    if num_tokens != 256:
        raise ValueError(f"expected 256 SigLIP tokens before cam2 pool, got {num_tokens}")
    grouped = img_emb.reshape(bsize, EMBEDL_CAM2_TOKENS, EMBEDL_CAM2_POOL, dim)
    return grouped.sum(dim=2)


def _use_embedl_empty_cam_pool(model) -> bool:
    if os.environ.get("PI05_PREFIX_LAYOUT", "embedl").lower() in {"openpi", "full", "968"}:
        return False
    return int(getattr(model.config, "empty_cameras", 0) or 0) > 0


def install_empty_cam_cache(model) -> bool:
    """Precompute the empty-camera SigLIP embedding (constant -1 image, so it never changes)."""
    import torch

    if os.environ.get("PI05_CACHE_EMPTY_CAM", "0") == "0":
        return False
    n_empty = int(getattr(model.config, "empty_cameras", 0) or 0)
    if n_empty <= 0:
        return False
    res = getattr(model.config, "image_resolution", (224, 224))
    ref = next(model.paligemma_with_expert.paligemma.parameters())
    blank = torch.full((1, 3, int(res[0]), int(res[1])), -1.0, dtype=ref.dtype, device=ref.device)
    with torch.no_grad():
        emb = model.paligemma_with_expert.embed_image(blank)
    model._empty_cam_emb = emb
    model._empty_cam_emb_pooled = pool_cam2_image_tokens(emb)
    model._n_empty_cams = n_empty
    print(f"rocm-opt: empty-camera SigLIP cached ({n_empty} slot(s); skips a full vision forward)", flush=True)
    return True


def _batched_embed_prefix(self, images, img_masks, tokens, masks):
    import torch

    embs, pad_masks, att_masks = [], [], []
    num_images = len(images)
    bsize = images[0].shape[0]
    pool_last = _use_embedl_empty_cam_pool(self) and num_images >= 1

    def image_embed_func(img):
        return self.paligemma_with_expert.embed_image(img)

    # Trailing empty cameras are a constant -1 image; reuse their cached embedding.
    n_cached = getattr(self, "_n_empty_cams", 0) if hasattr(self, "_empty_cam_emb") else 0
    n_real = num_images - n_cached if n_cached and num_images > n_cached else num_images
    batched_images = torch.cat(images[:n_real], dim=0)
    batched_emb = self._apply_checkpoint(image_embed_func, batched_images)
    img_embs = batched_emb.reshape(n_real, bsize, batched_emb.shape[1], batched_emb.shape[2])
    for i in range(num_images):
        if i < n_real:
            cam = img_embs[i]
            if pool_last and i == num_images - 1:
                cam = pool_cam2_image_tokens(cam)
        else:
            cached = (
                self._empty_cam_emb_pooled
                if (pool_last and i == num_images - 1)
                else self._empty_cam_emb
            )
            cam = cached.expand(bsize, -1, -1)
        num_img_embs = cam.shape[1]
        embs.append(cam)
        pad_masks.append(img_masks[i][:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    lang_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, tokens)
    embs.append(lang_emb)
    pad_masks.append(masks)
    att_masks += [0] * lang_emb.shape[1]
    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
    att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
    lm_dtype = getattr(self, "_prefix_lm_dtype", None)
    if lm_dtype is not None:
        embs = embs.to(lm_dtype)
    return embs, pad_masks, att_masks


def _cast_floating_tensors(module, dtype) -> int:
    """Cast float params/buffers; leave INT4 packed weights untouched."""
    n = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.is_floating_point() and tensor.dtype != dtype:
            tensor.data = tensor.data.to(dtype)
            n += 1
    return n


def enable_prefix_lm_fp16(model) -> None:
    """Run PaliGemma prefix LM in fp16 so W4A4 GEMMs match Embedl (no bf16 sandwich)."""
    import torch

    pal = model.paligemma_with_expert.paligemma
    lm = pal.model.language_model if hasattr(pal, "model") else pal.language_model
    n = _cast_floating_tensors(lm, torch.float16)
    model._prefix_lm_dtype = torch.float16
    print(f"rocm-opt: prefix LM fp16 ({n} tensors; SigLIP/expert stay bf16)", flush=True)


def _lang_token_budget() -> int:
    raw = os.environ.get("PI05_LANG_TOKENS", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def install_lang_token_trim(model) -> bool:
    """Drop trailing padded language tokens to a fixed bucket before the compiled call.

    Shrinks prefill M (776 -> 576 + budget). Static bucket keeps one compiled shape.
    Falls back to the full length if the instruction does not fit the bucket.
    """
    budget = _lang_token_budget()
    if budget <= 0:
        return False
    orig = model.sample_actions
    state = {"warned": False}

    def _trimmed(images, img_masks, tokens, masks, *args, **kwargs):
        n = tokens.shape[1]
        if n > budget:
            if bool(masks[:, budget:].any()):
                if not state["warned"]:
                    state["warned"] = True
                    need = int(masks.sum(dim=1).max().item())
                    print(
                        f"rocm-opt: instruction needs {need} of {n} tokens > "
                        f"PI05_LANG_TOKENS={budget}; using full {n}",
                        flush=True,
                    )
            else:
                tokens = tokens[:, :budget].contiguous()
                masks = masks[:, :budget].contiguous()
        return orig(images, img_masks, tokens, masks, *args, **kwargs)

    model.sample_actions = _trimmed
    print(
        f"rocm-opt: lang tokens trimmed to {budget} (prefix {576 + budget} vs 776)",
        flush=True,
    )
    return True


def apply_rocm_pi05_optimizations(policy: Any, *, compile_model: bool | None = None) -> Any:
    """Mutate a loaded PI05Policy for gfx1151 PyTorch inference."""
    import torch

    model = policy.model
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if torch.cuda.is_available():
        policy.to("cuda")
        model.to(torch.bfloat16)
        if hasattr(policy.config, "device"):
            policy.config.device = "cuda"

    model.embed_prefix = types.MethodType(_batched_embed_prefix, model)
    if _use_embedl_empty_cam_pool(model):
        print(
            "rocm-opt: embedl empty_camera pool 256→64 (prefix 256+256+64+200=776)",
            flush=True,
        )

    pal = model.paligemma_with_expert.paligemma
    lm = pal.model.language_model if hasattr(pal, "model") else pal.language_model
    lm.config._attn_implementation = "eager"  # noqa: SLF001
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

    torch.set_float32_matmul_precision("high")

    from pi05_fast.w4a4.apply import maybe_apply_w4a4_from_env

    n_w4a4 = maybe_apply_w4a4_from_env(model)
    if n_w4a4 and os.environ.get("PI05_PREFIX_LM_FP16", "1") != "0":
        enable_prefix_lm_fp16(model)

    if compile_model is None:
        compile_model = os.environ.get("PI05_COMPILE", "1") != "0"
    if compile_model and torch.cuda.is_available():
        from pi05_fast.rocm_antialias import enable_antialias

        enable_antialias()
        # ROCm 10.1 disables Inductor's origami GEMM heuristic, so its autotuner
        # mis-ranks triton over hipBLASLt for the bf16 SigLIP/expert GEMMs.
        # Forcing ATEN routes them to hipBLASLt (~2% E2E, accuracy-neutral).
        gemm_backends = os.environ.get("PI05_INDUCTOR_GEMM_BACKENDS", "ATEN").strip()
        if gemm_backends:
            import torch._inductor.config as _ic

            _ic.max_autotune_gemm_backends = gemm_backends
            print(f"rocm-opt: inductor GEMM backends = {gemm_backends}", flush=True)
        mode = os.environ.get("PI05_COMPILE_MODE", "max-autotune-no-cudagraphs")
        w4_note = " + W4A4 custom_op" if n_w4a4 else ""
        try:
            fullgraph = os.environ.get("PI05_COMPILE_FULLGRAPH", "1") != "0"
            if hasattr(model, "sample_actions_1step"):
                model.sample_actions_1step = torch.compile(
                    model.sample_actions_1step, mode=mode, fullgraph=fullgraph
                )

                def _sample_actions(
                    self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs
                ):
                    return self.sample_actions_1step(images, img_masks, tokens, masks, noise=noise)

                model.sample_actions = types.MethodType(_sample_actions, model)
            else:
                model.sample_actions = torch.compile(
                    model.sample_actions, mode=mode, fullgraph=fullgraph
                )
            print(
                f"rocm-opt: torch.compile({mode}, fullgraph={fullgraph}) + antialias "
                f"+ prefix-LM-fp16 + batched SigLIP{w4_note}",
                flush=True,
            )
            compile_model = True
        except Exception as e:
            if n_w4a4 and os.environ.get("PI05_COMPILE_FULLGRAPH", "1") != "0":
                print(f"rocm-opt: fullgraph compile failed ({e!r}); retrying fullgraph=False", flush=True)
                try:
                    if hasattr(model, "sample_actions_1step"):
                        model.sample_actions_1step = torch.compile(
                            model.sample_actions_1step, mode=mode, fullgraph=False
                        )

                        def _sample_actions(
                            self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs
                        ):
                            return self.sample_actions_1step(
                                images, img_masks, tokens, masks, noise=noise
                            )

                        model.sample_actions = types.MethodType(_sample_actions, model)
                    else:
                        model.sample_actions = torch.compile(
                            model.sample_actions, mode=mode, fullgraph=False
                        )
                    print(
                        f"rocm-opt: torch.compile({mode}, fullgraph=False) + antialias + bf16 "
                        f"+ batched SigLIP{w4_note}",
                        flush=True,
                    )
                    compile_model = True
                except Exception as e2:
                    print(f"rocm-opt: torch.compile failed ({e2!r}); using eager bf16", flush=True)
                    compile_model = False
            else:
                print(f"rocm-opt: torch.compile failed ({e!r}); using eager bf16", flush=True)
                compile_model = False
    elif not compile_model:
        print("rocm-opt: bf16 + batched SigLIP + eager attn (compile off)", flush=True)

    install_empty_cam_cache(model)
    install_lang_token_trim(model)

    if torch.cuda.is_available():
        _orig_pred = policy.predict_action_chunk
        prefix_fp16 = getattr(model, "_prefix_lm_dtype", None) is not None

        if prefix_fp16:
            # Global bf16 autocast would promote LM activations back to bf16
            # around each W4A4Linear. Dtypes are already set per submodule.
            def _pred_no_autocast(batch, **kwargs):
                with torch.no_grad():
                    return _orig_pred(batch, **kwargs)

            policy.predict_action_chunk = _pred_no_autocast
        else:

            def _pred_autocast(batch, **kwargs):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return _orig_pred(batch, **kwargs)

            policy.predict_action_chunk = _pred_autocast
    return policy


def install_rocm_pi05_eval_hooks() -> None:
    """Apply optimizations after PI05Policy.from_pretrained (weights already loaded)."""
    global _FROM_PRETRAINED_PATCHED
    if _FROM_PRETRAINED_PATCHED:
        return
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    orig = PI05Policy.from_pretrained.__func__

    @classmethod
    def wrapped(
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        force_download=False,
        resume_download=None,
        proxies=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
        revision=None,
        strict=True,
        **kwargs,
    ):
        # PI05Policy.__init__ does model.to(config.device) with device=None → cuda.
        # from_pretrained then load_file's a host state dict and load_state_dict
        # copies it onto those params. On gfx1151 UMA that is two ~16 GB copies
        # in the same DRAM; copy_ runs GPU-idle for many minutes (or swap-thrashes).
        # Build on CPU; apply_rocm_pi05_optimizations then .to("cuda") + bf16.
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig

            # Match LeRobot's original config construction, including policy
            # overrides supplied through **kwargs. Only the device is changed.
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        prev = getattr(config, "device", None)
        if hasattr(config, "device") and prev != "cpu":
            config.device = "cpu"
            print(
                f"rocm-opt: load on CPU (was {prev!r}) then move to iGPU "
                "(Strix Halo UMA)",
                flush=True,
            )
        policy = orig(
            cls,
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
        if hasattr(config, "device"):
            config.device = prev
        return apply_rocm_pi05_optimizations(policy)

    PI05Policy.from_pretrained = wrapped
    _FROM_PRETRAINED_PATCHED = True
    print("rocm-opt: PI05Policy.from_pretrained hooked", flush=True)
