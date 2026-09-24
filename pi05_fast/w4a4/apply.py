from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from pi05_fast.w4a4.linear import W4A4Linear
from pi05_fast.w4a4.pack import preshuffle_wmma_weight, should_rotate

# Prefix language-model GEMMs only (SigLIP / expert / embeddings stay fp).
DEFAULT_NAME_PREFIXES = (
    "paligemma_with_expert.paligemma.model.language_model.layers.",
)


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent = root
    *parts, leaf = name.split(".")
    for p in parts:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def iter_packable_linears(
    model: nn.Module,
    name_prefixes: Iterable[str] = DEFAULT_NAME_PREFIXES,
) -> list[tuple[str, nn.Linear]]:
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name.startswith(p) for p in name_prefixes):
            continue
        if mod.in_features % 64 != 0:
            continue
        out.append((name, mod))
    return out


def replace_linears_from_packed(
    model: nn.Module,
    packed: dict[str, torch.Tensor],
    meta_layers: dict[str, dict],
    device: torch.device | None = None,
) -> int:
    row_pad_words = max(1, int(os.environ.get("PI05_W4A4_ROW_PAD_WORDS", "8")))
    gate_preshuffle = os.environ.get("PI05_W4A4_GATE_PRESHUFFLE", "1") != "0"
    n = 0
    for name, spec in meta_layers.items():
        key = f"{name}.packed"
        if key not in packed:
            continue
        w = packed[key]
        logical_words = int(spec["in_features"]) // 8
        preshuffled = (
            gate_preshuffle
            and name.endswith((".mlp.gate_proj", ".mlp.up_proj"))
            and int(spec["in_features"]) == 2048
            and int(spec["out_features"]) == 16384
        )
        if preshuffled:
            w = preshuffle_wmma_weight(w, int(spec["in_features"]))
        elif row_pad_words > 1 and w.shape[1] == logical_words + 1:
            w = torch.nn.functional.pad(w, (0, row_pad_words - 1))
        if device is not None:
            w = w.to(device)
        bias = packed.get(f"{name}.bias")
        if bias is not None and device is not None:
            bias = bias.to(device)
        rotate = spec.get("rotate")
        _set_module(
            model,
            name,
            W4A4Linear(
                w,
                bias=bias,
                rotate=rotate,
                in_features=int(spec["in_features"]),
                preshuffled=preshuffled,
            ),
        )
        n += 1
    return n


def apply_w4a4_from_dir(model: nn.Module, pack_dir: str | Path) -> int:
    pack_dir = Path(pack_dir).expanduser()
    meta_path = pack_dir / "meta.json"
    st_path = pack_dir / "packed.safetensors"
    if not meta_path.is_file() or not st_path.is_file():
        raise FileNotFoundError(f"need {meta_path} and {st_path}")
    meta = json.loads(meta_path.read_text())
    from safetensors.torch import load_file

    packed = load_file(str(st_path))
    device = next(model.parameters()).device
    from pi05_fast.w4a4.extension import preload

    preload()
    n = replace_linears_from_packed(model, packed, meta["layers"], device=device)
    n_fuse = fuse_shared_activation_quant(model)
    n_preshuffled = sum(
        isinstance(module, W4A4Linear) and module.preshuffled for module in model.modules()
    )
    row_pad_words = max(1, int(os.environ.get("PI05_W4A4_ROW_PAD_WORDS", "8")))
    print(
        f"w4a4: replaced {n} Linear modules from {pack_dir}; "
        f"shared-quant fused {n_fuse} QKV/MLP groups; "
        f"row stride pad {row_pad_words} word(s); "
        f"WMMA-preshuffled gate/up {n_preshuffled}",
        flush=True,
    )
    return n


def _as_fp16_contig(x: torch.Tensor) -> torch.Tensor:
    xf = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
    return xf if xf.is_contiguous() else xf.contiguous()


def _fused_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    from pi05_fast.w4a4.extension import (
        int4_gemm,
        int4_gemm2,
        int4_gemm2_tiled,
    )

    xf = _as_fp16_contig(x)
    if self.gate_proj.preshuffled and self.up_proj.preshuffled:
        gate, up = int4_gemm2_tiled(xf, self.gate_proj.packed, self.up_proj.packed)
    else:
        gate, up = int4_gemm2(xf, self.gate_proj.packed, self.up_proj.packed)
    h = self.act_fn(gate) * up
    if not h.is_contiguous():
        h = h.contiguous()
    y = int4_gemm(h, self.down_proj.packed)
    return y if y.dtype == x.dtype else y.to(dtype=x.dtype)


def _fused_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs,
):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, eager_attention_forward

    from pi05_fast.w4a4.extension import int4_gemm3

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q, k, v = int4_gemm3(
        _as_fp16_contig(hidden_states),
        self.q_proj.packed,
        self.k_proj.packed,
        self.v_proj.packed,
    )
    query_states = q.view(hidden_shape).transpose(1, 2)
    key_states = k.view(hidden_shape).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def fuse_shared_activation_quant(model: nn.Module) -> int:
    """One activation quant for QKV and for gate/up (same x, several GEMMs)."""
    if os.environ.get("PI05_W4A4_FUSE_QUANT", "1") == "0":
        return 0
    n = 0
    for mod in model.modules():
        qkv = (getattr(mod, "q_proj", None), getattr(mod, "k_proj", None), getattr(mod, "v_proj", None))
        if all(isinstance(p, W4A4Linear) for p in qkv):
            mod.forward = types.MethodType(_fused_attn_forward, mod)
            n += 1
            continue
        gate_up = (
            getattr(mod, "gate_proj", None),
            getattr(mod, "up_proj", None),
            getattr(mod, "down_proj", None),
        )
        if all(isinstance(p, W4A4Linear) for p in gate_up) and hasattr(mod, "act_fn"):
            mod.forward = types.MethodType(_fused_mlp_forward, mod)
            n += 1
    return n


def maybe_apply_w4a4_from_env(model: nn.Module) -> int:
    path = os.environ.get("PI05_W4A4_PACK", "").strip()
    if not path:
        return 0
    return apply_w4a4_from_dir(model, path)


def pack_model_linears(
    model: nn.Module,
    name_prefixes: Iterable[str] = DEFAULT_NAME_PREFIXES,
) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    from pi05_fast.w4a4.pack import pack_linear_weight

    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict] = {}
    for name, lin in iter_packable_linears(model, name_prefixes):
        rotate = should_rotate(lin.in_features)
        packed = pack_linear_weight(lin.weight, rotate=rotate)
        tensors[f"{name}.packed"] = packed.contiguous()
        if lin.bias is not None:
            tensors[f"{name}.bias"] = lin.bias.detach().cpu().contiguous()
        layers[name] = {
            "in_features": lin.in_features,
            "out_features": lin.out_features,
            "rotate": rotate,
            "has_bias": lin.bias is not None,
        }
    return tensors, layers
