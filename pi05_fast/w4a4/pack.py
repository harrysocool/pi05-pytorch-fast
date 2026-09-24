"""Offline weight pack matching Embedl int4_gemm (Hadamard + INT4 [-7,7]).

PyTorch Linear is Y = X @ W.T, W.shape = [N, K] (out, in). Packed layout is
[N, K/8 + 1] int32; the last entry of each row is the per-channel scale bitcast.
"""

from __future__ import annotations

import math
from typing import Any

import torch

ROTATE_K = frozenset({1024, 2048, 16384})
WMMA_PRESHUFFLE_PAD_WORDS = 9


def fwht_unnormalized(x: torch.Tensor) -> torch.Tensor:
    """In-place-style unnormalized Walsh–Hadamard (a+b, a-b) along the last dim."""
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT length must be a power of 2, got {n}")
    out = x.contiguous().clone()
    h = 1
    while h < n:
        out = out.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return out


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """q[..., K] int32 in [-7, 7] → packed[..., K/8] int32, 8 nibbles per word."""
    if q.shape[-1] % 8 != 0:
        raise ValueError(f"K must be a multiple of 8, got {q.shape[-1]}")
    q = q.to(torch.int32).view(*q.shape[:-1], q.shape[-1] // 8, 8)
    packed = torch.zeros(*q.shape[:-1], dtype=torch.int32, device=q.device)
    for j in range(8):
        packed = packed | ((q[..., j] & 0xF) << (j * 4))
    return packed


def preshuffle_wmma_weight(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Reorder packed weights as [K/16, N, 2 words] for coalesced WMMA loads.

    The per-channel scales follow the packed payload in one contiguous block.
    An otherwise-unused ninth padding word distinguishes this internal layout
    from the ordinary row-major ``K/8 + 8`` representation.
    """
    if packed.dim() != 2:
        raise ValueError(f"expected 2D packed weight, got {tuple(packed.shape)}")
    n = packed.shape[0]
    words = k // 8
    if k % 16 != 0 or packed.shape[1] < words + 1:
        raise ValueError(f"invalid packed weight shape {tuple(packed.shape)} for K={k}")

    out = torch.zeros(
        (n, words + WMMA_PRESHUFFLE_PAD_WORDS),
        dtype=packed.dtype,
        device=packed.device,
    )
    payload = packed[:, :words].view(n, words // 2, 2).permute(1, 0, 2).contiguous()
    flat = out.view(-1)
    flat[: n * words].copy_(payload.view(-1))
    flat[n * words : n * words + n].copy_(packed[:, words])
    return out


def unpack_nibbles(packed: torch.Tensor, k: int) -> torch.Tensor:
    q = torch.empty(*packed.shape, 8, dtype=torch.int32, device=packed.device)
    for j in range(8):
        nibble = (packed >> (j * 4)) & 0xF
        q[..., j] = torch.where(nibble >= 8, nibble - 16, nibble)
    return q.reshape(*packed.shape[:-1], k)


def should_rotate(k: int, rotate: bool | None = None) -> bool:
    if rotate is not None:
        return bool(rotate)
    return k in ROTATE_K


def pack_linear_weight(weight: torch.Tensor, rotate: bool | None = None) -> torch.Tensor:
    """Pack Linear.weight [N, K] → int32 [N, K/8+1]."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % 64 != 0:
        raise ValueError(f"K must be a multiple of 64 for gemm_iu4, got {k}")
    w = weight.detach().float().cpu()
    do_rot = should_rotate(k, rotate)
    if do_rot:
        w = fwht_unnormalized(w) * (1.0 / math.sqrt(k))
    amax = w.abs().amax(dim=-1).clamp_min(1e-12)
    scale = amax / 7.0
    q = (w / scale.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32)
    packed = pack_nibbles(q)
    scale_bits = scale.contiguous().view(torch.int32).unsqueeze(-1)
    return torch.cat([packed, scale_bits], dim=-1)


def unpack_linear_weight(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (q [N,K], scale [N], K)."""
    n, row = packed.shape
    k = (row - 1) * 8
    q = unpack_nibbles(packed[:, :-1], k)
    scale = packed[:, -1].contiguous().view(torch.float32)
    return q, scale, k


def quant_activation_cpu(x: torch.Tensor, rotate: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Match quant_rot_rb / quant_rows. x [M, K] → (q [M,K], scale [M])."""
    if x.dim() != 2:
        raise ValueError("quant_activation_cpu expects [M, K]")
    m, k = x.shape
    xf = x.detach().float()
    do_rot = should_rotate(k, rotate)
    if do_rot:
        xf = fwht_unnormalized(xf)
        amax = xf.abs().amax(dim=-1).clamp_min(0.0)
        sdiv = amax / 7.0 + 1e-12
        scale = sdiv * (1.0 / math.sqrt(k))
        q = (xf / sdiv.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32)
    else:
        amax = xf.abs().amax(dim=-1).clamp_min(0.0)
        sdiv = amax / 7.0 + 1e-12
        scale = sdiv
        q = (xf / sdiv.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32)
    return q, scale


def int4_gemm_ref(x: torch.Tensor, packed_w: torch.Tensor, rotate: bool | None = None) -> torch.Tensor:
    """CPU reference for int4_gemm (fp32 accum)."""
    orig = x.shape
    k = orig[-1]
    x2 = x.reshape(-1, k)
    q_a, s_a = quant_activation_cpu(x2, rotate=rotate)
    q_w, s_w, k_w = unpack_linear_weight(packed_w)
    if k_w != k:
        raise ValueError(f"K mismatch act={k} weight={k_w}")
    acc = q_a.float() @ q_w.float().T
    y = acc * s_a.unsqueeze(-1) * s_w.unsqueeze(0)
    return y.reshape(*orig[:-1], packed_w.shape[0]).to(torch.float16)


def layer_meta(name: str, packed: torch.Tensor, rotate: bool, has_bias: bool) -> dict[str, Any]:
    n, row = packed.shape
    k = (row - 1) * 8
    return {
        "name": name,
        "in_features": k,
        "out_features": n,
        "rotate": rotate,
        "has_bias": has_bias,
    }
