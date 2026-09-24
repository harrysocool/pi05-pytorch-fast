from __future__ import annotations

import torch
import torch.nn as nn

from pi05_fast.w4a4.extension import int4_gemm
from pi05_fast.w4a4.pack import pack_linear_weight, should_rotate


class W4A4Linear(nn.Module):
    """Drop-in Linear: runtime INT4 activations + packed INT4 weights."""

    def __init__(
        self,
        packed: torch.Tensor,
        bias: torch.Tensor | None = None,
        rotate: bool | None = None,
        in_features: int | None = None,
    ):
        super().__init__()
        if packed.dim() != 2:
            raise ValueError("packed weight must be [N, K/8+1]")
        self.out_features = packed.shape[0]
        self.in_features = in_features or (packed.shape[1] - 1) * 8
        if packed.shape[1] < self.in_features // 8 + 1:
            raise ValueError("packed weight row is too short for in_features")
        self.rotate = should_rotate(self.in_features, rotate)
        self.register_buffer("packed", packed.to(torch.int32), persistent=True)
        # LeRobot reads ``proj.weight.dtype``; keep a dtype marker, not real weights.
        self.register_buffer(
            "weight",
            torch.empty(0, dtype=torch.float16, device=packed.device),
            persistent=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rotate: bool | None = None) -> W4A4Linear:
        packed = pack_linear_weight(linear.weight, rotate=rotate)
        bias = linear.bias.detach() if linear.bias is not None else None
        return cls(packed, bias=bias, rotate=rotate, in_features=linear.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
        if not xf.is_contiguous():
            xf = xf.contiguous()
        y = int4_gemm(xf, self.packed)
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rotate={self.rotate}"
