from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from pi05_fast.w4a4.extension import int4_gemm, int4_gemm_tiled
from pi05_fast.w4a4.pack import preshuffle_wmma_weight


def _packed_weight(n: int, k: int, device: torch.device) -> torch.Tensor:
    payload = torch.randint(
        -(2**31),
        2**31 - 1,
        (n, k // 8),
        dtype=torch.int32,
    )
    scales = torch.rand(n, dtype=torch.float32) * 0.05 + 0.01
    packed = torch.cat((payload, scales.view(torch.int32).unsqueeze(1)), dim=1)
    return packed.to(device)


class PreshuffleLayoutTest(unittest.TestCase):
    def test_payload_and_scales_round_trip(self) -> None:
        n, k = 7, 64
        packed = _packed_weight(n, k, torch.device("cpu"))
        tiled = preshuffle_wmma_weight(packed, k)
        words = k // 8

        flat = tiled.view(-1)
        restored_payload = (
            flat[: n * words]
            .view(words // 2, n, 2)
            .permute(1, 0, 2)
            .reshape(n, words)
        )
        restored_scales = flat[n * words : n * words + n]

        self.assertEqual(tuple(tiled.shape), (n, words + 9))
        self.assertTrue(torch.equal(restored_payload, packed[:, :words]))
        self.assertTrue(torch.equal(restored_scales, packed[:, words]))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None,
    "requires a ROCm GPU",
)
class W4A4GpuCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(20260923)
        cls.device = torch.device("cuda")

    def test_row_padding_is_exact_for_down_projection(self) -> None:
        m, k, n = 3, 16384, 128
        x = torch.randn(m, k, dtype=torch.float16, device=self.device)
        packed = _packed_weight(n, k, self.device)
        padded = F.pad(packed, (0, 7))

        reference = int4_gemm(x, packed)
        candidate = int4_gemm(x, padded)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(reference, candidate))

    def test_preshuffled_gate_weight_is_exact(self) -> None:
        # Covers one 256-row main tile plus the specialized 16-row tail.
        m, k, n = 272, 2048, 16384
        x = torch.randn(m, k, dtype=torch.float16, device=self.device)
        packed = _packed_weight(n, k, self.device)
        tiled = preshuffle_wmma_weight(packed, k)

        reference = int4_gemm(x, packed)
        candidate = int4_gemm_tiled(x, tiled)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(reference, candidate))

    def test_tiled_op_rejects_row_major_weight(self) -> None:
        x = torch.randn(1, 2048, dtype=torch.float16, device=self.device)
        packed = _packed_weight(16384, 2048, self.device)

        with self.assertRaisesRegex(RuntimeError, "K/8\\+9 marker stride"):
            int4_gemm_tiled(x, packed)


if __name__ == "__main__":
    unittest.main()
