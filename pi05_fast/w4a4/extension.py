"""HIP int4_gemm + torch.library custom op (Inductor / torch.compile)."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import torch

_mod = None
_op_registered = False


def load_int4_gemm():
    global _mod
    if _mod is not None:
        return _mod

    import _rocm_sdk_core

    sdk_root = Path(_rocm_sdk_core.__file__).resolve().parent
    if "ROCM_PATH" in os.environ:
        rocm_root = Path(os.environ["ROCM_PATH"])
    else:
        # Prefer the rocm-sdk-devel tree shipped inside the venv over any host
        # install (full ROCm root: llvm, amdgcn bitcode, thrust/rocprim headers).
        try:
            import _rocm_sdk_devel

            rocm_root = Path(_rocm_sdk_devel.__file__).resolve().parent
        except ImportError:
            rocm_root = sdk_root
    if not (rocm_root / "lib" / "llvm" / "bin" / "clang++").is_file():
        rocm_root = Path("/opt/rocm")

    os.environ.setdefault("PYTORCH_ROCM_ARCH", os.environ.get("GPU_ARCHS", "gfx1151"))
    os.environ["ROCM_PATH"] = str(rocm_root)
    os.environ["HIP_PATH"] = str(rocm_root)

    # Imported here: cpp_extension resolves ROCM_HOME once at module import time.
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).resolve().parent / "csrc"

    rocm_lib = sdk_root / "lib"
    link_dir = Path("/tmp/pi05_fast_w4a4_rocm_lib")
    link_dir.mkdir(parents=True, exist_ok=True)
    soname = rocm_lib / "libamdhip64.so.7"
    link = link_dir / "libamdhip64.so"
    if soname.is_file() and not link.exists():
        link.symlink_to(soname)
    cuda_cflags = ["-O3", "-std=c++20", f"--rocm-path={rocm_root}"]
    for device_libs in (
        rocm_root / "amdgcn" / "bitcode",
        rocm_root / "lib" / "llvm" / "amdgcn" / "bitcode",
    ):
        if device_libs.is_dir():
            cuda_cflags.append(f"--rocm-device-lib-path={device_libs}")
            break
    if (rocm_root / "include" / "thrust").is_dir():
        cuda_cflags.append(f"-isystem{rocm_root / 'include'}")
    extra_cuda_cflags = shlex.split(os.environ.get("PI05_W4A4_EXTRA_CFLAGS", ""))
    if extra_cuda_cflags:
        cuda_cflags.extend(extra_cuda_cflags)
        print(f"w4a4: extra HIP flags: {' '.join(extra_cuda_cflags)}", flush=True)
    _mod = load(
        name="pi05_fast_w4a4_int4",
        sources=[str(csrc / "int4_gemm.cu")],
        extra_cuda_cflags=cuda_cflags,
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=[f"-L{link_dir}", f"-L{rocm_lib}", f"-Wl,-rpath,{rocm_lib}"],
        verbose=os.environ.get("PI05_W4A4_VERBOSE", "0") == "1",
    )
    return _mod


def _register_custom_op() -> None:
    global _op_registered
    if _op_registered:
        return
    try:
        torch.ops.pi05_w4a4.int4_gemm2_tiled
        _op_registered = True
        return
    except (AttributeError, RuntimeError):
        pass

    def _fake_out(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch.empty(*x.shape[:-1], packed.shape[0], device=x.device, dtype=torch.float16)

    @torch.library.custom_op("pi05_w4a4::int4_gemm", mutates_args=())
    def _int4_gemm(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return load_int4_gemm().int4_gemm(x, packed)

    @_int4_gemm.register_fake
    def _int4_gemm_fake(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return _fake_out(x, packed)

    @torch.library.custom_op("pi05_w4a4::int4_gemm_tiled", mutates_args=())
    def _int4_gemm_tiled(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return load_int4_gemm().int4_gemm_tiled(x, packed)

    @_int4_gemm_tiled.register_fake
    def _int4_gemm_tiled_fake(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return _fake_out(x, packed)

    @torch.library.custom_op("pi05_w4a4::int4_gemm2", mutates_args=())
    def _int4_gemm2(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return load_int4_gemm().int4_gemm2(x, w0, w1)

    @_int4_gemm2.register_fake
    def _int4_gemm2_fake(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor):
        return _fake_out(x, w0), _fake_out(x, w1)

    @torch.library.custom_op("pi05_w4a4::int4_gemm2_tiled", mutates_args=())
    def _int4_gemm2_tiled(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return load_int4_gemm().int4_gemm2_tiled(x, w0, w1)

    @_int4_gemm2_tiled.register_fake
    def _int4_gemm2_tiled_fake(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor):
        return _fake_out(x, w0), _fake_out(x, w1)

    @torch.library.custom_op("pi05_w4a4::int4_gemm3", mutates_args=())
    def _int4_gemm3(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return load_int4_gemm().int4_gemm3(x, w0, w1, w2)

    @_int4_gemm3.register_fake
    def _int4_gemm3_fake(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ):
        return _fake_out(x, w0), _fake_out(x, w1), _fake_out(x, w2)

    _op_registered = True


_register_custom_op()


def int4_gemm(x: torch.Tensor, packed_w: torch.Tensor) -> torch.Tensor:
    """Compiled-graph-friendly INT4 GEMM. x: fp16 [..., K], packed: int32 [N, K/8+1]."""
    return torch.ops.pi05_w4a4.int4_gemm(x, packed_w)


def int4_gemm_tiled(x: torch.Tensor, packed_w: torch.Tensor) -> torch.Tensor:
    """INT4 GEMM with weights stored in WMMA-fragment-major order."""
    return torch.ops.pi05_w4a4.int4_gemm_tiled(x, packed_w)


def int4_gemm2(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.pi05_w4a4.int4_gemm2(x, w0, w1)


def int4_gemm2_tiled(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared activation quant plus two preshuffled INT4 GEMMs."""
    return torch.ops.pi05_w4a4.int4_gemm2_tiled(x, w0, w1)


def int4_gemm3(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.pi05_w4a4.int4_gemm3(x, w0, w1, w2)


def preload() -> None:
    """JIT the HIP extension before torch.compile traces the graph."""
    load_int4_gemm()
    _register_custom_op()
