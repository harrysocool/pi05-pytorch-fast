// Embedl-compatible W4A4 GEMM: Hadamard (selected K) + packed INT4 + iu4 WMMA.
// Layout: X[M,K] fp16, W[N, K/8+1] int32 (last col = per-output-channel scale bits).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

typedef int int2v __attribute__((ext_vector_type(2)));
typedef int int8v __attribute__((ext_vector_type(8)));

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 64
#endif
#ifndef DOWN_BK
#define DOWN_BK 128
#endif
#ifndef GATE_SPECIAL
#define GATE_SPECIAL 1
#endif
#ifndef GEMM_TPB
#define GEMM_TPB 256
#endif
#ifndef MI
#define MI 2
#endif
#ifndef NI
#define NI 4
#endif
#define QUANT_TPB 256
#define NWAVE_N (BN / (NI * 16))

static_assert(BM % (MI * 16) == 0, "BM must be divisible by the per-wave M tile");
static_assert(BN % (NI * 16) == 0, "BN must be divisible by the per-wave N tile");
static_assert(
    (BM / (MI * 16)) * NWAVE_N == GEMM_TPB / 32,
    "GEMM_TPB does not match the BM/BN wave layout");

template <bool TILED_B>
__device__ __forceinline__ long weight_word_offset(
    int col, int word, int N, int K, int ROW) {
  if constexpr (TILED_B) {
    return ((long)(word >> 1) * N + col) * 2 + (word & 1);
  } else {
    return (long)col * ROW + word;
  }
}

template <bool TILED_B>
__device__ __forceinline__ long weight_scale_offset(int col, int N, int K, int ROW) {
  if constexpr (TILED_B) {
    return (long)N * (K / 8) + col;
  } else {
    return (long)col * ROW + (K / 8);
  }
}

// Register-prefetched single-LDS mode keeps global loads overlapped with WMMA,
// but reuses the same LDS tile after the post-compute barrier. This cuts LDS
// residency without changing accumulation order.
template <int TILE_K, bool SINGLE_BUFFER = false, bool TILED_B = false>
__global__ void gemm_iu4(
    const int* A,
    const int* B,
    _Float16* Cf,
    const float* sa,
    int M,
    int N,
    int K,
    int ROW) {
  constexpr int KPW = TILE_K / 8;
  constexpr int LDAi = KPW + 1;
  constexpr int NLA = (BM * KPW) / GEMM_TPB;
  constexpr int NLB = (BN * KPW) / GEMM_TPB;
  static_assert(TILE_K % 16 == 0, "TILE_K must be a multiple of WMMA K");
  static_assert((BM * KPW) % GEMM_TPB == 0, "A cooperative load must divide evenly");
  static_assert((BN * KPW) % GEMM_TPB == 0, "B cooperative load must divide evenly");
  __shared__ int As[SINGLE_BUFFER ? 1 : 2][BM * LDAi];
  __shared__ int Bs[SINGLE_BUFFER ? 1 : 2][BN * LDAi];
  int bm = blockIdx.y * BM, bn = blockIdx.x * BN, tid = threadIdx.x;
  int wid = tid / 32, lid = tid % 32, lane = lid % 16;
  int wm = wid / NWAVE_N, wn = wid % NWAVE_N;
  int8v c[MI][NI];
  for (int i = 0; i < MI; i++)
    for (int j = 0; j < NI; j++) c[i][j] = int8v{};
  for (int i = 0; i < NLA; i++) {
    int x = tid + i * GEMM_TPB;
    int r = x / KPW, kk = x % KPW;
    int gr = bm + r;
    As[0][r * LDAi + kk] = (gr < M) ? A[gr * (K / 8) + kk] : 0;
  }
  for (int i = 0; i < NLB; i++) {
    int x = tid + i * GEMM_TPB;
    int co, kk;
    if constexpr (TILED_B) {
      int pair = x / 2;
      co = pair % BN;
      kk = (pair / BN) * 2 + (x & 1);
    } else {
      co = x / KPW;
      kk = x % KPW;
    }
    int gc = bn + co;
    Bs[0][co * LDAi + kk] =
        (gc < N) ? B[weight_word_offset<TILED_B>(gc, kk, N, K, ROW)] : 0;
  }
  __syncthreads();
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    int nk = k0 + TILE_K;
    int ra[NLA], rb[NLB];
    if (nk < K) {
      for (int i = 0; i < NLA; i++) {
        int x = tid + i * GEMM_TPB;
        int r = x / KPW, kk = x % KPW;
        int gr = bm + r;
        ra[i] = (gr < M) ? A[gr * (K / 8) + (nk / 8 + kk)] : 0;
      }
      for (int i = 0; i < NLB; i++) {
        int x = tid + i * GEMM_TPB;
        int co, kk;
        if constexpr (TILED_B) {
          int pair = x / 2;
          co = pair % BN;
          kk = (pair / BN) * 2 + (x & 1);
        } else {
          co = x / KPW;
          kk = x % KPW;
        }
        int gc = bn + co;
        rb[i] = (gc < N)
            ? B[weight_word_offset<TILED_B>(gc, nk / 8 + kk, N, K, ROW)]
            : 0;
      }
    }
    for (int ks = 0; ks < TILE_K; ks += 16)
      for (int mi = 0; mi < MI; mi++)
        for (int ni = 0; ni < NI; ni++) {
          int r0 = wm * 32 + mi * 16, c0 = wn * 64 + ni * 16;
          int2v af, bf;
          af[0] = As[buf][(r0 + lane) * LDAi + ks / 8];
          af[1] = As[buf][(r0 + lane) * LDAi + ks / 8 + 1];
          bf[0] = Bs[buf][(c0 + lane) * LDAi + ks / 8];
          bf[1] = Bs[buf][(c0 + lane) * LDAi + ks / 8 + 1];
          c[mi][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af, true, bf, c[mi][ni], false);
        }
    __syncthreads();
    if (nk < K) {
      int next_buf = SINGLE_BUFFER ? 0 : (buf ^ 1);
      for (int i = 0; i < NLA; i++) {
        int x = tid + i * GEMM_TPB;
        As[next_buf][x / KPW * LDAi + x % KPW] = ra[i];
      }
      for (int i = 0; i < NLB; i++)
      {
        int x = tid + i * GEMM_TPB;
        int co, kk;
        if constexpr (TILED_B) {
          int pair = x / 2;
          co = pair % BN;
          kk = (pair / BN) * 2 + (x & 1);
        } else {
          co = x / KPW;
          kk = x % KPW;
        }
        Bs[next_buf][co * LDAi + kk] = rb[i];
      }
      __syncthreads();
    }
    if constexpr (!SINGLE_BUFFER) {
      buf ^= 1;
    }
  }
  // Reuse each output-column scale across both M fragments, and each row
  // scale across all N fragments owned by this lane.
  float col_scale[NI];
  for (int ni = 0; ni < NI; ni++) {
    int c0 = wn * 64 + ni * 16;
    int col = bn + c0 + lane;
    if (col < N)
      col_scale[ni] = __int_as_float(B[weight_scale_offset<TILED_B>(col, N, K, ROW)]);
  }
  for (int mi = 0; mi < MI; mi++) {
    int r0 = wm * 32 + mi * 16;
    for (int e = 0; e < 8; e++) {
      int r = 2 * e + lid / 16;
      int gr = bm + r0 + r;
      if (gr >= M) continue;
      float row_scale = sa[gr];
      for (int ni = 0; ni < NI; ni++) {
        int c0 = wn * 64 + ni * 16;
        int col = bn + c0 + lane;
        if (col < N)
          Cf[gr * N + col] = (_Float16)(
              (float)c[mi][ni][e] * row_scale * col_scale[ni]);
      }
    }
  }
}

// Shape-specialized companion used only by the M=776 gate/up projections.
template <int TILE_M, int TILE_N, int TILE_K, int THREADS, int M_ITERS, int N_ITERS,
          bool SINGLE_BUFFER = false, bool TILED_B = false, bool FULL_TILE = false>
__global__ void gemm_iu4_shape(
    const int* A,
    const int* B,
    _Float16* Cf,
    const float* sa,
    int M,
    int N,
    int K,
    int ROW) {
  constexpr int KPW = TILE_K / 8;
  constexpr int LDAi = KPW + 1;
  constexpr int A_WORDS = TILE_M * KPW;
  constexpr int B_WORDS = TILE_N * KPW;
  constexpr int NLA = (A_WORDS + THREADS - 1) / THREADS;
  constexpr int NLB = (B_WORDS + THREADS - 1) / THREADS;
  constexpr int N_WAVES = TILE_N / (N_ITERS * 16);
  static_assert(TILE_K % 16 == 0, "TILE_K must be a multiple of WMMA K");
  static_assert(TILE_M % (M_ITERS * 16) == 0, "invalid M wave tile");
  static_assert(TILE_N % (N_ITERS * 16) == 0, "invalid N wave tile");
  static_assert(
      (TILE_M / (M_ITERS * 16)) * N_WAVES == THREADS / 32,
      "thread count does not match the wave layout");

  __shared__ int As[SINGLE_BUFFER ? 1 : 2][TILE_M * LDAi];
  __shared__ int Bs[SINGLE_BUFFER ? 1 : 2][TILE_N * LDAi];
  int bm = blockIdx.y * TILE_M, bn = blockIdx.x * TILE_N, tid = threadIdx.x;
  if constexpr (FULL_TILE) {
    // This specialization is launched only for the full gate/up body. Keeping
    // its fixed dimensions visible lets clang fold the address arithmetic.
    __builtin_assume(K == 2048);
    __builtin_assume(N == 16384);
  }
  int wid = tid / 32, lid = tid % 32, lane = lid % 16;
  int wm = wid / N_WAVES, wn = wid % N_WAVES;
  int8v c[M_ITERS][N_ITERS];
  for (int i = 0; i < M_ITERS; i++)
    for (int j = 0; j < N_ITERS; j++) c[i][j] = int8v{};

  for (int i = 0; i < NLA; i++) {
    int x = tid + i * THREADS;
    if (x >= A_WORDS) continue;
    int r = x / KPW, kk = x % KPW;
    int gr = bm + r;
    if constexpr (FULL_TILE)
      As[0][r * LDAi + kk] = A[gr * (K / 8) + kk];
    else
      As[0][r * LDAi + kk] = (gr < M) ? A[gr * (K / 8) + kk] : 0;
  }
  for (int i = 0; i < NLB; i++) {
    int x = tid + i * THREADS;
    if (x >= B_WORDS) continue;
    int co, kk;
    if constexpr (TILED_B) {
      int pair = x / 2;
      co = pair % TILE_N;
      kk = (pair / TILE_N) * 2 + (x & 1);
    } else {
      co = x / KPW;
      kk = x % KPW;
    }
    int gc = bn + co;
    if constexpr (FULL_TILE)
      Bs[0][co * LDAi + kk] = B[weight_word_offset<TILED_B>(gc, kk, N, K, ROW)];
    else
      Bs[0][co * LDAi + kk] =
          (gc < N) ? B[weight_word_offset<TILED_B>(gc, kk, N, K, ROW)] : 0;
  }
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    int nk = k0 + TILE_K;
    int ra[NLA], rb[NLB];
    if (nk < K) {
      for (int i = 0; i < NLA; i++) {
        int x = tid + i * THREADS;
        if (x >= A_WORDS) {
          ra[i] = 0;
          continue;
        }
        int r = x / KPW, kk = x % KPW;
        int gr = bm + r;
        if constexpr (FULL_TILE)
          ra[i] = A[gr * (K / 8) + (nk / 8 + kk)];
        else
          ra[i] = (gr < M) ? A[gr * (K / 8) + (nk / 8 + kk)] : 0;
      }
      for (int i = 0; i < NLB; i++) {
        int x = tid + i * THREADS;
        if (x >= B_WORDS) {
          rb[i] = 0;
          continue;
        }
        int co, kk;
        if constexpr (TILED_B) {
          int pair = x / 2;
          co = pair % TILE_N;
          kk = (pair / TILE_N) * 2 + (x & 1);
        } else {
          co = x / KPW;
          kk = x % KPW;
        }
        int gc = bn + co;
        if constexpr (FULL_TILE)
          rb[i] = B[weight_word_offset<TILED_B>(gc, nk / 8 + kk, N, K, ROW)];
        else
          rb[i] = (gc < N)
              ? B[weight_word_offset<TILED_B>(gc, nk / 8 + kk, N, K, ROW)]
              : 0;
      }
    }
    for (int ks = 0; ks < TILE_K; ks += 16)
      for (int mi = 0; mi < M_ITERS; mi++)
        for (int ni = 0; ni < N_ITERS; ni++) {
          int r0 = wm * (M_ITERS * 16) + mi * 16;
          int c0 = wn * (N_ITERS * 16) + ni * 16;
          int2v af, bf;
          af[0] = As[buf][(r0 + lane) * LDAi + ks / 8];
          af[1] = As[buf][(r0 + lane) * LDAi + ks / 8 + 1];
          bf[0] = Bs[buf][(c0 + lane) * LDAi + ks / 8];
          bf[1] = Bs[buf][(c0 + lane) * LDAi + ks / 8 + 1];
          c[mi][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, af, true, bf, c[mi][ni], false);
        }
    __syncthreads();
    if (nk < K) {
      int next_buf = SINGLE_BUFFER ? 0 : (buf ^ 1);
      for (int i = 0; i < NLA; i++) {
        int x = tid + i * THREADS;
        if (x < A_WORDS) As[next_buf][x / KPW * LDAi + x % KPW] = ra[i];
      }
      for (int i = 0; i < NLB; i++) {
        int x = tid + i * THREADS;
        if (x < B_WORDS) {
          int co, kk;
          if constexpr (TILED_B) {
            int pair = x / 2;
            co = pair % TILE_N;
            kk = (pair / TILE_N) * 2 + (x & 1);
          } else {
            co = x / KPW;
            kk = x % KPW;
          }
          Bs[next_buf][co * LDAi + kk] = rb[i];
        }
      }
      __syncthreads();
    }
    if constexpr (!SINGLE_BUFFER) {
      buf ^= 1;
    }
  }

  // The main gate/up tile is interior in both dimensions. Cache scales once
  // per row/column instead of reloading them for every accumulator element.
  float col_scale[N_ITERS];
  for (int ni = 0; ni < N_ITERS; ni++) {
    int c0 = wn * (N_ITERS * 16) + ni * 16;
    int col = bn + c0 + lane;
    if constexpr (FULL_TILE)
      col_scale[ni] = __int_as_float(B[weight_scale_offset<TILED_B>(col, N, K, ROW)]);
    else if (col < N)
      col_scale[ni] = __int_as_float(B[weight_scale_offset<TILED_B>(col, N, K, ROW)]);
  }
  for (int mi = 0; mi < M_ITERS; mi++) {
    int r0 = wm * (M_ITERS * 16) + mi * 16;
    for (int e = 0; e < 8; e++) {
      int r = 2 * e + lid / 16;
      int gr = bm + r0 + r;
      if constexpr (!FULL_TILE) {
        if (gr >= M) continue;
      }
      float row_scale = sa[gr];
      for (int ni = 0; ni < N_ITERS; ni++) {
        int c0 = wn * (N_ITERS * 16) + ni * 16;
        int col = bn + c0 + lane;
        if constexpr (FULL_TILE)
          Cf[gr * N + col] = (_Float16)(
              (float)c[mi][ni][e] * row_scale * col_scale[ni]);
        else if (col < N)
          Cf[gr * N + col] = (_Float16)(
              (float)c[mi][ni][e] * row_scale * col_scale[ni]);
      }
    }
  }
}

template <int K, int R>
__global__ void quant_rot_rb(const _Float16* X, int* Q, float* sc, int M, long RS) {
  int row = blockIdx.x;
  if (row >= M) return;
  int tid = threadIdx.x;
  __shared__ _Float16 lds[K];
  __shared__ float red[QUANT_TPB];
  float reg[R];
#pragma unroll
  for (int j = 0; j < R; j++) reg[j] = (float)X[row * RS + tid * R + j];
#pragma unroll
  for (int len = 1; len < R; len <<= 1)
    for (int i = 0; i < R; i += 2 * len)
      for (int j = i; j < i + len; j++) {
        float a = reg[j], b = reg[j + len];
        reg[j] = a + b;
        reg[j + len] = a - b;
      }
#pragma unroll
  for (int j = 0; j < R; j++) lds[tid * R + j] = (_Float16)reg[j];
  __syncthreads();
  for (int len = R; len < K; len <<= 1) {
    for (int idx = tid; idx < K / 2; idx += QUANT_TPB) {
      int group = idx / len, off = idx % len;
      int p = group * 2 * len + off;
      float a = (float)lds[p], b = (float)lds[p + len];
      lds[p] = (_Float16)(a + b);
      lds[p + len] = (_Float16)(a - b);
    }
    __syncthreads();
  }
  float m = 0;
  for (int k = tid; k < K; k += QUANT_TPB) {
    float v = fabsf((float)lds[k]);
    m = fmaxf(m, v);
  }
  red[tid] = m;
  __syncthreads();
  for (int s = QUANT_TPB / 2; s; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float sdiv = red[0] / 7.f + 1e-12f;
  if (tid == 0) sc[row] = sdiv * rsqrtf((float)K);
  for (int p = tid; p < K / 8; p += QUANT_TPB) {
    int pk = 0;
    for (int j = 0; j < 8; j++) {
      int q = (int)lrintf((float)lds[p * 8 + j] / sdiv);
      q = max(-7, min(7, q));
      pk |= (q & 0xF) << (j * 4);
    }
    Q[row * (K / 8) + p] = pk;
  }
}

__global__ void quant_rows(const _Float16* X, int* Q, float* sc, int M, int K) {
  int row = blockIdx.x;
  if (row >= M) return;
  __shared__ float sm[QUANT_TPB];
  float m = 0;
  for (int k = threadIdx.x; k < K; k += QUANT_TPB) {
    float v = fabsf((float)X[row * K + k]);
    m = fmaxf(m, v);
  }
  sm[threadIdx.x] = m;
  __syncthreads();
  for (int s = QUANT_TPB / 2; s; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  float s0 = sm[0] / 7.f + 1e-12f;
  if (threadIdx.x == 0) sc[row] = s0;
  for (int p = threadIdx.x; p < K / 8; p += QUANT_TPB) {
    int pk = 0;
    for (int j = 0; j < 8; j++) {
      int q = (int)lrintf((float)X[row * K + p * 8 + j] / s0);
      q = max(-7, min(7, q));
      pk |= (q & 0xF) << (j * 4);
    }
    Q[row * (K / 8) + p] = pk;
  }
}

struct QuantizedX {
  torch::Tensor x2;
  torch::Tensor q;
  torch::Tensor sa;
  long M = 0;
  int K = 0;
};

static void check_packed(const torch::Tensor& w, int K, const char* name) {
  TORCH_CHECK(w.is_cuda(), name, " must be on CUDA/HIP");
  TORCH_CHECK(w.scalar_type() == torch::kInt32 && w.dim() == 2, name, " must be int32 [N, K/8+1]");
  TORCH_CHECK(
      w.size(1) >= K / 8 + 1,
      name,
      " second dim must contain K/8 packed words plus a scale");
}

static QuantizedX quantize_x(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "int4_gemm expects CUDA/HIP tensors");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "int4_gemm activations must be float16");
  TORCH_CHECK(x.size(-1) > 0);
  const int K = static_cast<int>(x.size(-1));
  TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64, got ", K);

  QuantizedX qx;
  qx.x2 = x.contiguous();
  qx.K = K;
  qx.M = 1;
  for (int i = 0; i + 1 < qx.x2.dim(); i++) qx.M *= qx.x2.size(i);

  const auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(qx.x2.device());
  const auto opts_f = torch::TensorOptions().dtype(torch::kFloat).device(qx.x2.device());
  qx.q = torch::empty({qx.M, (long)(K / 8)}, opts_i);
  qx.sa = torch::empty({qx.M}, opts_f);

  const c10::cuda::CUDAGuard guard(qx.x2.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const _Float16* xp = reinterpret_cast<const _Float16*>(qx.x2.data_ptr<at::Half>());
  int* qp = qx.q.data_ptr<int>();
  float* sap = qx.sa.data_ptr<float>();

  if (K == 16384)
    quant_rot_rb<16384, 64><<<static_cast<int>(qx.M), QUANT_TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  else if (K == 2048)
    quant_rot_rb<2048, 8><<<static_cast<int>(qx.M), QUANT_TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  else if (K == 1024)
    quant_rot_rb<1024, 4><<<static_cast<int>(qx.M), QUANT_TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  else
    quant_rows<<<static_cast<int>(qx.M), QUANT_TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return qx;
}

template <bool TILED_B>
static void launch_gate_up_gemm(
    const QuantizedX& qx, torch::Tensor w, torch::Tensor y, int N, hipStream_t stream) {
  constexpr int MAIN_M = 256;
  const int M = static_cast<int>(qx.M);
  const int main_rows = (M / MAIN_M) * MAIN_M;
  const int tail_rows = M - main_rows;
  const int* a = qx.q.data_ptr<int>();
  const int* b = w.data_ptr<int>();
  _Float16* out = reinterpret_cast<_Float16*>(y.data_ptr<at::Half>());
  const float* scales = qx.sa.data_ptr<float>();
  const int row = static_cast<int>(w.size(1));

  if (main_rows > 0) {
    dim3 grid((N + 127) / 128, main_rows / MAIN_M);
    gemm_iu4_shape<256, 128, 128, 512, 2, 4, true, TILED_B, true><<<grid, 512, 0, stream>>>(
        a, b, out, scales, main_rows, N, qx.K, row);
  }
  if (tail_rows > 0 && tail_rows <= 16) {
    dim3 grid((N + 511) / 512, 1);
    const long a_offset = (long)main_rows * (qx.K / 8);
    const long y_offset = (long)main_rows * N;
    gemm_iu4_shape<16, 512, BK, 256, 1, 4, false, TILED_B><<<grid, 256, 0, stream>>>(
        a + a_offset, b, out + y_offset, scales + main_rows, tail_rows, N, qx.K, row);
  } else if (tail_rows > 0) {
    dim3 grid((N + BN - 1) / BN, 1);
    const long a_offset = (long)main_rows * (qx.K / 8);
    const long y_offset = (long)main_rows * N;
    gemm_iu4<BK, false, TILED_B><<<grid, GEMM_TPB, 0, stream>>>(
        a + a_offset, b, out + y_offset, scales + main_rows, tail_rows, N, qx.K, row);
  }
}

template <bool TILED_B = false>
static torch::Tensor gemm_from_q(const QuantizedX& qx, torch::Tensor w) {
  check_packed(w, qx.K, "packed W");
  const int N = static_cast<int>(w.size(0));
  if constexpr (TILED_B) {
    TORCH_CHECK(
        qx.K == 2048 && N == 16384,
        "WMMA-fragment-major weights currently support only K=2048, N=16384");
    TORCH_CHECK(
        w.size(1) == qx.K / 8 + 9,
        "WMMA-fragment-major weights must use the K/8+9 marker stride");
  }
  const auto opts_h = torch::TensorOptions().dtype(torch::kHalf).device(qx.x2.device());
  auto y = torch::empty({qx.M, (long)N}, opts_h);

  const c10::cuda::CUDAGuard guard(qx.x2.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int row = static_cast<int>(w.size(1));
  dim3 grid((N + BN - 1) / BN, (static_cast<int>(qx.M) + BM - 1) / BM);
#if GATE_SPECIAL
  if (qx.K == 2048 && N == 16384) {
    launch_gate_up_gemm<TILED_B>(qx, w, y, N, stream);
  } else
#endif
  if (qx.K == 16384) {
    gemm_iu4<DOWN_BK, true><<<grid, GEMM_TPB, 0, stream>>>(
        qx.q.data_ptr<int>(),
        w.data_ptr<int>(),
        reinterpret_cast<_Float16*>(y.data_ptr<at::Half>()),
        qx.sa.data_ptr<float>(),
        static_cast<int>(qx.M),
        N,
        qx.K,
        row);
  } else {
    gemm_iu4<BK><<<grid, GEMM_TPB, 0, stream>>>(
        qx.q.data_ptr<int>(),
        w.data_ptr<int>(),
        reinterpret_cast<_Float16*>(y.data_ptr<at::Half>()),
        qx.sa.data_ptr<float>(),
        static_cast<int>(qx.M),
        N,
        qx.K,
        row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto yshape = qx.x2.sizes().vec();
  yshape.back() = N;
  return y.view(yshape);
}

static torch::Tensor int4_gemm(torch::Tensor x, torch::Tensor w) {
  return gemm_from_q<>(quantize_x(x), w);
}

static torch::Tensor int4_gemm_tiled(torch::Tensor x, torch::Tensor w) {
  return gemm_from_q<true>(quantize_x(x), w);
}

static std::tuple<torch::Tensor, torch::Tensor> int4_gemm2(
    torch::Tensor x, torch::Tensor w0, torch::Tensor w1) {
  auto qx = quantize_x(x);
  return {gemm_from_q<>(qx, w0), gemm_from_q<>(qx, w1)};
}

static std::tuple<torch::Tensor, torch::Tensor> int4_gemm2_tiled(
    torch::Tensor x, torch::Tensor w0, torch::Tensor w1) {
  auto qx = quantize_x(x);
  return {gemm_from_q<true>(qx, w0), gemm_from_q<true>(qx, w1)};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> int4_gemm3(
    torch::Tensor x, torch::Tensor w0, torch::Tensor w1, torch::Tensor w2) {
  auto qx = quantize_x(x);
  return {gemm_from_q<>(qx, w0), gemm_from_q<>(qx, w1), gemm_from_q<>(qx, w2)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_gemm", &int4_gemm, "W4A4 packed INT4 GEMM (Hadamard on K in {1024,2048,16384})");
  m.def("int4_gemm_tiled", &int4_gemm_tiled, "W4A4 GEMM with WMMA-fragment-major weights");
  m.def("int4_gemm2", &int4_gemm2, "Shared activation quant + two INT4 GEMMs");
  m.def("int4_gemm2_tiled", &int4_gemm2_tiled, "Shared activation quant + two preshuffled INT4 GEMMs");
  m.def("int4_gemm3", &int4_gemm3, "Shared activation quant + three INT4 GEMMs");
}
