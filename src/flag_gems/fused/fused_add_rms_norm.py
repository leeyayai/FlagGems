import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel A: multi-row kernel, used when N is small/medium.
#
# Each program processes `ROWS` rows at once instead of just one. This:
#   - amortizes the (small) weight load over several rows,
#   - shrinks the grid, which helps when M is small and the GPU would
#     otherwise be under-occupied with tiny single-row programs,
#   - is autotuned instead of hand-tuned, so the same code finds a good
#     configuration on NVIDIA GPUs, MetaX C500, etc. without vendor branches.
#
# Only used when the reduced (last) dimension is contiguous (stride == 1) in
# both tensors; the row stride is still a runtime argument so a non-default
# row stride (e.g. a view into a larger contiguous buffer) needs no copy.
#
# NOTE on ROWS bounds: register usage scales with ROWS * BLOCK_N. The config
# list below is a conservative starting point (max 4 rows * 2048 cols in
# fp32 accumulation) to avoid register-spill/occupancy cliffs; profile with
# `TRITON_PRINT_AUTOTUNING=1` on the target hardware and widen if it helps.
# ---------------------------------------------------------------------------
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"ROWS": rows}, num_warps=warps)
        for rows in (1, 2, 4)
        for warps in (2, 4, 8)
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_multirow_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    M,
    in_stride_r,
    r_stride_r,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    # hint the backend that the reduced dimension is contiguous and aligned,
    # so it can emit wide vectorized loads/stores instead of scalar ones.
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N)
    row_ok = rows < M

    in_offsets = rows[:, None] * in_stride_r + cols[None, :]
    r_offsets = rows[:, None] * r_stride_r + cols[None, :]

    if EVEN_N:
        mask = row_ok[:, None]
        x = tl.load(input_ptr + in_offsets, mask=mask, other=0.0).to(cdtype)
        r = tl.load(residual_ptr + r_offsets, mask=mask, other=0.0).to(cdtype)
        w = tl.load(w_ptr + cols)
    else:
        col_ok = cols < N
        mask = row_ok[:, None] & col_ok[None, :]
        x = tl.load(input_ptr + in_offsets, mask=mask, other=0.0).to(cdtype)
        r = tl.load(residual_ptr + r_offsets, mask=mask, other=0.0).to(cdtype)
        w = tl.load(w_ptr + cols, mask=col_ok, other=0.0)

    s = x + r
    # write the residual (x + r) back once; this is the only extra store
    # beyond the theoretical minimum of 2 reads + 2 writes per element.
    tl.store(residual_ptr + r_offsets, s, mask=mask)

    var = tl.sum(s * s, axis=1) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    y = (s * rrms[:, None]).to(input_ptr.dtype.element_ty) * w[None, :]
    tl.store(input_ptr + in_offsets, y.to(cdtype), mask=mask)


# ---------------------------------------------------------------------------
# Kernel B: generic strided, single-row-per-program kernel.
#
# Kept as the fallback for the rare case where the reduced dimension is NOT
# contiguous in `x` and/or `residual` (e.g. a transposed view that `.view()`
# can still express without a copy, just with stride != 1). Also used as the
# single-pass kernel for the upper part of the "small/medium N" range, where
# multi-row batching stops being worth the extra register pressure.
#
# Only `num_warps` is autotuned here; BLOCK_SIZE is fixed to the next power
# of two of N and computed in Python, matching the original kernel.
# ---------------------------------------------------------------------------
@libentry()
@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8, 16)],
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
    r = tl.load(residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

    x += r
    # write back to residual
    tl.store(residual_ptr + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = ((x * rrms).to(input_ptr.dtype.element_ty) * w).to(cdtype)
    # write back to input
    tl.store(input_ptr + cols * in_stride_c, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel C: two-pass loop kernel for large N that doesn't fit one block.
#
# Pass 1 computes x + r, writes it back to `residual`, and accumulates the
# sum of squares. Pass 2 re-reads the updated `residual` and writes the
# normalized*weight result to `input`. This re-read is the one place this
# kernel is above the 2-read/2-write theoretical minimum; the eviction hints
# below are a best-effort nudge to keep the just-written residual block in
# L2 for pass 2 rather than evicting it in favor of `x`, which pass 1 never
# touches again. (Support for `eviction_policy` depends on the Triton/MXMACA
# backend version in use; if unsupported it's safe to drop.)
#
# BLOCK_SIZE/num_warps are autotuned; the entry point only switches to this
# kernel once N is large enough that a single block can't hold a whole row
# (raised from the original 4096 threshold -- see `fused_add_rms_norm`).
# ---------------------------------------------------------------------------
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for bs in (1024, 2048, 4096)
        for w in (4, 8, 16)
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_loop_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = ext.program_id(0)
    row_start = pid * N

    # Pass 1: add residual and compute variance
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    num_steps = tl.cdiv(N, BLOCK_SIZE)

    for step in range(0, num_steps):
        start_n = step * BLOCK_SIZE
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # `x` is never read again after this pass -> let it be evicted first
        x = tl.load(
            input_ptr + row_start + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        r = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0).to(cdtype)
        x = x + r
        # write back to residual; pass 2 will re-read this block shortly
        tl.store(residual_ptr + row_start + cols, x, mask=mask)
        var_acc += x * x

    var = tl.sum(var_acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Pass 2: normalize and write back to input
    for step in range(0, num_steps):
        start_n = step * BLOCK_SIZE
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Re-read from residual (which now has x+r)
        x = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0).to(cdtype)
        # weight is reused by every row -> keep it around in cache longer
        w = tl.load(
            w_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last"
        )
        y = ((x * rrms).to(input_ptr.dtype.element_ty) * w).to(cdtype)
        tl.store(input_ptr + row_start + cols, y, mask=mask)


# Above this many columns, a single block can no longer comfortably hold a
# whole row in registers alongside the fp32 accumulator, so we switch to the
# two-pass loop kernel. Raised from 4096 -> 8192: for N in (4096, 8192] a
# single-pass kernel (kernel B) with more warps still keeps the whole row in
# registers and avoids kernel C's extra residual re-read. Re-check this
# threshold on the target GPU if dtype/occupancy limits differ noticeably.
_SINGLE_PASS_MAX_N = 8192
# Below this many columns, multi-row batching (kernel A) is attempted; above
# it, the per-program register cost of batching multiple rows outweighs the
# benefit and we fall back to one row per program (kernel B) instead.
_MULTIROW_MAX_N = 2048


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.

    Unlike the original implementation, this does not unconditionally call
    `.contiguous()` on `x`/`residual`. It first tries a zero-copy `.view()`
    into a (M, N) shape; this both avoids an extra copy kernel and preserves
    true in-place aliasing with the caller's original storage whenever the
    memory layout allows it (anything PyTorch could express as a reshape
    without copying). `.contiguous()` is only used as a last resort when the
    layout genuinely requires a copy -- in that case the returned tensor is
    a new one and no longer aliases whatever the caller passed in, same as
    any other PyTorch op that can't avoid a copy.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "GEMS FUSED_ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
            x.size(),
            residual.size(),
            weight.size(),
        )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    try:
        x2 = x.view(M, N)
    except RuntimeError:
        x = x.contiguous()
        x2 = x.view(M, N)
    try:
        r2 = residual.view(M, N)
    except RuntimeError:
        residual = residual.contiguous()
        r2 = residual.view(M, N)

    weight = weight.contiguous()

    in_sr, in_sc = x2.stride(0), x2.stride(1)
    r_sr, r_sc = r2.stride(0), r2.stride(1)

    with torch_device_fn.device(x.device):
        if in_sc == 1 and r_sc == 1:
            # Fast paths: reduced dimension is contiguous in both tensors.
            if N <= _MULTIROW_MAX_N:
                BLOCK_N = triton.next_power_of_2(N)
                grid = lambda meta: (triton.cdiv(M, meta["ROWS"]),)  # noqa: E731
                fused_add_rms_norm_multirow_kernel[grid](
                    input_ptr=x2,
                    residual_ptr=r2,
                    w_ptr=weight,
                    M=M,
                    in_stride_r=in_sr,
                    r_stride_r=r_sr,
                    N=N,
                    eps=eps,
                    BLOCK_N=BLOCK_N,
                    EVEN_N=(N == BLOCK_N),
                )
            elif N <= _SINGLE_PASS_MAX_N:
                BLOCK_SIZE = triton.next_power_of_2(N)
                fused_add_rms_norm_kernel[(M,)](
                    input_ptr=x2,
                    residual_ptr=r2,
                    w_ptr=weight,
                    in_stride_r=in_sr,
                    in_stride_c=1,
                    r_stride_r=r_sr,
                    r_stride_c=1,
                    N=N,
                    eps=eps,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            else:
                fused_add_rms_norm_loop_kernel[(M,)](
                    input_ptr=x2,
                    residual_ptr=r2,
                    w_ptr=weight,
                    N=N,
                    eps=eps,
                )
        else:
            # Rare path: a non-default stride on the reduced dimension of at
            # least one tensor. Still handled without an extra copy via the
            # fully strided kernel.
            BLOCK_SIZE = triton.next_power_of_2(N)
            fused_add_rms_norm_kernel[(M,)](
                input_ptr=x2,
                residual_ptr=r2,
                w_ptr=weight,
                in_stride_r=in_sr,
                in_stride_c=in_sc,
                r_stride_r=r_sr,
                r_stride_c=r_sc,
                N=N,
                eps=eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    return x, residual



# # Copyright 2026 FlagOS Contributors
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import logging
# import math

# import triton
# import triton.language as tl

# from flag_gems import runtime
# from flag_gems.runtime import torch_device_fn
# from flag_gems.utils import libentry
# from flag_gems.utils import triton_lang_extension as ext

# logger = logging.getLogger(__name__)


# @triton.jit(do_not_specialize=["eps"])
# def _fast_fused_add_rms_norm_kernel(
#     input_ptr,
#     residual_ptr,
#     w_ptr,
#     N: tl.constexpr,
#     eps,
#     BLOCK_SIZE: tl.constexpr,
#     EVEN_N: tl.constexpr,
# ):
#     row = tl.program_id(0)
#     cols = tl.max_contiguous(
#         tl.multiple_of(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE), BLOCK_SIZE
#     )
#     offsets = row * N + cols
#     if EVEN_N:
#         x = tl.load(input_ptr + offsets).to(tl.float32)
#         residual = tl.load(residual_ptr + offsets).to(tl.float32)
#         weight = tl.load(w_ptr + cols)
#     else:
#         mask = cols < N
#         x = tl.load(input_ptr + offsets, mask, other=0.0).to(tl.float32)
#         residual = tl.load(residual_ptr + offsets, mask, other=0.0).to(tl.float32)
#         weight = tl.load(w_ptr + cols, mask, other=0.0)

#     summed = x + residual
#     tl.store(residual_ptr + offsets, summed, mask=cols < N)
#     rrms = tl.math.rsqrt(tl.sum(summed * summed, axis=0) / N + eps)
#     normalized = (summed * rrms).to(input_ptr.dtype.element_ty)
#     output = normalized * weight
#     tl.store(input_ptr + offsets, output, mask=cols < N)


# @libentry()
# @triton.jit(do_not_specialize=["eps"])
# def fused_add_rms_norm_kernel(
#     input_ptr,  # pointer to the input
#     residual_ptr,  # pointer to the residual
#     w_ptr,  # pointer to the weights
#     in_stride_r,  # how much to increase the pointer when moving by 1 row
#     in_stride_c,  # how much to increase the pointer when moving by 1 col
#     r_stride_r,  # how much to increase the pointer when moving by 1 row
#     r_stride_c,  # how much to increase the pointer when moving by 1 col
#     N,  # number of columns in in_ptr
#     eps,  # epsilon to avoid division by zero
#     BLOCK_SIZE: tl.constexpr,
# ):
#     if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
#         input_ptr.dtype.element_ty == tl.bfloat16
#     ):
#         cdtype = tl.float32
#     else:
#         cdtype = input_ptr.dtype.element_ty

#     pid = ext.program_id(0)
#     input_ptr += pid * in_stride_r
#     residual_ptr += pid * r_stride_r

#     mask = tl.arange(0, BLOCK_SIZE) < N
#     cols = tl.arange(0, BLOCK_SIZE)
#     x = tl.load(input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
#     r = tl.load(residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

#     x += r
#     # write back to residual
#     tl.store(residual_ptr + cols * r_stride_c, x, mask=mask)

#     var = tl.sum(x * x / N, axis=0)
#     rrms = 1 / tl.sqrt(var + eps)

#     w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
#     y = ((x * rrms).to(input_ptr.dtype.element_ty) * w).to(cdtype)
#     # write back to input
#     tl.store(input_ptr + cols * in_stride_c, y, mask=mask)


# @libentry()
# @triton.jit(do_not_specialize=["eps"])
# def fused_add_rms_norm_loop_kernel(
#     input_ptr,  # pointer to the input
#     residual_ptr,  # pointer to the residual
#     w_ptr,  # pointer to the weights
#     N,  # number of columns in in_ptr
#     eps,  # epsilon to avoid division by zero
#     BLOCK_SIZE: tl.constexpr,
# ):
#     if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
#         input_ptr.dtype.element_ty == tl.bfloat16
#     ):
#         cdtype = tl.float32
#     else:
#         cdtype = input_ptr.dtype.element_ty

#     pid = ext.program_id(0)
#     row_start = pid * N

#     # Pass 1: add residual and compute variance
#     var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
#     num_steps = tl.cdiv(N, BLOCK_SIZE)

#     for step in range(0, num_steps):
#         start_n = step * BLOCK_SIZE
#         cols = start_n + tl.arange(0, BLOCK_SIZE)
#         mask = cols < N
#         x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(cdtype)
#         r = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0).to(cdtype)
#         x = x + r
#         # write back to residual
#         tl.store(residual_ptr + row_start + cols, x, mask=mask)
#         var_acc += x * x

#     var = tl.sum(var_acc) / N
#     rrms = 1 / tl.sqrt(var + eps)

#     # Pass 2: normalize and write back to input
#     for step in range(0, num_steps):
#         start_n = step * BLOCK_SIZE
#         cols = start_n + tl.arange(0, BLOCK_SIZE)
#         mask = cols < N
#         # Re-read from residual (which now has x+r)
#         x = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0).to(cdtype)
#         w = tl.load(w_ptr + cols, mask=mask, other=0.0)
#         y = ((x * rrms).to(input_ptr.dtype.element_ty) * w).to(cdtype)
#         tl.store(input_ptr + row_start + cols, y, mask=mask)


# def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
#     """
#     This function performs fused residual addition and RMS normalization **in-place**.
#     Both `x` and `residual` tensors will be modified. Use with caution if these tensors
#     are reused elsewhere or require gradients.
#     """
#     logger.debug(
#         "GEMS FUSED_ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
#         x.size(),
#         residual.size(),
#         weight.size(),
#     )
#     dim = x.ndim - len(normalized_shape)
#     M = math.prod(x.shape[:dim])
#     N = math.prod(normalized_shape)

#     if not x.is_contiguous():
#         x = x.contiguous()
#     if not residual.is_contiguous():
#         residual = residual.contiguous()
#     weight = weight.contiguous()

#     with torch_device_fn.device(x.device):
#         if N <= 4096:
#             BLOCK_SIZE = triton.next_power_of_2(N)
#             if runtime.device.vendor_name == "metax":
#                 num_warps = (
#                     4 if M >= 512 and BLOCK_SIZE >= 1024 else
#                     8 if M < 512 and BLOCK_SIZE >= 2048 else
#                     4
#                 )
#                 _fast_fused_add_rms_norm_kernel[(M,)](
#                     x,
#                     residual,
#                     weight,
#                     N,
#                     eps,
#                     BLOCK_SIZE,
#                     N == BLOCK_SIZE,
#                     num_warps=num_warps,
#                 )
#             else:
#                 fused_add_rms_norm_kernel[M,](
#                     x, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
#                 )
#         else:
#             BLOCK_SIZE = 1024
#             fused_add_rms_norm_loop_kernel[M,](x, residual, weight, N, eps, BLOCK_SIZE)
#     return x, residual
