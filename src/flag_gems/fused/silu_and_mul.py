# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems._probe import probe
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_SILU_BLOCK = 1024
_SILU_BLOCK_M = 4


@triton.jit
def silu_and_mul_packed_2d_kernel(
    x_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    X_STRIDE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * X_STRIDE_ROW + cols[None, :]

    a = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + offsets + N, mask=mask, other=0.0)
    y = (a / (1.0 + tl.exp(-a))) * b
    out_offsets = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_and_mul_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    A_STRIDE_ROW,
    B_STRIDE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    a_offsets = rows[:, None] * A_STRIDE_ROW + cols[None, :]
    b_offsets = rows[:, None] * B_STRIDE_ROW + cols[None, :]
    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    y = (a / (1.0 + tl.exp(-a))) * b
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], y.to(out_ptr.dtype.element_ty), mask=mask)


def _can_flatten_leading_dims(x: torch.Tensor) -> bool:
    if x.ndim <= 2:
        return True
    for dim in range(x.ndim - 2, 0, -1):
        if x.shape[dim] > 1 and x.stride(dim) != x.stride(dim + 1) * x.shape[dim + 1]:
            return False
    return True


def _silu_tile_config(rows: int, cols: int) -> tuple[int, int, int]:
    if device.vendor_name != "metax":
        return min(_SILU_BLOCK_M, triton.next_power_of_2(rows)), _SILU_BLOCK, 4
    if cols >= 1024 and rows <= 16:
        return 1, 256, 1
    if cols >= 1024 and rows >= 512:
        return 4, 1024, 4
    if cols >= 1024:
        return 1, 1024, 2
    return 1, max(16, triton.next_power_of_2(cols)), 2


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")], num_outputs=2
)
@triton.jit
def silu_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    sig = 1 / (1 + tl.exp(-x_fp32))
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu
    return dx, dy


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        logger.debug("GEMS SILU AND MUL FORWARD")
        return silu_and_mul_kernel(A, B)

    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        logger.debug("GEMS SILU AND MUL BACKWARD")
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output)
        return grad_A, grad_B


def silu_and_mul(A, B):
    fast_inference = (
        not torch.is_grad_enabled()
        and A.ndim >= 2
        and A.shape == B.shape
        and A.dtype == B.dtype
        and A.device == B.device
        and A.shape[-1] > 0
        and A.stride(-1) == 1
        and B.stride(-1) == 1
        and _can_flatten_leading_dims(A)
        and _can_flatten_leading_dims(B)
    )
    if fast_inference:
        cols = A.shape[-1]
        rows = A.numel() // cols
        if rows == 0:
            return torch.empty(A.shape, dtype=A.dtype, device=A.device)
        a_2d = A.view(rows, cols)
        b_2d = B.view(rows, cols)
        out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
        out_2d = out.view(rows, cols)
        block_m, block_n, num_warps = _silu_tile_config(rows, cols)
        with torch_device_fn.device(A.device):
            silu_and_mul_2d_kernel[(triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))](
                a_2d,
                b_2d,
                out_2d,
                rows,
                cols,
                a_2d.stride(0),
                b_2d.stride(0),
                block_m,
                block_n,
                num_warps=num_warps,
            )
        return out
    return SiluAndMul.apply(A, B)


def silu_and_mul_packed(x):
    if x.ndim < 1 or x.shape[-1] % 2 != 0:
        raise ValueError("packed SiLU-and-mul input must have an even last dimension")

    packed_cols = x.shape[-1]
    cols = packed_cols // 2
    out_shape = x.shape[:-1] + (cols,)
    if x.numel() == 0 or cols == 0:
        return x.new_empty(out_shape)

    if (
        torch.is_grad_enabled()
        or x.stride(-1) != 1
        or not _can_flatten_leading_dims(x)
    ):
        a, b = x.split(cols, dim=-1)
        return silu_and_mul(a, b)

    rows = x.numel() // packed_cols
    x_2d = x.view(rows, packed_cols)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out_2d = out.view(rows, cols)
    block_m, block_n, num_warps = _silu_tile_config(rows, cols)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    with torch_device_fn.device(x.device):
        silu_and_mul_packed_2d_kernel[grid](
            x_2d,
            out_2d,
            rows,
            cols,
            x_2d.stride(0),
            block_m,
            block_n,
            num_warps=num_warps,
        )
    return out


def silu_and_mul_out(A, B, out):
    logger.debug("GEMS SILU AND MUL OUT")
    silu_and_mul_kernel(A, B, out0=out)
    return out
