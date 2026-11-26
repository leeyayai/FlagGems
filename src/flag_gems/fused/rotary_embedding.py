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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def _rotary_head_tiles(
    out_ptr, in_ptr, cos, sin,  # 改动2：cos/sin 改为已加载好的值（寄存器），不再是指针
    in_stride_h, in_stride_d, out_stride_h, out_stride_d,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr, ROTARY_INTERLEAVED: tl.constexpr,
    BLOCK_H: tl.constexpr,  # 改动1：BLOCK_H 不再硬编码为 8，由调用方按头数传入
    NEEDS_MASK: tl.constexpr,  # 改动4：当 PADDED_HEAD_DIM == HEAD_DIM 时跳过 dim_mask
):
    dims = tl.arange(0, PADDED_HEAD_DIM // 2)
    if NEEDS_MASK:
        dim_mask = dims < HEAD_DIM // 2
    if ROTARY_INTERLEAVED:
        first = 2 * dims
        second = first + 1
    else:
        first = dims
        second = dims + HEAD_DIM // 2

    for head_start in range(0, NUM_HEADS, BLOCK_H):
        heads = head_start + tl.arange(0, BLOCK_H)
        if NEEDS_MASK:
            mask = (heads[:, None] < NUM_HEADS) & dim_mask[None, :]
        else:
            mask = heads[:, None] < NUM_HEADS
        in_base = in_ptr + heads[:, None] * in_stride_h
        out_base = out_ptr + heads[:, None] * out_stride_h
        low = tl.load(in_base + first[None, :] * in_stride_d, mask=mask, other=0.0).to(tl.float32)
        high = tl.load(in_base + second[None, :] * in_stride_d, mask=mask, other=0.0).to(tl.float32)
        out_low = low * cos[None, :] - high * sin[None, :]
        out_high = high * cos[None, :] + low * sin[None, :]
        tl.store(out_base + first[None, :] * out_stride_d, out_low, mask=mask)
        tl.store(out_base + second[None, :] * out_stride_d, out_high, mask=mask)


@libentry()
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
    BLOCK_H_Q: tl.constexpr,  # 改动1：Q、K 分别传入各自的 BLOCK_H
    BLOCK_H_K: tl.constexpr,
    NEEDS_MASK: tl.constexpr,  # 改动4
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    # 改动2：cos/sin 在这里只加载一次，Q、K 两次调用复用同一份寄存器值，
    # 不再各自在 _rotary_head_tiles 内部重复从 global memory 读取。
    dims = tl.arange(0, PADDED_HEAD_DIM // 2)
    if NEEDS_MASK:
        dim_mask = dims < HEAD_DIM // 2
        cos = tl.load(cos_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        cos = tl.load(cos_ptr + dims).to(tl.float32)
        sin = tl.load(sin_ptr + dims).to(tl.float32)

    _rotary_head_tiles(
        oq_ptr + s_id * oq_stride_s, q_ptr + s_id * q_stride_s,
        cos, sin, q_stride_h, q_stride_d, oq_stride_h, oq_stride_d,
        NUM_Q_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
        BLOCK_H_Q, NEEDS_MASK,
    )
    _rotary_head_tiles(
        ok_ptr + s_id * ok_stride_s, k_ptr + s_id * k_stride_s,
        cos, sin, k_stride_h, k_stride_d, ok_stride_h, ok_stride_d,
        NUM_K_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
        BLOCK_H_K, NEEDS_MASK,
    )


@libentry()
@triton.jit
def apply_rotary_pos_emb_inplace_kernel(
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
    BLOCK_H_Q: tl.constexpr,
    BLOCK_H_K: tl.constexpr,
    NEEDS_MASK: tl.constexpr,
):
    s_id = ext.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    dims = tl.arange(0, PADDED_HEAD_DIM // 2)
    if NEEDS_MASK:
        dim_mask = dims < HEAD_DIM // 2
        cos = tl.load(cos_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        cos = tl.load(cos_ptr + dims).to(tl.float32)
        sin = tl.load(sin_ptr + dims).to(tl.float32)

    q_ptr += s_id * q_stride_s
    k_ptr += s_id * k_stride_s
    _rotary_head_tiles(
        q_ptr, q_ptr, cos, sin,
        q_stride_h, q_stride_d, q_stride_h, q_stride_d,
        NUM_Q_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
        BLOCK_H_Q, NEEDS_MASK,
    )
    _rotary_head_tiles(
        k_ptr, k_ptr, cos, sin,
        k_stride_h, k_stride_d, k_stride_h, k_stride_d,
        NUM_K_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
        BLOCK_H_K, NEEDS_MASK,
    )


def _select_block_h(num_heads: int) -> int:
    """改动1：头数很小（如 MiniCPM5-2B 的 2 个 KV 头）时，
    直接用实际头数做 tile 宽度，不再套用固定的 BLOCK_H=8，
    避免 8 条 lane 里有 6 条被 mask 掉、白白浪费算力和带宽。
    头数较多（如 16 个 Q 头）时仍用 8，保持原有的分块粒度。"""
    if num_heads <= 2:
        return num_heads
    return 8


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    """
    Apply rotary position embedding to q and k

    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way

    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    logger.debug("GEMS ROTARY_POS_EMBEDDING")
    assert (
        k.shape[-1] == q.shape[-1]
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape
    k_heads = k.shape[-2]

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    # 改动1：Q、K 按各自头数分别选 BLOCK_H。
    # MiniCPM5-2B 上 q_heads=16（沿用 8）、k_heads=2（改为 2），两者不再共用一个常量。
    block_h_q = _select_block_h(q_heads)
    block_h_k = _select_block_h(k_heads)

    # 改动4：head_dim 本身就是 2 的幂（MiniCPM5-2B 的 head_dim=128）时，
    # padded_head_dim == head_dim，dim_mask 恒为 True，是多余的 predicated load/store。
    # 这种情况下关掉 mask 分支，走无条件 load/store。
    needs_mask = padded_head_dim != head_dim

    if inplace:
        grid = (n_tokens,)
        with torch_device_fn.device(q.device):
            apply_rotary_pos_emb_inplace_kernel[grid](
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                position_ids.stride(0) if position_ids is not None else 0,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                q.shape[-2],
                k.shape[-2],
                head_dim,
                padded_head_dim,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
                BLOCK_H_Q=block_h_q,
                BLOCK_H_K=block_h_k,
                NEEDS_MASK=needs_mask,
            )
        return q.view(q_shape), k.view(k_shape)
    # If not inplace, we need to create new tensors for q_embed and k_embed
    else:
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)

        grid = (n_tokens,)
        with torch_device_fn.device(q_embed.device):
            apply_rotary_pos_emb_kernel[grid](
                q_embed,
                k_embed,
                q,
                k,
                cos,
                sin,
                position_ids,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                q_embed.stride(0),
                q_embed.stride(1),
                q_embed.stride(2),
                k_embed.stride(0),
                k_embed.stride(1),
                k_embed.stride(2),
                position_ids.stride(0) if position_ids is not None else 0,
                cos.stride(0),
                sin.stride(0),
                seq_len,
                q.shape[-2],
                k.shape[-2],
                head_dim,
                padded_head_dim,
                rotary_interleaved,
                MAX_POSITION_EMBEDDINGS=cos.shape[0],
                BLOCK_H_Q=block_h_q,
                BLOCK_H_K=block_h_k,
                NEEDS_MASK=needs_mask,
            )
        q_embed = q_embed.view(q_shape)
        k_embed = k_embed.view(k_shape)
        return q_embed, k_embed


# import logging
# from typing import Optional

# import torch
# import triton
# import triton.language as tl

# from flag_gems.runtime import torch_device_fn
# from flag_gems.utils import libentry
# from flag_gems.utils import triton_lang_extension as ext

# logger = logging.getLogger(__name__)


# @triton.jit
# def _rotary_head_tiles(
#     out_ptr, in_ptr, cos_ptr, sin_ptr,
#     in_stride_h, in_stride_d, out_stride_h, out_stride_d,
#     NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
#     PADDED_HEAD_DIM: tl.constexpr, ROTARY_INTERLEAVED: tl.constexpr,
# ):
#     BLOCK_H: tl.constexpr = 8
#     dims = tl.arange(0, PADDED_HEAD_DIM // 2)
#     dim_mask = dims < HEAD_DIM // 2
#     cos = tl.load(cos_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
#     sin = tl.load(sin_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
#     if ROTARY_INTERLEAVED:
#         first = 2 * dims
#         second = first + 1
#     else:
#         first = dims
#         second = dims + HEAD_DIM // 2
#     for head_start in range(0, NUM_HEADS, BLOCK_H):
#         heads = head_start + tl.arange(0, BLOCK_H)
#         mask = (heads[:, None] < NUM_HEADS) & dim_mask[None, :]
#         in_base = in_ptr + heads[:, None] * in_stride_h
#         out_base = out_ptr + heads[:, None] * out_stride_h
#         low = tl.load(in_base + first[None, :] * in_stride_d, mask=mask, other=0.0).to(tl.float32)
#         high = tl.load(in_base + second[None, :] * in_stride_d, mask=mask, other=0.0).to(tl.float32)
#         out_low = low * cos[None, :] - high * sin[None, :]
#         out_high = high * cos[None, :] + low * sin[None, :]
#         tl.store(out_base + first[None, :] * out_stride_d, out_low, mask=mask)
#         tl.store(out_base + second[None, :] * out_stride_d, out_high, mask=mask)


# @libentry()
# @triton.jit
# def apply_rotary_pos_emb_kernel(
#     oq_ptr,
#     ok_ptr,
#     q_ptr,  # (n_tokens, q_heads, head_dim)
#     k_ptr,  # (n_tokens, k_heads, head_dim)
#     cos_ptr,  # (max_seq_len, dim // 2)
#     sin_ptr,  # (max_seq_len, dim // 2)
#     pos_ptr,  # (n_tokens, )
#     q_stride_s,
#     q_stride_h,
#     q_stride_d,
#     k_stride_s,
#     k_stride_h,
#     k_stride_d,
#     oq_stride_s,
#     oq_stride_h,
#     oq_stride_d,
#     ok_stride_s,
#     ok_stride_h,
#     ok_stride_d,
#     p_stride_s,
#     cos_stride_s,
#     sin_stride_s,
#     seq_len,
#     NUM_Q_HEADS: tl.constexpr,
#     NUM_K_HEADS: tl.constexpr,
#     HEAD_DIM: tl.constexpr,
#     PADDED_HEAD_DIM: tl.constexpr,
#     ROTARY_INTERLEAVED: tl.constexpr,
#     MAX_POSITION_EMBEDDINGS: tl.constexpr,
# ):
#     s_id = ext.program_id(0)

#     if pos_ptr is None:
#         pos_id = s_id % seq_len
#     else:
#         pos_ptr += s_id * p_stride_s
#         pos_id = tl.load(pos_ptr)
#     cos_ptr += pos_id * cos_stride_s
#     sin_ptr += pos_id * sin_stride_s

#     # note: set TRITON_DEBUG=1 to enable this check
#     tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

#     _rotary_head_tiles(
#         oq_ptr + s_id * oq_stride_s, q_ptr + s_id * q_stride_s,
#         cos_ptr, sin_ptr, q_stride_h, q_stride_d, oq_stride_h, oq_stride_d,
#         NUM_Q_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
#     )
#     _rotary_head_tiles(
#         ok_ptr + s_id * ok_stride_s, k_ptr + s_id * k_stride_s,
#         cos_ptr, sin_ptr, k_stride_h, k_stride_d, ok_stride_h, ok_stride_d,
#         NUM_K_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
#     )


# @libentry()
# @triton.jit
# def apply_rotary_pos_emb_inplace_kernel(
#     q_ptr,  # (n_tokens, q_heads, head_dim)
#     k_ptr,  # (n_tokens, k_heads, head_dim)
#     cos_ptr,  # (max_seq_len, dim // 2)
#     sin_ptr,  # (max_seq_len, dim // 2)
#     pos_ptr,  # (n_tokens, )
#     q_stride_s,
#     q_stride_h,
#     q_stride_d,
#     k_stride_s,
#     k_stride_h,
#     k_stride_d,
#     p_stride_s,
#     cos_stride_s,
#     sin_stride_s,
#     seq_len,
#     NUM_Q_HEADS: tl.constexpr,
#     NUM_K_HEADS: tl.constexpr,
#     HEAD_DIM: tl.constexpr,
#     PADDED_HEAD_DIM: tl.constexpr,
#     ROTARY_INTERLEAVED: tl.constexpr,
#     MAX_POSITION_EMBEDDINGS: tl.constexpr,
# ):
#     s_id = ext.program_id(0)

#     if pos_ptr is None:
#         pos_id = s_id % seq_len
#     else:
#         pos_ptr += s_id * p_stride_s
#         pos_id = tl.load(pos_ptr)
#     cos_ptr += pos_id * cos_stride_s
#     sin_ptr += pos_id * sin_stride_s

#     # note: set TRITON_DEBUG=1 to enable this check
#     tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

#     q_ptr += s_id * q_stride_s
#     k_ptr += s_id * k_stride_s
#     _rotary_head_tiles(
#         q_ptr, q_ptr, cos_ptr, sin_ptr,
#         q_stride_h, q_stride_d, q_stride_h, q_stride_d,
#         NUM_Q_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
#     )
#     _rotary_head_tiles(
#         k_ptr, k_ptr, cos_ptr, sin_ptr,
#         k_stride_h, k_stride_d, k_stride_h, k_stride_d,
#         NUM_K_HEADS, HEAD_DIM, PADDED_HEAD_DIM, ROTARY_INTERLEAVED,
#     )

# def apply_rotary_pos_emb(
#     q,
#     k,
#     cos,
#     sin,
#     position_ids: Optional[torch.IntTensor] = None,
#     rotary_interleaved: bool = False,
#     inplace: bool = False,
# ):
#     """
#     Apply rotary position embedding to q and k

#     Args:
#         q: (*, q_heads, head_dim)
#         k: (*, k_heads, head_dim)
#         cos: (max_seq_len, head_dim // 2)
#         sin: (max_seq_len, head_dim // 2)
#         position_ids: (*, ), optional, position ids for each token
#         rotary_interleaved: whether the head_dim is rotated in an interleaved way

#     Returns:
#         q_embed: (*, q_heads, head_dim)
#         k_embed: (*, k_heads, head_dim)
#     """
#     logger.debug("GEMS ROTARY_POS_EMBEDDING")
#     assert (
#         k.shape[-1] == q.shape[-1]
#     ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
#     assert (
#         cos.shape[-1] == sin.shape[-1]
#     ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
#     assert (
#         cos.shape[-1] * 2 == q.shape[-1]
#     ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
#     assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
#     assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

#     q_shape = q.shape
#     k_shape = k.shape

#     assert (
#         q.shape[:-2] == k.shape[:-2]
#     ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
#     if position_ids is None:
#         assert (
#             len(q.shape) == 4
#         ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
#         seq_len = q.shape[-3]
#     else:
#         assert (
#             position_ids.shape == q.shape[:-2]
#         ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

#         position_ids = position_ids.view(-1)
#         seq_len = None

#     q = q.view(-1, q.shape[-2], q.shape[-1])
#     k = k.view(-1, k.shape[-2], k.shape[-1])

#     n_tokens, q_heads, head_dim = q.shape

#     # The block size must be the next power of two, sometimes we need to pad it.
#     padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

#     if inplace:
#         grid = (n_tokens,)
#         with torch_device_fn.device(q.device):
#             apply_rotary_pos_emb_inplace_kernel[grid](
#                 q,
#                 k,
#                 cos,
#                 sin,
#                 position_ids,
#                 q.stride(0),
#                 q.stride(1),
#                 q.stride(2),
#                 k.stride(0),
#                 k.stride(1),
#                 k.stride(2),
#                 position_ids.stride(0) if position_ids is not None else 0,
#                 cos.stride(0),
#                 sin.stride(0),
#                 seq_len,
#                 q.shape[-2],
#                 k.shape[-2],
#                 head_dim,
#                 padded_head_dim,
#                 rotary_interleaved,
#                 MAX_POSITION_EMBEDDINGS=cos.shape[0],
#             )
#         return q.view(q_shape), k.view(k_shape)
#     # If not inplace, we need to create new tensors for q_embed and k_embed
#     else:
#         q_embed = torch.empty_like(q)
#         k_embed = torch.empty_like(k)

#         grid = (n_tokens,)
#         with torch_device_fn.device(q_embed.device):
#             apply_rotary_pos_emb_kernel[grid](
#                 q_embed,
#                 k_embed,
#                 q,
#                 k,
#                 cos,
#                 sin,
#                 position_ids,
#                 q.stride(0),
#                 q.stride(1),
#                 q.stride(2),
#                 k.stride(0),
#                 k.stride(1),
#                 k.stride(2),
#                 q_embed.stride(0),
#                 q_embed.stride(1),
#                 q_embed.stride(2),
#                 k_embed.stride(0),
#                 k_embed.stride(1),
#                 k_embed.stride(2),
#                 position_ids.stride(0) if position_ids is not None else 0,
#                 cos.stride(0),
#                 sin.stride(0),
#                 seq_len,
#                 q.shape[-2],
#                 k.shape[-2],
#                 head_dim,
#                 padded_head_dim,
#                 rotary_interleaved,
#                 MAX_POSITION_EMBEDDINGS=cos.shape[0],
#             )
#         q_embed = q_embed.view(q_shape)
#         k_embed = k_embed.view(k_shape)
#         return q_embed, k_embed
