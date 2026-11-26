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

from flag_gems.config import use_c_extension
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

vendor_name = device.vendor_name
_IS_METAX = vendor_name == "metax"


@libentry()
@triton.jit
def fused_rope_and_cache_flash_kernel(
    q_ptr, k_ptr, v_ptr, key_cache_ptr, value_cache_ptr,
    cos_ptr, sin_ptr, pos_ptr, slot_mapping_ptr,
    q_stride_s, q_stride_h, q_stride_d,
    k_stride_s, k_stride_h, k_stride_d,
    v_stride_s, v_stride_h, v_stride_d,
    p_stride_s, cos_stride_s, sin_stride_s, cache_block_stride,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    PADDED_HALF_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = ext.program_id(0).to(tl.int64)
    if pos_ptr is None:
        pos = token % seq_len
    else:
        pos = tl.load(pos_ptr + token * p_stride_s).to(tl.int64)

    dims = tl.arange(0, PADDED_HALF_DIM)
    dim_mask = dims < HALF_HEAD_DIM
    cos = tl.load(cos_ptr + pos * cos_stride_s + dims, dim_mask, other=0).to(tl.float32)
    sin = tl.load(sin_ptr + pos * sin_stride_s + dims, dim_mask, other=0).to(tl.float32)
    if ROTARY_INTERLEAVED:
        d0, d1 = 2 * dims, 2 * dims + 1
    else:
        d0, d1 = dims, dims + HALF_HEAD_DIM

    heads = tl.arange(0, BLOCK_H)[:, None]
    d0q = (d0 * q_stride_d)[None, :]
    d1q = (d1 * q_stride_d)[None, :]
    q_base = q_ptr + token * q_stride_s
    for hbase in range(0, NUM_Q_HEADS, BLOCK_H):
        h = hbase + heads
        mask = (h < NUM_Q_HEADS) & dim_mask[None, :]
        ptr = q_base + h * q_stride_h
        x0 = tl.load(ptr + d0q, mask, other=0).to(tl.float32)
        x1 = tl.load(ptr + d1q, mask, other=0).to(tl.float32)
        tl.store(ptr + d0q, x0 * cos[None, :] - x1 * sin[None, :], mask)
        tl.store(ptr + d1q, x1 * cos[None, :] + x0 * sin[None, :], mask)

    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    if slot < 0:
        return
    block = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE
    head_dim: tl.constexpr = HALF_HEAD_DIM * 2
    cache_token = block * cache_block_stride + offset * NUM_KV_HEADS * head_dim
    d0k = (d0 * k_stride_d)[None, :]
    d1k = (d1 * k_stride_d)[None, :]
    d0v = (d0 * v_stride_d)[None, :]
    d1v = (d1 * v_stride_d)[None, :]

    if NUM_KV_HEADS * 2 <= BLOCK_H:
        is_key = heads < NUM_KV_HEADS
        kv_head = tl.where(is_key, heads, heads - NUM_KV_HEADS)
        mask = (heads < NUM_KV_HEADS * 2) & dim_mask[None, :]
        src = tl.where(
            is_key,
            k_ptr + token * k_stride_s + kv_head * k_stride_h,
            v_ptr + token * v_stride_s + kv_head * v_stride_h,
        )
        stride_d = tl.where(is_key, k_stride_d, v_stride_d)
        x0 = tl.load(src + d0[None, :] * stride_d, mask, other=0).to(tl.float32)
        x1 = tl.load(src + d1[None, :] * stride_d, mask, other=0).to(tl.float32)
        y0 = tl.where(is_key, x0 * cos[None, :] - x1 * sin[None, :], x0)
        y1 = tl.where(is_key, x1 * cos[None, :] + x0 * sin[None, :], x1)
        dst = tl.where(is_key, key_cache_ptr, value_cache_ptr) + cache_token + kv_head * head_dim
        tl.store(dst + d0[None, :], y0, mask)
        tl.store(dst + d1[None, :], y1, mask)
    else:
        for hbase in range(0, NUM_KV_HEADS, BLOCK_H):
            h = hbase + heads
            mask = (h < NUM_KV_HEADS) & dim_mask[None, :]
            kp = k_ptr + token * k_stride_s + h * k_stride_h
            vp = v_ptr + token * v_stride_s + h * v_stride_h
            k0 = tl.load(kp + d0k, mask, other=0).to(tl.float32)
            k1 = tl.load(kp + d1k, mask, other=0).to(tl.float32)
            v0 = tl.load(vp + d0v, mask, other=0)
            v1 = tl.load(vp + d1v, mask, other=0)
            dst = cache_token + h * head_dim
            tl.store(key_cache_ptr + dst + d0[None, :], k0 * cos[None, :] - k1 * sin[None, :], mask)
            tl.store(key_cache_ptr + dst + d1[None, :], k1 * cos[None, :] + k0 * sin[None, :], mask)
            tl.store(value_cache_ptr + dst + d0[None, :], v0, mask)
            tl.store(value_cache_ptr + dst + d1[None, :], v1, mask)


@triton.jit(do_not_specialize=["q_stride_s", "k_stride_s", "v_stride_s", "cache_block_stride"])
def _fast_fused_rope_cache_kernel(
    q_ptr, k_ptr, v_ptr, key_cache_ptr, value_cache_ptr, cos_ptr, sin_ptr,
    pos_ptr, slot_mapping_ptr, q_stride_s, k_stride_s, v_stride_s,
    cache_block_stride, cos_stride_s, sin_stride_s,
    BLOCK_SIZE: tl.constexpr, NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, HALF_HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + token).to(tl.int64)
    dims = tl.arange(0, HALF_HEAD_DIM)
    cos = tl.load(cos_ptr + pos * cos_stride_s + dims).to(tl.float32)[None, :]
    sin = tl.load(sin_ptr + pos * sin_stride_s + dims).to(tl.float32)[None, :]
    heads = tl.arange(0, BLOCK_H)[:, None]
    d0, d1 = dims[None, :], dims[None, :] + HALF_HEAD_DIM
    head_dim: tl.constexpr = HALF_HEAD_DIM * 2
    qbase = q_ptr + token * q_stride_s
    for hbase in range(0, NUM_Q_HEADS, BLOCK_H):
        h = hbase + heads
        mask = h < NUM_Q_HEADS
        ptr = qbase + h * head_dim
        x0 = tl.load(ptr + d0, mask, other=0).to(tl.float32)
        x1 = tl.load(ptr + d1, mask, other=0).to(tl.float32)
        tl.store(ptr + d0, x0 * cos - x1 * sin, mask)
        tl.store(ptr + d1, x1 * cos + x0 * sin, mask)
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    if slot < 0:
        return
    cache_token = (slot // BLOCK_SIZE) * cache_block_stride + (slot % BLOCK_SIZE) * NUM_KV_HEADS * head_dim
    is_key = heads < NUM_KV_HEADS
    kv_head = tl.where(is_key, heads, heads - NUM_KV_HEADS)
    mask = heads < NUM_KV_HEADS * 2
    src = tl.where(is_key, k_ptr + token * k_stride_s + kv_head * head_dim,
                   v_ptr + token * v_stride_s + kv_head * head_dim)
    x0 = tl.load(src + d0, mask, other=0).to(tl.float32)
    x1 = tl.load(src + d1, mask, other=0).to(tl.float32)
    y0 = tl.where(is_key, x0 * cos - x1 * sin, x0)
    y1 = tl.where(is_key, x1 * cos + x0 * sin, x1)
    dst = tl.where(is_key, key_cache_ptr, value_cache_ptr) + cache_token + kv_head * head_dim
    tl.store(dst + d0, y0, mask)
    tl.store(dst + d1, y1, mask)


@libentry()
@triton.jit
def reshape_and_cache_flash_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    k_scale,
    v_scale,
    n: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    i = tl.arange(0, triton.next_power_of_2(n))
    mask = i < n

    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i
    head_idx = i // head_size
    head_offset = i % head_size
    tgt_key_value_idx = (
        block_idx * block_stride
        + block_offset * num_heads * head_size
        + head_idx * head_size
        + head_offset
    )

    tgt_key = tl.load(key + src_key_idx, mask=mask)
    tgt_value = tl.load(value + src_value_idx, mask=mask)

    # TODO: support fp8 dtype
    tl.store(key_cache + tgt_key_value_idx, tgt_key, mask=mask)
    tl.store(value_cache + tgt_key_value_idx, tgt_value, mask=mask)


def reshape_and_cache_flash(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    if use_c_extension:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH(C EXTENSION)")
        torch.ops.flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH")
        # GCU300 rejects 64-bit IR; slot_mapping arrives as int64 and its
        # index arithmetic would emit 64-bit ops. Downcast to int32 on enflame
        # (mirrors the enflame concat_and_cache_mla override). Vendor+dtype
        # guarded so no other backend is affected.
        if vendor_name == "enflame" and slot_mapping.dtype == torch.int64:
            slot_mapping = slot_mapping.to(torch.int32)
        num_tokens = slot_mapping.size(0)
        num_heads = key.size(1)
        head_size = key.size(2)
        block_size = key_cache.size(1)

        key_stride = key.stride(0)
        value_stride = value.stride(0)
        block_stride = key_cache.stride(0)
        assert key_cache.stride(0) == value_cache.stride(0)

        grid = (num_tokens,)
        with torch_device_fn.device(key.device):
            reshape_and_cache_flash_kernel[grid](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                block_stride,
                key_stride,
                value_stride,
                num_heads,
                head_size,
                block_size,
                k_scale,
                v_scale,
                num_heads * head_size,
            )


def fused_rope_and_cache_flash(
    q, k, v, key_cache, value_cache, cos, sin, position_ids, slot_mapping,
    rotary_interleaved=False,
):
    """Apply RoPE to Q in place and write rotated K plus V to paged caches."""
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    seq_len = q.shape[-3] if position_ids is None and q.ndim == 4 else num_tokens
    head_dim = q.shape[-1]
    assert head_dim == k.shape[-1] == v.shape[-1]
    assert cos.shape[-1] * 2 == head_dim and sin.shape[-1] == cos.shape[-1]
    q = q.reshape(num_tokens, -1, head_dim)
    k = k.reshape(num_tokens, -1, head_dim)
    v = v.reshape(num_tokens, -1, head_dim)
    positions = position_ids.reshape(-1) if position_ids is not None else None
    q_heads, kv_heads = q.shape[1], k.shape[1]
    half_dim = head_dim // 2
    block_size = key_cache.shape[1]
    block_stride = key_cache.stride(0)
    block_h = 4
    grid = (num_tokens,)

    fast = (
        _IS_METAX and positions is not None and not rotary_interleaved
        and half_dim >= 64 and half_dim & (half_dim - 1) == 0
        and kv_heads * 2 <= block_h
        and q.stride(2) == k.stride(2) == v.stride(2) == 1
        and q.stride(1) == k.stride(1) == v.stride(1) == head_dim
        and key_cache.is_contiguous() and value_cache.is_contiguous()
        and cos.stride(-1) == sin.stride(-1) == 1
        and positions.is_contiguous() and slot_mapping.is_contiguous()
        and q.is_cuda
    )
    num_warps = (2 if num_tokens >= 512 else 4) if fast else 4
    with torch_device_fn.device(q.device):
        if fast:
            _fast_fused_rope_cache_kernel[grid](
                q, k, v, key_cache, value_cache, cos, sin, positions,
                slot_mapping, q.stride(0), k.stride(0), v.stride(0),
                block_stride, cos.stride(0), sin.stride(0),
                BLOCK_SIZE=block_size, NUM_Q_HEADS=q_heads,
                NUM_KV_HEADS=kv_heads, HALF_HEAD_DIM=half_dim,
                BLOCK_H=block_h, num_warps=num_warps,
            )
        else:
            padded_half_dim = max(triton.next_power_of_2(half_dim), 16)
            fused_rope_and_cache_flash_kernel[grid](
                q, k, v, key_cache, value_cache, cos, sin, positions,
                slot_mapping,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                positions.stride(0) if positions is not None else 0,
                cos.stride(0), sin.stride(0), block_stride,
                seq_len,
                BLOCK_SIZE=block_size, NUM_Q_HEADS=q_heads,
                NUM_KV_HEADS=kv_heads, HALF_HEAD_DIM=half_dim,
                PADDED_HALF_DIM=padded_half_dim,
                ROTARY_INTERLEAVED=rotary_interleaved, BLOCK_H=block_h,
                num_warps=num_warps,
            )
