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

import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    HEAD_SIZE_LIST = [64]
    BLOCK_SIZE_LIST = [8, 32]
    NUM_BLOCKS_LIST = [1024]
    DTYPE_LIST = [torch.float]
else:
    HEAD_SIZE_LIST = [64, 80, 120, 256]
    BLOCK_SIZE_LIST = [8, 16, 32]
    NUM_BLOCKS_LIST = [1024, 10000]
    DTYPE_LIST = [torch.half, torch.bfloat16, torch.float]


def create_kv_caches_with_random_flash(
    num_blocks,
    block_size,
    num_layers,
    num_heads,
    head_size,
    cache_dtype,
    model_dtype,
    seed,
    device,
):
    utils.init_seed(seed)
    torch_dtype = model_dtype
    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(
            size=key_value_cache_shape, dtype=torch_dtype, device=device
        )

        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Key cache type {cache_dtype} is not supported")

        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])

    return key_caches, value_caches


@pytest.mark.reshape_and_cache_flash
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", HEAD_SIZE_LIST)
@pytest.mark.parametrize("block_size", BLOCK_SIZE_LIST)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("seed", [2025])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@torch.inference_mode()
def test_reshape_and_cache_flash(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
) -> None:
    utils.init_seed(seed)

    with torch.device(device):
        # Create a random slot mapping.
        num_slots = block_size * num_blocks
        slot_mapping_lst = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)
        qkv = torch.randn(
            num_tokens, 3, num_heads, head_size, dtype=dtype, device=device
        )
        _, key, value = qkv.unbind(dim=1)

        # Create the KV caches.
        key_caches, value_caches = create_kv_caches_with_random_flash(
            num_blocks,
            block_size,
            1,
            num_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            seed=seed,
            device=device,
        )
        key_cache, value_cache = (
            key_caches[0].contiguous(),
            value_caches[0].contiguous(),
        )
        del key_caches
        del value_caches

        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)

        # Clone the KV caches.
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

        # Call the reshape_and_cache kernel.
        flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        # Run the reference implementation.
        block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_indicies_lst = block_indicies.cpu().tolist()
        block_offsets = slot_mapping % block_size
        block_offsets_lst = block_offsets.cpu().tolist()

        for i in range(num_tokens):
            block_idx = block_indicies_lst[i]
            block_offset = block_offsets_lst[i]
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]

        torch.testing.assert_close(key_cache.cpu(), cloned_key_cache.cpu())
        torch.testing.assert_close(value_cache.cpu(), cloned_value_cache.cpu())


@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 3])
@pytest.mark.parametrize("num_tokens,head_dim", [(5, 64), (511, 128), (512, 128)])
def test_fused_rope_and_cache_flash(rotary_interleaved, num_kv_heads, num_tokens, head_dim):
    q_heads, block_size = 7, 8
    dtype = torch.bfloat16
    q = torch.randn(num_tokens, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    original_q = q.clone()
    original_k = k.clone()
    original_v = v.clone()
    cos_sin = torch.randn(num_tokens + 2, head_dim, device=device, dtype=dtype)
    cos, sin = cos_sin.chunk(2, dim=-1)
    positions = torch.arange(num_tokens, device=device, dtype=torch.long) + 2
    slots = torch.arange(num_tokens, device=device, dtype=torch.long) * 3
    slots[2] = -1
    num_blocks = (num_tokens * 3 + block_size - 1) // block_size
    key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)

    if rotary_interleaved:
        pair0, pair1 = original_q[..., ::2].float(), original_q[..., 1::2].float()
        key0, key1 = original_k[..., ::2].float(), original_k[..., 1::2].float()
        cos_full = cos.repeat_interleave(2, dim=-1)[positions, None, :]
        sin_full = sin.repeat_interleave(2, dim=-1)[positions, None, :]
        q_ref = original_q.float() * cos_full + torch.stack((-pair1, pair0), -1).flatten(-2) * sin_full
        k_ref = original_k.float() * cos_full[:, :1] + torch.stack((-key1, key0), -1).flatten(-2) * sin_full[:, :1]
    else:
        cos_full = torch.cat((cos, cos), dim=-1)[positions, None, :]
        sin_full = torch.cat((sin, sin), dim=-1)[positions, None, :]
        q_rot = torch.cat((-original_q[..., head_dim // 2:], original_q[..., :head_dim // 2]), -1)
        k_rot = torch.cat((-original_k[..., head_dim // 2:], original_k[..., :head_dim // 2]), -1)
        q_ref = original_q.float() * cos_full + q_rot.float() * sin_full
        k_ref = original_k.float() * cos_full[:, :1] + k_rot.float() * sin_full[:, :1]

    flag_gems.fused_rope_and_cache_flash(
        q, k, v, key_cache, value_cache, cos, sin, positions, slots,
        rotary_interleaved=rotary_interleaved,
    )
    torch.testing.assert_close(q.float(), q_ref, atol=0.04, rtol=0.04)
    torch.testing.assert_close(k, original_k, atol=0, rtol=0)
    torch.testing.assert_close(v, original_v, atol=0, rtol=0)
    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        torch.testing.assert_close(key_cache[block, offset].float(), k_ref[token].float(), atol=0.04, rtol=0.04)
        torch.testing.assert_close(value_cache[block, offset], v[token])
