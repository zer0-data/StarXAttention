# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch.nn.functional as F
# ...existing code...

import torch

__all__ = ["update_out_and_lse"]


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()

@torch.jit.script
def calculate_heuristic_score(
    q: torch.Tensor, k: torch.Tensor, softmax_scale: float, stride: int = 1
) -> torch.Tensor:
    """
    Calculates an efficient, approximate heuristic score using strided sampling,
    inspired by the XAttention paper.
    q: (batch_size, nheads, seqlen_q, headdim)
    k: (batch_size, nheads_k, seqlen_k, headdim)
    """
    # Support grouped-query attention (GQA): align K heads to match Q heads
    Hq = int(q.size(1))
    Hk = int(k.size(1))
    if Hq != Hk:
        if Hq % Hk == 0:
            repeats = Hq // Hk
            k = k.repeat(1, repeats, 1, 1)
        elif Hk % Hq == 0:
            groups = Hk // Hq
            B = int(k.size(0))
            Tk = int(k.size(2))
            Dh = int(k.size(3))
            k = k.reshape(B, Hq, groups, Tk, Dh).mean(dim=2)
        else:
            raise RuntimeError(
                "calculate_heuristic_score: head mismatch not divisible (Hq="
                + str(Hq)
                + ", Hk="
                + str(Hk)
                + ")"
            )

    # Ensure Q and K use the same sequence length for strided sampling
    Tq = int(q.size(2))
    Tk = int(k.size(2))
    if Tq != Tk:
        T = Tq if Tq < Tk else Tk
        q = q[:, :, :T, :]
        k = k[:, :, :T, :]
        Tq = T

    # Guard stride
    if stride < 1:
        stride = 1

    B = int(q.size(0))
    H = int(q.size(1))
    Dh = int(q.size(3))
    if Dh != int(k.size(3)):
        raise RuntimeError(
            f"calculate_heuristic_score: q and k must have same head_dim; got Dh_q={Dh}, Dh_k={int(k.size(3))}"
        )

    # Make seqlen divisible by stride via right padding (constant 0)
    if Tq % stride != 0:
        pad_len = stride - (Tq % stride)
        # Pad along sequence length dimension (dim=2)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        Tq = int(q.size(2))

    # Reshape for strided approximate scoring
    # (B, H, T, Dh) -> (B, H, T/stride, stride, Dh)
    q_reshaped = q.reshape(B, H, Tq // stride, stride, Dh)
    k_reshaped = k.reshape(B, H, Tq // stride, stride, Dh)

    # Group strided elements and compute compact SxS score per head (S=stride)
    # (B, H, stride, (T/stride*Dh))
    q_final = q_reshaped.permute(0, 1, 3, 2, 4).reshape(B, H, stride, -1)
    k_final = k_reshaped.permute(0, 1, 3, 2, 4).reshape(B, H, stride, -1)

    approx_scores = torch.matmul(q_final, k_final.transpose(-2, -1)) * softmax_scale

    # Antidiagonal selection with stride S: take the antidiagonal of each compact SxS block
    # and normalize by S (mean) to get a robust per-head score.
    # TorchScript-friendly antidiagonal: flip the last dim then take the main diagonal.
    antidiag = approx_scores.flip(-1).diagonal(0, -2, -1)  # shape: (B, H, stride)
    # Sum across the antidiagonal entries; normalization across blocks is handled by softmax later
    score = antidiag.sum(dim=-1)
    return score
