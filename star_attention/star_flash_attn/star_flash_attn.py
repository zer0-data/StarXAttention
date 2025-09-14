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

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward

from .utils import update_out_and_lse, calculate_heuristic_score
from ..anchor_registry import get_anchor_summary

def _star_flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax, stride: int = 16, layer_idx: int = -1
):
    # Step 1: Process each block locally to get its attention output A(h).
    # The LSE is no longer needed for the new aggregation logic.
    block_out, _, _, _, _, _, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
    )

    # Step 2: Simultaneously, calculate the predictive importance score for the block.
    # Use (anchor summary + current) for scoring if available.
    q_permuted = q.permute(0, 2, 1, 3)
    k_permuted = k.permute(0, 2, 1, 3)
    # Fetch per-layer anchor summary (rank 0 sets it during Phase 1)
    anchor_summary = get_anchor_summary(layer_idx)
    # Broadcast presence flag from rank 0
    present_flag = torch.tensor([1 if (anchor_summary is not None and dist.get_rank() == 0) else 0], device=k_permuted.device, dtype=torch.int64)
    dist.broadcast(present_flag, src=0)
    has_anchor = int(present_flag.item()) == 1
    if has_anchor:
        # Prepare buffer and broadcast anchor summary tensor (B, Hk, 1, Dh)
        B, Hk, Tk_local, Dh = k_permuted.shape
        anchor_buf = torch.empty((B, Hk, 1, Dh), device=k_permuted.device, dtype=k_permuted.dtype)
        if dist.get_rank() == 0:
            if anchor_summary.device != k_permuted.device:
                anchor_summary = anchor_summary.to(k_permuted.device)
            anchor_buf.copy_(anchor_summary)
        dist.broadcast(anchor_buf, src=0)
        # Concatenate along sequence dimension and score
        k_for_score = torch.cat((anchor_buf, k_permuted), dim=2)
        # Use Q (Hq heads) vs (anchor|K) (Hk heads); the scorer aligns GQA to return (B, Hq)
        raw_score = calculate_heuristic_score(q_permuted, k_for_score, softmax_scale, stride)
    else:
        # Fallback: score using current block only
        raw_score = calculate_heuristic_score(q_permuted, k_permuted, softmax_scale, stride)

    # Step 3: Collect the raw scores and local attention outputs from all blocks (processes).
    out_gather = [torch.zeros_like(block_out) for _ in range(dist.get_world_size())]
    score_gather = [torch.zeros_like(raw_score) for _ in range(dist.get_world_size())]
    dist.all_gather(out_gather, block_out, async_op=False)
    dist.all_gather(score_gather, raw_score, async_op=False)
    dist.barrier()

    # Step 4: Normalize scores and compute the final global attention output.
    # Stack outputs and scores from all ranks.
    # Shape: (world_size, batch_size, seqlen, nheads, headdim)
    all_outputs = torch.stack(out_gather, dim=0)
    # Shape: (world_size, batch_size, nheads)
    all_scores = torch.stack(score_gather, dim=0)

    # Normalize the collected scores across all blocks (dim=0) to create weights w(h).
    # We apply softmax per-head, per-batch-item across all ranks.
    # Permute scores to (batch_size, nheads, world_size) for softmax.
    scores_to_normalize = all_scores.permute(1, 2, 0)
    weights = torch.nn.functional.softmax(scores_to_normalize, dim=-1)

    # Reshape weights for broadcasting during the weighted sum.
    # Final shape: (world_size, batch_size, 1, nheads, 1)
    weights_reshaped = weights.permute(2, 0, 1).unsqueeze(2).unsqueeze(4)

    # Compute the global attention output: Aglobal = sum(w(h) * A(h))
    out = torch.sum(all_outputs * weights_reshaped, dim=0)

    # Finalize output and return a dummy LSE to maintain function signature.
    out = out.to(q.dtype)
    # softmax_lse expected shape: (batch_size, nheads, seqlen)
    B, T, H, _ = q.shape
    softmax_lse = torch.zeros(B, H, T, device=q.device, dtype=torch.float32)
    return out, softmax_lse


class StarFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_softmax,
        layer_idx,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = _star_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            layer_idx=layer_idx,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        return out if not return_softmax else (out, softmax_lse, None)


def star_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    return_attn_probs=False,
    layer_idx: int = -1,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return StarFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        layer_idx,
    )
