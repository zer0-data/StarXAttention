import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

from .utils import flatten_varlen_lse, unflatten_varlen_lse, update_out_and_lse, calculate_heuristic_score

def _starx_flash_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    block_table=None,
    leftpad_k=None,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # Step 1: Compute local attention output A(h)
    block_out, _, _, _, _, _, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
        block_table=block_table,
        leftpad_k=leftpad_k,
    )

    # Step 2: Calculate heuristic score for each sequence in the batch
    batch_size = len(cu_seqlens_q) - 1
    scores_list = []
    for i in range(batch_size):
        q_seq = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        k_seq = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
        
        # Reshape for score calculation: (seqlen, nheads, dim) -> (1, nheads, seqlen, dim)
        q_seq_reshaped = q_seq.permute(1, 0, 2).unsqueeze(0)
        k_seq_reshaped = k_seq.permute(1, 0, 2).unsqueeze(0)

        # Score shape: (1, nheads)
        seq_score = calculate_heuristic_score(q_seq_reshaped, k_seq_reshaped, softmax_scale)
        scores_list.append(seq_score)
    
    # Shape: (batch_size, nheads)
    raw_score = torch.cat(scores_list, dim=0)

    # Step 3: Gather scores and outputs from all ranks
    out_gather = [torch.zeros_like(block_out) for _ in range(dist.get_world_size())]
    score_gather = [torch.zeros_like(raw_score) for _ in range(dist.get_world_size())]
    dist.all_gather(out_gather, block_out, async_op=False)
    dist.all_gather(score_gather, raw_score, async_op=False)
    dist.barrier()

    # Step 4: Normalize scores and compute weighted sum
    # Shape: (world_size, total_tokens, nheads, headdim)
    all_outputs = torch.stack(out_gather, dim=0)
    # Shape: (world_size, batch_size, nheads)
    all_scores = torch.stack(score_gather, dim=0)
    
    # Normalize scores across ranks
    scores_to_normalize = all_scores.permute(1, 2, 0) # (batch_size, nheads, world_size)
    weights = torch.nn.functional.softmax(scores_to_normalize, dim=-1) # (batch_size, nheads, world_size)

    # Expand weights from per-sequence to per-token to match output shape
    weights_per_rank_list = []
    for r in range(dist.get_world_size()):
        weights_for_rank = weights[:, :, r] # (batch_size, nheads)
        weights_expanded = torch.repeat_interleave(
            weights_for_rank, cu_seqlens_q.diff(), dim=0
        ) # (total_tokens, nheads)
        weights_per_rank_list.append(weights_expanded)

    # Shape: (world_size, total_tokens, nheads)
    weights_per_token = torch.stack(weights_per_rank_list, dim=0)
    # Reshape for broadcasting: (world_size, total_tokens, nheads, 1)
    weights_reshaped = weights_per_token.unsqueeze(-1)
    
    # Compute weighted sum
    out = torch.sum(all_outputs * weights_reshaped, dim=0)

    # Finalize and return
    out = out.to(q.dtype)
    lse = unflatten_varlen_lse(torch.zeros(q.shape[0], q.shape[1], 1, device=q.device, dtype=torch.float32), cu_seqlens_q, max_seqlen_q)
    return out, lse


class StarXFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_softmax,
        block_table,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        out, softmax_lse = _starx_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        return out if not return_softmax else (out, softmax_lse, None)


def starx_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    return_attn_probs=False,
    block_table=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
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
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return StarXFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        block_table,
    )