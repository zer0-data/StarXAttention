# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

import torch

# Global registry for per-layer anchor summaries.
# Key: layer index (int), Value: Tensor of shape (B, Hk, 1, Dh)
_ANCHOR_SUMMARIES: Dict[int, torch.Tensor] = {}


def set_anchor_summary(layer_idx: int, summary: torch.Tensor) -> None:
    """Register the anchor summary for a given layer.

    summary shape: (B, Hk, 1, Dh)
    """
    _ANCHOR_SUMMARIES[layer_idx] = summary


def get_anchor_summary(layer_idx: int) -> Optional[torch.Tensor]:
    return _ANCHOR_SUMMARIES.get(layer_idx, None)


def clear_anchor_summaries() -> None:
    _ANCHOR_SUMMARIES.clear()
