from __future__ import annotations
import torch

def stage_read_mask(mode: str, query_length: int, device: torch.device, prompt_length: int | None=None, positions: torch.Tensor | None=None, decode_call_index: int | None=None):
    if mode not in {'all_on', 'prefill_only', 'warm4'}:
        raise ValueError(mode)
    if mode == 'all_on':
        return None
    if prompt_length is not None:
        if prompt_length < 0:
            raise ValueError('prompt_length must be nonnegative')
        if positions is None:
            positions = torch.arange(query_length, device=device)
        else:
            positions = positions.to(device=device, dtype=torch.long)
        threshold = prompt_length if mode == 'prefill_only' else prompt_length + 4
        return positions < threshold
    if query_length > 1:
        return None
    if mode == 'prefill_only':
        return False
    if decode_call_index is None:
        raise ValueError('warm4 generation requires a decode call index')
    return 1 <= decode_call_index <= 4
