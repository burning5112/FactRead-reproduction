from __future__ import annotations
from contextlib import nullcontext
import os
import torch

def profile_scope(name: str):
    if os.environ.get('FACTREAD_PROFILE_COMPONENTS') == '1':
        return torch.profiler.record_function(name)
    return nullcontext()
