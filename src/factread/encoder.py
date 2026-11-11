from __future__ import annotations
import math
from collections.abc import Iterable
import torch
from torch import nn

QUERY_LAYERS = tuple(range(0, 31, 3))

class KnowledgeEncoder(nn.Module):

    def __init__(self, output_dim: int):
        super().__init__()
        self.projector_k = nn.Linear(384, output_dim, dtype=torch.float32)
        self.projector_v = nn.Linear(384, output_dim, dtype=torch.float32)
        self.key_layernorm = nn.LayerNorm(output_dim, elementwise_affine=False, bias=False)

    def load_parent(self, state: dict[str, torch.Tensor]) -> None:
        projection_state = {key: value for (key, value) in state.items() if key.startswith(('projector_k.', 'projector_v.'))}
        (missing, unexpected) = self.load_state_dict(projection_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f'A0 projection state mismatch: {missing}/{unexpected}')

    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected_keys = self.key_layernorm(self.projector_k(keys.float())).to(torch.bfloat16)
        projected_values = self.projector_v(values.float()).to(torch.bfloat16)
        return (projected_keys, projected_values)

class QueryLoRA(nn.Module):

    def __init__(self, hidden_size: int, rank: int=8, alpha: float=8.0):
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()
        self._handles = []
        self._merged = False
        for layer in QUERY_LAYERS:
            a = nn.Parameter(torch.empty(rank, hidden_size, dtype=torch.float32))
            b = nn.Parameter(torch.zeros(hidden_size, rank, dtype=torch.float32))
            nn.init.normal_(a, mean=0.0, std=0.01)
            self.a[str(layer)] = a
            self.b[str(layer)] = b

    def attach(self, model) -> None:
        modules = {int(module.layer_idx): module for module in model.modules() if getattr(module, 'layer_idx', None) is not None and hasattr(module, 'q_proj_kg')}
        missing = sorted(set(QUERY_LAYERS) - modules.keys())
        if missing:
            raise ValueError(f'Missing q_proj_kg layers for LoRA: {missing}')
        for layer in QUERY_LAYERS:

            def add_delta(_module, inputs, output, layer_idx=layer):
                from factread.profile import profile_scope
                with profile_scope('factread/query_lora'):
                    x = inputs[0].float()
                    delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a[str(layer_idx)]), self.b[str(layer_idx)]) * self.scale
                return output + delta.to(output.dtype)
            self._handles.append(modules[layer].q_proj_kg.register_forward_hook(add_delta))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def merge_into(self, model) -> dict:
        if self._merged:
            raise RuntimeError('QueryLoRA has already been merged')
        modules = {int(module.layer_idx): module for module in model.modules() if getattr(module, 'layer_idx', None) is not None and hasattr(module, 'q_proj_kg')}
        missing = sorted(set(QUERY_LAYERS) - modules.keys())
        if missing:
            raise ValueError(f'Missing q_proj_kg layers for LoRA merge: {missing}')
        merged_layers = []
        with torch.autocast(device_type=next(self.parameters()).device.type, enabled=False):
            for layer in QUERY_LAYERS:
                projection = modules[layer].q_proj_kg
                if not hasattr(projection, 'weight'):
                    raise TypeError(f'Layer {layer} q_proj_kg has no dense weight to merge')
                delta = self.b[str(layer)].float() @ self.a[str(layer)].float()
                if delta.shape != projection.weight.shape:
                    raise ValueError(f'LoRA update shape {tuple(delta.shape)} does not match layer {layer} weight {tuple(projection.weight.shape)}')
                projection.weight.copy_((projection.weight.float() + self.scale * delta).to(projection.weight.dtype))
                merged_layers.append(layer)
        self.remove()
        self._merged = True
        return {'status': 'MERGED_INFERENCE_COPY', 'target': 'q_proj_kg only', 'layers': merged_layers, 'rank': self.rank, 'scale': self.scale, 'merge_dtype': 'fp32 accumulation, cast to projection weight dtype', 'checkpoint_modified': False}
