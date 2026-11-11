from __future__ import annotations
import math
import torch
from torch import nn
from factread.adapter import ReadoutAdapter
from factread.integration import RecordCoupledKnowledgeRead
from factread.metric import orthonormal_columns
from factread.restrictions import project_to_shared_axes, rotate_coefficients

def balanced_coeff(raw: torch.Tensor, rho: float=0.75) -> torch.Tensor:
    raw = raw.float()
    return rho * raw * torch.rsqrt(1.0 + raw.square().sum(-1, keepdim=True))

def shared_axis_coeff(raw: torch.Tensor, theta: torch.Tensor, rho: float=0.75) -> torch.Tensor:
    raw = raw.float()
    strength = rho * torch.tanh((raw[..., 0] + raw[..., 1]) / math.sqrt(2.0))
    axis = torch.stack((torch.cos(2.0 * theta.float()), torch.sin(2.0 * theta.float())), -1)
    return strength[..., None] * axis

def directional_read_with_basis(masses: torch.Tensor, projected_values: torch.Tensor, base: torch.Tensor, coeff: torch.Tensor, basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    (gamma, values, y, ab, bmat) = (x.float() for x in (masses, projected_values, base, coeff, basis))
    from factread.profile import profile_scope
    with profile_scope('factread/vectorized_2x2_read'):
        z0 = torch.einsum('qhd,hds->qhs', y, bmat)
        (d0, d1) = (values - z0[:, None]).unbind(-1)
        (a, b) = ab.unbind(-1)
        ga = (gamma * a).sum(dim=1)
        gb = (gamma * b).sum(dim=1)
        t0 = (gamma * (a * d0 + b * d1)).sum(dim=1)
        t1 = (gamma * (b * d0 - a * d1)).sum(dim=1)
        det = 1.0 - ga.square() - gb.square()
        delta = torch.stack(((1.0 - ga) * t0 - gb * t1, (1.0 + ga) * t1 - gb * t0), dim=-1)
        delta = delta / det[..., None]
        update = torch.einsum('hds,qhs->qhd', bmat, delta)
    return (y + update, det)

class DirectionalCompiler(nn.Module):

    def __init__(self, mode: str, key_dim: int=384, heads: int=32, head_dim: int=128, rho: float=0.75):
        super().__init__()
        if mode not in {'balanced', 'shared_axis'}:
            raise ValueError(mode)
        self.mode = mode
        (self.heads, self.head_dim, self.size) = (heads, head_dim, 2)
        (self.rank, self.rho) = (heads * 2, float(rho))
        self.project = nn.Linear(key_dim, heads * 2, dtype=torch.float32)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.raw_basis = nn.Parameter(orthonormal_columns(torch.randn(heads, head_dim, 2)))
        if mode == 'shared_axis':
            self.theta = nn.Parameter(torch.zeros(heads, dtype=torch.float32))

    def initialize_basis_from_parent_value_weight(self, head_value_weight: torch.Tensor) -> None:
        with torch.no_grad():
            (u, _, _) = torch.linalg.svd(head_value_weight.float(), full_matrices=False)
            self.raw_basis.copy_(u[..., :2])

    def basis(self) -> torch.Tensor:
        return orthonormal_columns(self.raw_basis)

    def compile(self, raw_keys: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=raw_keys.device.type, enabled=False):
            keys = raw_keys.float()
            normalized = math.sqrt(keys.shape[-1]) * keys
            normalized = normalized / keys.norm(dim=-1, keepdim=True).clamp_min(1e-08)
            raw = self.project(normalized).reshape(*keys.shape[:-1], self.heads, 2)
            if self.mode == 'balanced':
                return balanced_coeff(raw, self.rho)
            return shared_axis_coeff(raw, self.theta, self.rho)

class DirectionalKnowledgeRead(RecordCoupledKnowledgeRead):
    is_directional_reader = True

    def __init__(self, mode: str, key_dim: int=384, heads: int=32, head_dim: int=128, selected_records: int=8, query_chunk: int=8, rho: float=0.75):
        super().__init__(mode='full', key_dim=key_dim, heads=heads, head_dim=head_dim, per_head_rank=2, selected_records=selected_records, query_chunk=query_chunk, solve_mode='joint')
        self.mode = mode
        self.reader = DirectionalCompiler(mode, key_dim, heads, head_dim, rho)
        self.fixed_shared_axes: torch.Tensor | None = None
        self.rotation_angles: torch.Tensor | None = None

    def set_diagnostic_axes(self, axes: torch.Tensor | None=None, angles: torch.Tensor | None=None) -> None:
        if axes is not None:
            if tuple(axes.shape) != (self.reader.heads, 2):
                raise ValueError(f'Expected [{self.reader.heads},2] shared axes, got {tuple(axes.shape)}')
            self.fixed_shared_axes = axes.detach().float()
        if angles is not None:
            if tuple(angles.shape) != (self.reader.heads,):
                raise ValueError(f'Expected [{self.reader.heads}] rotation angles, got {tuple(angles.shape)}')
            self.rotation_angles = angles.detach().float()

    def prepare_bank(self, raw_keys, raw_values, parent_values, retain_parent_values=False, compiled_metric_deltas=None, compile_inference_metrics=True):
        if compiled_metric_deltas is not None:
            batch_size = int(parent_values.shape[0])
            metric_batch_size = int(compiled_metric_deltas.shape[0])
            if metric_batch_size == 1 and batch_size > 1:
                compiled_metric_deltas = compiled_metric_deltas.expand(batch_size, *compiled_metric_deltas.shape[1:])
            elif metric_batch_size != batch_size:
                raise ValueError(f'Compiled metric batch {metric_batch_size} does not match values batch {batch_size}')
        (values, context) = super().prepare_bank(raw_keys, raw_values, parent_values, retain_parent_values=retain_parent_values, compiled_metric_deltas=compiled_metric_deltas)
        if compiled_metric_deltas is None and compile_inference_metrics and (not torch.is_grad_enabled()):
            from factread.profile import profile_scope
            with torch.autocast(device_type=parent_values.device.type, enabled=False):
                with profile_scope('factread/compile_active_bank_keys'):
                    context.metric_deltas = torch.stack([self.reader.compile(context.raw_keys[index].float()) for index in range(context.raw_keys.shape[0])]).detach()
        return (values, context)

    def _selected_coefficients(self, context, batch_index: int, ids: torch.Tensor) -> torch.Tensor:
        if context.metric_deltas is not None:
            return context.metric_deltas[batch_index][ids]
        return self.reader.compile(context.raw_keys[batch_index][ids])

    def apply_read(self, base: torch.Tensor, gamma: torch.Tensor, context, metric_intervention: str='none', **_kwargs) -> torch.Tensor:
        if metric_intervention not in {'none', 'identity', 'fixed_spectrum_shared_axis', 'rotate'}:
            raise ValueError(f'Unsupported directional-reader intervention: {metric_intervention}')
        (batch, heads, length, dim) = base.shape
        count = min(self.selected_records, context.raw_keys.shape[1])
        outputs = []
        for batch_index in range(batch):
            chunks = []
            for start in range(0, length, self.query_chunk):
                stop = min(start + self.query_chunk, length)
                from factread.profile import profile_scope
                with profile_scope('factread/topk_gather_select'):
                    weights = gamma[batch_index, :, start:stop].permute(1, 2, 0).float()
                    ids = weights.mean(-1).topk(count, dim=-1).indices
                    masses = weights.gather(1, ids[..., None].expand(-1, -1, heads))
                    values = context.projected_values[batch_index][ids]
                    coeff = self._selected_coefficients(context, batch_index, ids)
                if metric_intervention == 'identity':
                    coeff = torch.zeros_like(coeff)
                elif metric_intervention == 'fixed_spectrum_shared_axis':
                    if self.fixed_shared_axes is None:
                        raise RuntimeError('Fixed-spectrum shared axes were not initialized')
                    coeff = project_to_shared_axes(coeff, self.fixed_shared_axes)
                elif metric_intervention == 'rotate':
                    if self.rotation_angles is None:
                        raise RuntimeError('Spectrum-preserving rotation angles were not initialized')
                    coeff = rotate_coefficients(coeff, self.rotation_angles)
                query = base[batch_index, :, start:stop].transpose(0, 1)
                (corrected, _) = directional_read_with_basis(masses, values, query, coeff, context.basis)
                chunks.append(corrected)
            outputs.append(torch.cat(chunks, dim=0).transpose(0, 1))
        return torch.stack(outputs, dim=0)

class FactReadAdapter(ReadoutAdapter):

    def __init__(self, model, mode: str, layers: tuple[int, ...]=(9, 18, 27), key_dim: int=384, heads: int=32, head_dim: int=128, selected_records: int=8, query_chunk: int=8, effect_scale: float=0.5, rho: float=0.75):
        if mode not in {'balanced', 'shared_axis'}:
            raise ValueError(mode)
        super().__init__(model, layers=layers, mode='full', key_dim=key_dim, heads=heads, head_dim=head_dim, per_head_rank=2, selected_records=selected_records, query_chunk=query_chunk, effect_scale=effect_scale)
        for layer in layers:
            self.readers[str(layer)] = DirectionalKnowledgeRead(mode, key_dim, heads, head_dim, selected_records, query_chunk, rho)
