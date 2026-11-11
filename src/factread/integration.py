from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional
import torch
from torch import nn
if __package__:
    from .metric import FactMetricCompiler, project_values, read_chunk
    from .controls import ValueDelta, HeadSet, FEMBasedReader
else:
    from metric_torch_reference import FactMetricCompiler, project_values, read_chunk
    from extra_controls_torch import ValueDelta, HeadSet, FEMBasedReader

@dataclass
class BankContext:
    raw_keys: torch.Tensor
    projected_values: Optional[torch.Tensor]
    metric_deltas: Optional[torch.Tensor]
    basis: Optional[torch.Tensor]
    parent_values: Optional[torch.Tensor] = None
    adapted_values: Optional[torch.Tensor] = None

def project_full_metric(metrics, heads: int, block_size: int, intervention: str, rho: float=0.75):
    if intervention not in {'head_scalar', 'block_diag', 'balanced_projection'}:
        raise ValueError(intervention)
    rank = heads * block_size
    if metrics.shape[-2:] != (rank, rank) or block_size != 2:
        raise ValueError(f'Expected [...,{rank},{rank}] rank-2 head blocks')
    p = metrics.float() + torch.eye(rank, device=metrics.device, dtype=torch.float32)
    blocks = p.reshape(*p.shape[:-2], heads, block_size, heads, block_size)
    diagonal = blocks.diagonal(dim1=-4, dim2=-2).movedim(-1, -3)
    if intervention == 'head_scalar':
        gain = diagonal.diagonal(dim1=-2, dim2=-1).sum(-1) / block_size
        diagonal = gain[..., None, None] * torch.eye(block_size, device=p.device, dtype=p.dtype)
    elif intervention == 'balanced_projection':
        a = (diagonal[..., 0, 0] - diagonal[..., 1, 1]) * 0.5
        b = (diagonal[..., 0, 1] + diagonal[..., 1, 0]) * 0.5
        radius = torch.sqrt(a.square() + b.square())
        scale = torch.clamp(float(rho) / radius.clamp_min(torch.finfo(radius.dtype).tiny), max=1.0)
        (a, b) = (a * scale, b * scale)
        diagonal = torch.stack((torch.stack((1.0 + a, b), dim=-1), torch.stack((b, 1.0 - a), dim=-1)), dim=-2)
    projected = torch.zeros_like(blocks)
    for head in range(heads):
        projected[..., head, :, head, :] = diagonal[..., head, :, :]
    return projected.reshape_as(p) - torch.eye(rank, device=p.device, dtype=p.dtype)

class RecordCoupledKnowledgeRead(nn.Module):

    def __init__(self, mode='full', key_dim=384, heads=32, head_dim=128, per_head_rank=2, selected_records=8, query_chunk=8, solve_mode='joint', metric_cache_max_bytes=64 * 1024 * 1024):
        super().__init__()
        if metric_cache_max_bytes < 0:
            raise ValueError('metric_cache_max_bytes must be nonnegative')
        (self.mode, self.heads, self.head_dim) = (mode, heads, head_dim)
        (self.selected_records, self.query_chunk) = (selected_records, query_chunk)
        self.solve_mode = 'linear' if mode == 'linear' else solve_mode
        self.metric_cache_max_bytes = int(metric_cache_max_bytes)
        self.value_delta = ValueDelta(key_dim, heads, head_dim)
        if mode == 'head_set':
            self.reader = HeadSet(key_dim, heads, head_dim, per_head_rank)
        elif mode == 'fem':
            self.reader = FEMBasedReader(heads=heads, head_dim=head_dim, per_head_rank=per_head_rank)
        elif mode in ('full', 'diag_rotated', 'block', 'linear'):
            compiler_mode = 'full' if mode == 'linear' else mode
            self.reader = FactMetricCompiler(key_dim, heads, head_dim, per_head_rank, mode=compiler_mode)
        elif mode == 'value_delta_only':
            self.reader = None
        else:
            raise ValueError(mode)
        self._compiled_metric_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._compiled_metric_cache_bytes = 0
        self.metric_compile_stats = {'compile_calls': 0, 'compiled_unique_records': 0, 'cache_hit_unique_records': 0, 'cache_evicted_unique_records': 0, 'cache_bytes': 0}

    def clear_metric_cache(self, reset_stats: bool=False) -> None:
        self._compiled_metric_cache.clear()
        self._compiled_metric_cache_bytes = 0
        self.metric_compile_stats['cache_bytes'] = 0
        if reset_stats:
            self.metric_compile_stats = {'compile_calls': 0, 'compiled_unique_records': 0, 'cache_hit_unique_records': 0, 'cache_evicted_unique_records': 0, 'cache_bytes': 0}

    def _compile_selected_metrics(self, raw_keys, selected_ids):
        (unique_ids, inverse) = torch.unique(selected_ids.reshape(-1), sorted=True, return_inverse=True)
        if torch.is_grad_enabled():
            compiled = self.reader.compile(raw_keys[unique_ids])
            self.metric_compile_stats['compile_calls'] += 1
            self.metric_compile_stats['compiled_unique_records'] += int(unique_ids.numel())
            return compiled[inverse].reshape(*selected_ids.shape, self.reader.rank, self.reader.rank)
        ids_cpu = unique_ids.detach().cpu().tolist()
        missing_ids = []
        for record_id in ids_cpu:
            cached = self._compiled_metric_cache.pop(record_id, None)
            if cached is None:
                missing_ids.append(record_id)
            else:
                self._compiled_metric_cache[record_id] = cached
        if missing_ids:
            missing = torch.tensor(missing_ids, dtype=torch.long, device=selected_ids.device)
            compiled = self.reader.compile(raw_keys[missing])
            self.metric_compile_stats['compile_calls'] += 1
            self.metric_compile_stats['compiled_unique_records'] += len(missing_ids)
            compiled_by_id = {}
            for (record_id, metric) in zip(missing_ids, compiled, strict=True):
                metric = metric.detach()
                compiled_by_id[record_id] = metric
                size_bytes = metric.numel() * metric.element_size()
                if size_bytes > self.metric_cache_max_bytes:
                    continue
                while self._compiled_metric_cache and self._compiled_metric_cache_bytes + size_bytes > self.metric_cache_max_bytes:
                    (_evicted_id, evicted) = self._compiled_metric_cache.popitem(last=False)
                    self._compiled_metric_cache_bytes -= evicted.numel() * evicted.element_size()
                    self.metric_compile_stats['cache_evicted_unique_records'] += 1
                self._compiled_metric_cache[record_id] = metric
                self._compiled_metric_cache_bytes += size_bytes
            self.metric_compile_stats['cache_bytes'] = self._compiled_metric_cache_bytes
        else:
            compiled_by_id = {}
        self.metric_compile_stats['cache_hit_unique_records'] += len(ids_cpu) - len(missing_ids)
        unique_metrics = torch.stack([compiled_by_id[record_id] if record_id in compiled_by_id else self._compiled_metric_cache[record_id] for record_id in ids_cpu])
        return unique_metrics[inverse].reshape(*selected_ids.shape, self.reader.rank, self.reader.rank)

    def _selected_common_metrics(self, context, matrices, batch_index: int, selected_ids):
        if matrices is not None:
            selected = matrices[batch_index][selected_ids]
        else:
            selected = self._compile_selected_metrics(context.raw_keys[batch_index], selected_ids)
        return selected.mean(1, keepdim=True).expand(-1, selected_ids.shape[1], -1, -1)

    def initialize_basis_from_parent_value_weight(self, head_value_weight):
        if self.reader is not None:
            self.reader.initialize_basis_from_parent_value_weight(head_value_weight)

    def prepare_bank(self, raw_keys, raw_values, parent_values, retain_parent_values=False, compiled_metric_deltas=None, compile_inference_metrics=True):
        (batch, heads, n, dim) = parent_values.shape
        assert heads == self.heads and dim == self.head_dim
        if raw_keys.ndim == 2:
            raw_keys = raw_keys.unsqueeze(0).expand(batch, -1, -1)
        if raw_values.ndim == 2:
            raw_values = raw_values.unsqueeze(0).expand(batch, -1, -1)
        assert raw_keys.shape[:2] == raw_values.shape[:2] == (batch, n)
        values_bnhd = self.value_delta(raw_values, parent_values.transpose(1, 2))
        values = values_bnhd.transpose(1, 2)
        if self.reader is None:
            return (values, BankContext(raw_keys, None, None, None, parent_values if retain_parent_values else None, values if retain_parent_values else None))
        with torch.autocast(device_type=parent_values.device.type, enabled=False):
            basis = self.reader.basis()
            projected = project_values(values_bnhd, basis)
            deltas = compiled_metric_deltas
        return (values, BankContext(raw_keys, projected, deltas, basis, parent_values if retain_parent_values else None, values if retain_parent_values else None))

    @staticmethod
    def restore_parent_value_mix(base, gamma, context):
        if context.parent_values is None or context.adapted_values is None:
            raise RuntimeError('Parent-value references were not retained for effect scaling')
        with torch.autocast(device_type=base.device.type, enabled=False):
            weights = gamma.float()
            parent = context.parent_values.float()
            adapted = context.adapted_values.float()
            parent_mix = torch.einsum('bhtn,bhnd->bhtd', weights, parent)
            adapted_mix = torch.einsum('bhtn,bhnd->bhtd', weights, adapted)
            return (base.float() - adapted_mix + parent_mix).to(base.dtype)

    def apply_read(self, base, gamma, context, metric_intervention='none', record_permutation=None, token_hidden=None, metric_stats=None):
        if self.reader is None:
            return base
        (batch, heads, length, dim) = base.shape
        n = context.raw_keys.shape[1]
        assert gamma.shape == (batch, heads, length, n)
        count = min(self.selected_records, n)
        if count == 0:
            return base
        if self.mode == 'fem' and token_hidden is None:
            raise ValueError('FEM-based Reader requires the attention input hidden states')
        if self.mode in ('head_set', 'fem') and metric_intervention != 'none':
            raise ValueError(f'Metric intervention {metric_intervention!r} is unavailable for {self.mode}')
        matrices = context.metric_deltas
        per_position_permutation = False
        if self.mode not in ('head_set', 'fem'):
            if metric_intervention == 'permuted':
                if record_permutation is None:
                    raise ValueError('Supply a metric-only permutation; original K/V must remain fixed')
                if record_permutation.shape == (n,):
                    pass
                elif record_permutation.shape == (batch, length, count):
                    per_position_permutation = True
                else:
                    raise ValueError(f'Expected a bank permutation [{n}] or per-position selected IDs [{batch},{length},{count}], got {tuple(record_permutation.shape)}')
            elif metric_intervention not in ('none', 'common', 'identity', 'scalar', 'linear_norm', 'no_reference_norm', 'rayleigh', 'head_scalar', 'block_diag', 'balanced_projection'):
                raise ValueError(metric_intervention)
        batch_outputs = []
        for b in range(batch):
            pieces = []
            for start in range(0, length, self.query_chunk):
                stop = min(start + self.query_chunk, length)
                q = stop - start
                g = gamma[b, :, start:stop].permute(1, 2, 0).float()
                ids = g.mean(-1).topk(count, dim=-1).indices
                mass = g.gather(1, ids[..., None].expand(-1, -1, heads))
                projected = context.projected_values[b][ids]
                y = base[b, :, start:stop].transpose(0, 1)
                if self.mode == 'head_set':
                    out = self.reader(y, context.raw_keys[b][ids], projected, mass, context.basis)
                elif self.mode == 'fem':
                    hidden = token_hidden[b, start:stop]
                    out = self.reader(y, projected, mass, context.basis, hidden)
                else:
                    if metric_intervention == 'identity':
                        rank = self.reader.rank
                        selected_metrics = base.new_zeros(q, count, rank, rank, dtype=torch.float32)
                    elif metric_intervention == 'common':
                        selected_metrics = self._selected_common_metrics(context, matrices, b, ids)
                    else:
                        if per_position_permutation:
                            metric_ids = record_permutation[b, start:stop]
                        elif record_permutation is not None:
                            metric_ids = record_permutation[ids]
                        else:
                            metric_ids = ids
                        if matrices is not None:
                            selected_metrics = matrices[b][metric_ids]
                        else:
                            selected_metrics = self._compile_selected_metrics(context.raw_keys[b], metric_ids)
                        if metric_intervention == 'scalar':
                            rank = selected_metrics.shape[-1]
                            trace_mean = (selected_metrics.diagonal(dim1=-2, dim2=-1).sum(-1) + rank) / rank
                            identity = torch.eye(rank, device=selected_metrics.device, dtype=selected_metrics.dtype)
                            selected_metrics = (trace_mean[..., None, None] - 1.0) * identity
                        elif metric_intervention in ('head_scalar', 'block_diag', 'balanced_projection'):
                            selected_metrics = project_full_metric(selected_metrics, self.heads, self.reader.size, metric_intervention)
                    read_mode = metric_intervention if metric_intervention in ('linear_norm', 'no_reference_norm', 'rayleigh') else self.solve_mode
                    out = read_chunk(y, projected, mass, selected_metrics, context.basis, read_mode, metric_stats=metric_stats)
                pieces.append(out)
            batch_outputs.append(torch.cat(pieces, 0).transpose(0, 1))
        return torch.stack(batch_outputs, 0)
