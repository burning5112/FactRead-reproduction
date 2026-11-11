from __future__ import annotations
from collections.abc import Callable
import hashlib
import random
import torch
import torch.nn.functional as F
from torch import nn
from .integration import BankContext, RecordCoupledKnowledgeRead

class ReadoutAdapter(nn.Module):

    def __init__(self, model, layers: tuple[int, ...]=(9, 18, 27), mode: str='full', key_dim: int=384, heads: int=32, head_dim: int=128, per_head_rank: int=2, selected_records: int=8, query_chunk: int=8, collect_activation_stats: bool=False, effect_scale: float=1.0, collect_mass_stats: bool=True):
        super().__init__()
        if not 0.0 <= effect_scale <= 1.0:
            raise ValueError(f'effect_scale must be in [0, 1], got {effect_scale}')
        self.effect_scale = float(effect_scale)
        attention = {int(module.layer_idx): module for module in model.modules() if getattr(module, 'layer_idx', None) is not None and hasattr(module, 'q_proj_kg')}
        missing = sorted(set(layers) - attention.keys())
        if missing:
            raise ValueError(f'Requested FactRead layers are not AtlasKV injection layers: {missing}')
        self.layers = tuple(layers)
        self.matrix_layers = frozenset(self.layers)
        self.readers = nn.ModuleDict({str(layer): RecordCoupledKnowledgeRead(mode=mode, key_dim=key_dim, heads=heads, head_dim=head_dim, per_head_rank=per_head_rank, selected_records=selected_records, query_chunk=query_chunk) for layer in layers})
        self._bank: tuple[torch.Tensor, torch.Tensor] | None = None
        self._bank_ids: list[int] | None = None
        self._compiled_bank_metrics: dict[int, torch.Tensor] = {}
        self._prepared_values: dict[int, torch.Tensor] = {}
        self._prepared_signatures: dict[int, tuple] = {}
        self._prepared_cache_hits = 0
        self._prepared_compute_calls = 0
        self.active_bank_compile_report: dict[str, object] = {'status': 'EMPTY'}
        self._contexts: dict[int, BankContext] = {}
        self.last_stats: dict[int, dict] = {}
        self.last_correction_stats: dict[int, dict] = {}
        self.collect_activation_stats = collect_activation_stats
        self.collect_mass_stats = bool(collect_mass_stats)
        self.last_forward_gradients: dict[int, dict[str, float]] = {}
        self.metric_intervention = 'none'
        self.stage_mode = 'all_on'
        self.stage_prompt_length: int | None = None
        self.decode_call_index = 0
        self.intervention_qids: list[str] = []
        self.last_intervention_stats: dict[str, int | float] = {}
        self._previous: list[tuple[object, object, object]] = []
        self._callbacks: dict[int, tuple[Callable, Callable]] = {}
        self.directional_intervention: dict | None = None
        self._residual_pending: dict[int, torch.Tensor] = {}
        self._residual_handles: dict[int, object] = {}
        self._attention_modules: dict[int, object] = {}
        self.install(model)

    @torch.no_grad()
    def initialize_bases(self, encoder: nn.Module) -> None:
        weight = encoder.projector_v.weight.detach().float()
        hidden = self.readers[str(self.layers[0])].head_dim
        heads = self.readers[str(self.layers[0])].heads
        width = heads * hidden
        for layer in self.layers:
            start = layer // 3 * width
            stop = start + width
            if stop > weight.shape[0]:
                raise ValueError(f'Layer {layer} Value slice {start}:{stop} exceeds encoder output {weight.shape[0]}')
            per_head = weight[start:stop].reshape(heads, hidden, -1)
            self.readers[str(layer)].initialize_basis_from_parent_value_weight(per_head)

    def install(self, model) -> None:
        attention = {int(module.layer_idx): module for module in model.modules() if getattr(module, 'layer_idx', None) is not None and hasattr(module, 'q_proj_kg')}
        self._attention_modules = attention
        for layer in self.layers:
            module = attention[layer]
            prepare = getattr(module, '_factop_prepare_callback', None)
            read = getattr(module, '_factop_read_callback', None)
            self._previous.append((module, prepare, read))

            def prepare_values(values, layer_idx=layer):
                if self._bank is None:
                    raise RuntimeError('Call set_bank before a Flat-KV FactRead forward')
                (raw_keys, raw_values) = self._bank
                if self.effect_scale == 0.0:
                    context = BankContext(raw_keys, None, None, None, values, values)
                    self._contexts[layer_idx] = context
                    return values
                cacheable = not self.training and (not torch.is_grad_enabled())
                if cacheable:
                    try:
                        version = values._version
                    except RuntimeError:
                        version = None
                    signature = (values.device.type, values.device.index, values.data_ptr(), tuple(values.shape), tuple(values.stride()), values.dtype, version)
                    if self._prepared_signatures.get(layer_idx) == signature:
                        self._prepared_cache_hits += 1
                        self.active_bank_compile_report['prepared_cache_hits'] = self._prepared_cache_hits
                        return self._prepared_values[layer_idx]
                from .profile import profile_scope
                with profile_scope('factread/value_delta_basis_prepare'):
                    (changed, context) = self.readers[str(layer_idx)].prepare_bank(raw_keys, raw_values, values, retain_parent_values=self.effect_scale < 1.0, compiled_metric_deltas=self._compiled_bank_metrics.get(layer_idx), compile_inference_metrics=False)
                self._contexts[layer_idx] = context
                if changed.requires_grad:
                    changed.register_hook(lambda grad, idx=layer_idx: self._save_grad(idx, 'prepared_values', grad))
                if cacheable:
                    self._prepared_values[layer_idx] = changed.detach()
                    self._prepared_signatures[layer_idx] = signature
                self._prepared_compute_calls += 1
                self.active_bank_compile_report['prepare_bank_calls'] = self._prepared_compute_calls
                return changed

            def read_output(base, gamma, layer_idx=layer, cache_position=None, position_ids=None, token_hidden=None):
                context = self._contexts.get(layer_idx)
                if context is None:
                    raise RuntimeError(f'FactRead Value context missing for layer {layer_idx}')
                if self.collect_mass_stats and layer_idx not in self.last_stats:
                    from .profile import profile_scope
                    with profile_scope('factread/mass_stats_sync'):
                        self.last_stats[layer_idx] = self._summarize_mass(gamma)
                reader = self.readers[str(layer_idx)]
                matrix_active = layer_idx in self.matrix_layers
                diagnostic = self.directional_intervention
                diagnostic_active = diagnostic is not None and (diagnostic.get('layer') is None or diagnostic.get('layer') == layer_idx)
                stage_mask = self._stage_read_mask(base, layer_idx, cache_position=cache_position, position_ids=position_ids)
                if self.effect_scale == 0.0 or not matrix_active or stage_mask is False:
                    output = base
                elif diagnostic_active and diagnostic['mode'] == 'simplified':
                    simple_mode = diagnostic.get('simplification')
                    if not simple_mode:
                        raise ValueError('Simplified intervention requires a registered simplification')
                    output = reader.apply_read(base, gamma, context, metric_intervention=simple_mode)
                elif diagnostic_active and diagnostic['mode'] in {'direction_transfer', 'length_transfer', 'rotated_norm_matched'}:
                    if diagnostic['mode'] == 'rotated_norm_matched' and (not getattr(reader, 'is_directional_reader', False)):
                        raise ValueError('Rotated norm matching requires the rank-2 directional reader')
                    from factread.restrictions import norm_match_residual, residual_direction_length_crossover
                    parent_output = reader.restore_parent_value_mix(base, gamma, context)
                    full = reader.apply_read(base, gamma, context)
                    if diagnostic['mode'] == 'rotated_norm_matched':
                        simple = reader.apply_read(base, gamma, context, metric_intervention='rotate')
                    else:
                        simple_mode = diagnostic.get('simplification')
                        if not simple_mode:
                            raise ValueError('Direction/length transfer requires a simplification control')
                        simple = reader.apply_read(base, gamma, context, metric_intervention=simple_mode)
                    weight = module.o_proj.weight.detach().float()

                    def residual_delta(head_delta):
                        flat = head_delta.permute(0, 2, 1, 3).reshape(head_delta.shape[0], head_delta.shape[2], -1).float()
                        return F.linear(flat, weight, bias=None)
                    if diagnostic['mode'] == 'rotated_norm_matched':
                        ref_residual = residual_delta((full - parent_output) * self.effect_scale)
                        rotated_residual = residual_delta((simple - parent_output) * self.effect_scale)
                        (_, per_token_scale) = norm_match_residual(rotated_residual, ref_residual)
                        output = parent_output.float() + per_token_scale[:, None, :, None] * (simple.float() - parent_output.float())
                        output = output.to(base.dtype)
                    else:
                        full_residual = residual_delta((full - parent_output) * self.effect_scale)
                        simple_residual = residual_delta((simple - parent_output) * self.effect_scale)
                        crossed = residual_direction_length_crossover(full_residual, simple_residual)
                        self._residual_pending[layer_idx] = crossed[diagnostic['mode']]
                        output = parent_output
                elif self.metric_intervention == 'none':
                    if diagnostic_active and diagnostic['mode'] in {'identity', 'fixed_spectrum_shared_axis', 'rotate'}:
                        output = reader.apply_read(base, gamma, context, metric_intervention=diagnostic['mode'])
                    elif reader.mode == 'fem':
                        output = reader.apply_read(base, gamma, context, token_hidden=token_hidden)
                    else:
                        output = reader.apply_read(base, gamma, context)
                else:
                    record_permutation = None
                    if self.metric_intervention == 'permuted':
                        record_permutation = self._permuted_selected_ids(gamma, layer_idx, cache_position, position_ids)
                    output = reader.apply_read(base, gamma, context, metric_intervention=self.metric_intervention, record_permutation=record_permutation, token_hidden=token_hidden, metric_stats=self.last_intervention_stats)
                if isinstance(stage_mask, torch.Tensor):
                    mask = stage_mask
                    if mask.ndim == 1:
                        mask = mask.unsqueeze(0).expand(base.shape[0], -1)
                    output = torch.where(mask[:, None, :, None], output, base)
                if matrix_active and self.effect_scale < 1.0 and (self.effect_scale > 0.0):
                    parent_output = reader.restore_parent_value_mix(base, gamma, context)
                    output = (parent_output.float() + self.effect_scale * (output.float() - parent_output.float())).to(base.dtype)
                if self.collect_activation_stats and layer_idx not in self.last_correction_stats:
                    base_vector = base.detach().float()
                    output_vector = output.detach().float()
                    if base_vector.ndim >= 3:
                        base_vector = base_vector[..., -1, :]
                        output_vector = output_vector[..., -1, :]
                    base_vector = base_vector.reshape(base_vector.shape[0], -1)
                    delta = output_vector.reshape(output_vector.shape[0], -1) - base_vector
                    delta_norm = delta.norm(dim=-1)
                    base_norm = base_vector.norm(dim=-1)
                    self.last_correction_stats[layer_idx] = {'base_output_norm': base_norm.cpu().tolist(), 'correction_norm': delta_norm.cpu().tolist(), 'correction_to_base_norm_ratio': (delta_norm / base_norm.clamp_min(1e-12)).cpu().tolist(), 'exact_zero_correction': (delta_norm == 0).cpu().tolist(), 'near_zero_coordinate_fraction': (delta.abs() <= 1e-12).float().mean(-1).cpu().tolist()}
                if output.requires_grad:
                    output.register_hook(lambda grad, idx=layer_idx: self._save_grad(idx, 'read_output', grad))
                return output
            module._factop_prepare_callback = prepare_values
            module._factop_read_callback = read_output
            self._callbacks[layer] = (prepare_values, read_output)

    def set_stage_policy(self, mode: str='all_on', prompt_length: int | None=None) -> None:
        if mode not in {'all_on', 'prefill_only', 'warm4'}:
            raise ValueError(f'Unsupported matrix stage mode: {mode}')
        if prompt_length is not None and prompt_length < 0:
            raise ValueError('prompt_length must be nonnegative')
        self.stage_mode = mode
        self.stage_prompt_length = prompt_length
        self.decode_call_index = 0

    def set_collect_mass_stats(self, enabled: bool) -> None:
        self.collect_mass_stats = bool(enabled)
        if not self.collect_mass_stats:
            self.last_stats.clear()

    def set_matrix_layers(self, layers: tuple[int, ...] | list[int] | None=None) -> None:
        selected = self.layers if layers is None else tuple(map(int, layers))
        if len(set(selected)) != len(selected):
            raise ValueError('Matrix layer subset contains duplicates')
        unknown = sorted(set(selected) - set(self.layers))
        if unknown:
            raise ValueError(f'Matrix layer subset is outside installed FactRead layers: {unknown}')
        self.matrix_layers = frozenset(selected)

    def _stage_read_mask(self, base, layer_idx: int, cache_position=None, position_ids=None):
        if self.stage_mode == 'all_on':
            return None
        positions = cache_position if cache_position is not None else position_ids
        if self.stage_prompt_length is not None:
            from factread.schedule import stage_read_mask
            if positions is not None and positions.ndim == 1:
                positions = positions[-base.shape[-2]:]
            elif positions is not None:
                positions = positions[..., -base.shape[-2]:]
            return stage_read_mask(self.stage_mode, base.shape[-2], base.device, prompt_length=self.stage_prompt_length, positions=positions)
        if base.shape[-2] == 1 and layer_idx == self.layers[0]:
            self.decode_call_index += 1
        from factread.schedule import stage_read_mask
        return stage_read_mask(self.stage_mode, base.shape[-2], base.device, decode_call_index=self.decode_call_index if base.shape[-2] == 1 else None)

    def set_bank(self, raw_keys: torch.Tensor, raw_values: torch.Tensor, record_ids: list[int] | torch.Tensor | None=None) -> None:
        if raw_keys.ndim not in (2, 3) or raw_values.shape != raw_keys.shape:
            raise ValueError(f'Expected aligned raw MiniLM banks [N,E] or [B,N,E], got {tuple(raw_keys.shape)} and {tuple(raw_values.shape)}')
        device = next(self.parameters()).device
        self._bank = (raw_keys.to(device), raw_values.to(device))
        if record_ids is None:
            n = raw_keys.shape[-2]
            self._bank_ids = list(range(n))
        else:
            self._bank_ids = torch.as_tensor(record_ids, dtype=torch.long).tolist()
            if len(self._bank_ids) != raw_keys.shape[-2]:
                raise ValueError('record_ids length does not match the raw MiniLM bank')
        self._contexts.clear()
        self._prepared_values.clear()
        self._prepared_signatures.clear()
        self._prepared_cache_hits = 0
        self._prepared_compute_calls = 0
        self._residual_pending.clear()
        self.last_intervention_stats = {}
        self.invalidate_metric_cache(reset_stats=True)
        self.last_stats.clear()
        self.last_correction_stats.clear()
        self.last_forward_gradients.clear()
        self._compile_active_bank_metrics()

    def _compile_active_bank_metrics(self) -> None:
        self._compiled_bank_metrics.clear()
        keys = self._bank[0]
        if self.metric_intervention == 'identity' or self.effect_scale == 0.0:
            self.active_bank_compile_report = {'status': 'DISABLED_BY_IDENTITY'}
            return
        if self.training:
            self.active_bank_compile_report = {'status': 'GRADIENT_TRAINING_PATH'}
            return
        key_batch = keys if keys.ndim == 3 else keys.unsqueeze(0)
        estimates = []
        for layer in self.layers:
            reader = self.readers[str(layer)]
            compiler = getattr(reader, 'reader', None)
            if compiler is None or not hasattr(compiler, 'compile'):
                self.active_bank_compile_report = {'status': 'UNSUPPORTED_READER'}
                return
            if getattr(reader, 'is_directional_reader', False):
                per_item = reader.reader.heads * 2 * 4
            else:
                rank = reader.reader.rank
                per_item = rank * rank * 4
            estimates.append(key_batch.shape[0] * key_batch.shape[1] * per_item)
        budget = min((int(getattr(self.readers[str(layer)], 'metric_cache_max_bytes', 0)) for layer in self.layers))
        estimated_bytes = sum(estimates)
        if estimated_bytes > budget:
            self.active_bank_compile_report = {'status': 'LAZY_CACHE_FALLBACK', 'estimated_bytes': estimated_bytes, 'budget_bytes': budget}
            return
        with torch.inference_mode():
            for layer in self.layers:
                reader = self.readers[str(layer)]
                with torch.autocast(device_type=keys.device.type, enabled=False):
                    compiled = torch.stack([reader.reader.compile(key_batch[index].float()) for index in range(key_batch.shape[0])]).detach()
                self._compiled_bank_metrics[layer] = compiled
        self.active_bank_compile_report = {'status': 'ACTIVE_BANK_PRECOMPILED', 'active_records': int(key_batch.shape[1]), 'layer_count': len(self.layers), 'estimated_bytes': estimated_bytes, 'actual_bytes': sum((value.numel() * value.element_size() for value in self._compiled_bank_metrics.values())), 'budget_bytes': budget}

    def set_metric_intervention(self, mode: str='none', qids: list[str] | None=None) -> None:
        if mode not in ('none', 'common', 'identity', 'scalar', 'permuted', 'linear_norm', 'no_reference_norm', 'rayleigh', 'head_scalar', 'block_diag', 'balanced_projection'):
            raise ValueError(f'Unsupported FactRead metric intervention: {mode}')
        if mode != self.metric_intervention:
            self._contexts.clear()
            self._prepared_values.clear()
            self._prepared_signatures.clear()
        self.metric_intervention = mode
        self.intervention_qids = list(map(str, qids or []))
        self.last_intervention_stats = {}
        self.invalidate_metric_cache()
        if mode == 'identity':
            self._compiled_bank_metrics.clear()
            if self._bank is not None:
                self.active_bank_compile_report = {'status': 'DISABLED_BY_IDENTITY'}
        elif self._bank is not None and self.active_bank_compile_report.get('status') == 'DISABLED_BY_IDENTITY':
            self._compile_active_bank_metrics()
        if mode == 'permuted' and (not self.intervention_qids):
            raise ValueError('Permuted intervention requires frozen question IDs')

    def set_directional_intervention(self, mode: str='none', *, layer: int | None=None, simplification: str | None=None, axes_by_layer: dict[int, torch.Tensor] | None=None, angles_by_layer: dict[int, torch.Tensor] | None=None) -> None:
        allowed = {'none', 'identity', 'simplified', 'fixed_spectrum_shared_axis', 'rotate', 'rotated_norm_matched', 'direction_transfer', 'length_transfer'}
        if mode not in allowed:
            raise ValueError(f'Unsupported directional intervention: {mode}')
        if layer is not None and layer not in self.layers:
            raise ValueError(f'Diagnostic layer {layer} is not attached: {self.layers}')
        if mode in {'simplified', 'direction_transfer', 'length_transfer'} and simplification not in {'head_scalar', 'fixed_spectrum_shared_axis'}:
            raise ValueError('Simplified/crossover diagnostics require HeadScalar or FixedSpectrum-SharedAxis')
        axes_by_layer = axes_by_layer or {}
        angles_by_layer = angles_by_layer or {}
        for layer_idx in self.layers:
            reader = self.readers[str(layer_idx)]
            if hasattr(reader, 'set_diagnostic_axes'):
                reader.set_diagnostic_axes(axes=axes_by_layer.get(layer_idx), angles=angles_by_layer.get(layer_idx))
        self.directional_intervention = {'mode': mode, 'layer': layer, 'simplification': simplification}
        self._residual_pending.clear()
        if mode in {'direction_transfer', 'length_transfer'}:
            target_layers = self.layers if layer is None else (layer,)
            for layer_idx in target_layers:
                if layer_idx in self._residual_handles:
                    continue
                module = self._attention_modules[layer_idx]
                if not hasattr(module, 'o_proj'):
                    raise RuntimeError(f'FactRead layer {layer_idx} lacks the expected output projection')

                def add_residual_intervention(_projection, _inputs, output, idx=layer_idx):
                    delta = self._residual_pending.pop(idx, None)
                    if delta is None:
                        return output
                    if not torch.is_tensor(output) or output.shape != delta.shape:
                        raise RuntimeError(f'Residual intervention shape mismatch at layer {idx}')
                    return output + delta.to(device=output.device, dtype=output.dtype)
                self._residual_handles[layer_idx] = module.o_proj.register_forward_hook(add_residual_intervention)

    def clear_directional_intervention(self) -> None:
        self.directional_intervention = None
        self._residual_pending.clear()
        for handle in self._residual_handles.values():
            handle.remove()
        self._residual_handles.clear()

    def invalidate_metric_cache(self, reset_stats: bool=False) -> None:
        for reader in self.readers.values():
            reader.clear_metric_cache(reset_stats=reset_stats)

    def step_optimizer_and_invalidate_cache(self, optimizer) -> None:
        optimizer.step()
        self.invalidate_metric_cache()

    def _permuted_selected_ids(self, gamma, layer_idx: int, cache_position, position_ids):
        (batch, _, length, bank_size) = gamma.shape
        count = min(self.readers[str(layer_idx)].selected_records, bank_size)
        selected = gamma.float().mean(dim=1).topk(count, dim=-1).indices
        result = selected.clone()
        positions = cache_position if cache_position is not None else position_ids
        if positions is None:
            position_rows = [list(range(length)) for _ in range(batch)]
        else:
            position_rows = positions.detach().cpu().tolist()
            if positions.ndim == 1:
                position_rows = [position_rows for _ in range(batch)]
            position_rows = [list(row)[-length:] for row in position_rows]
        selected_rows = selected.detach().cpu().tolist()
        stable_ids = self._bank_ids or list(range(bank_size))
        if len(self.intervention_qids) not in (1, batch):
            raise ValueError(f'Need one question ID per batch row (or one shared ID), got {len(self.intervention_qids)} for {batch} rows')
        for batch_index in range(batch):
            qid = self.intervention_qids[batch_index if len(self.intervention_qids) == batch else 0]
            for token_index in range(length):
                if count < 2:
                    continue
                absolute_position = int(position_rows[batch_index][token_index])
                selected_fact_ids = [stable_ids[int(index)] for index in selected_rows[batch_index][token_index]]
                payload = f'{qid}|{layer_idx}|{absolute_position}|{selected_fact_ids}|20261030'.encode('utf-8')
                seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
                rng = random.Random(seed)
                order = list(range(count))
                for _ in range(32):
                    rng.shuffle(order)
                    if all((source != target for (source, target) in enumerate(order))):
                        break
                else:
                    order = list(range(1, count)) + [0]
                result[batch_index, token_index] = selected[batch_index, token_index, order]
        return result

    def _summarize_mass(self, gamma: torch.Tensor) -> dict:
        if self._bank is None:
            raise RuntimeError('FactRead bank is not set')
        g = gamma.detach().float()[:, :, -1, :]
        (batch, heads, n) = g.shape
        count = min(8, n)
        shared = g.mean(1).topk(count, dim=-1).indices
        shared_mass = g.gather(2, shared[:, None, :].expand(-1, heads, -1)).sum(-1)
        per_head = g.topk(count, dim=-1).indices
        total_by_head = g.sum(-1)
        top1_fraction = g.max(-1).values / total_by_head.clamp_min(1e-12)
        pairwise = []
        for left in range(heads):
            for right in range(left + 1, heads):
                overlap = (per_head[:, left, :, None] == per_head[:, right, None, :]).any(-1).float().mean(-1)
                pairwise.append(overlap)
        overlap_mean = torch.stack(pairwise, -1).mean(-1) if pairwise else torch.zeros(batch)
        ids = self._bank_ids or list(range(n))
        return {'active_n': n, 'selected_k': count, 'record_ids_by_batch': [[ids[int(index)] for index in row] for row in shared.detach().cpu().tolist()], 'total_mass_by_head': g.sum(-1).cpu().tolist(), 'selected_mass_by_head': shared_mass.cpu().tolist(), 'top1_record_mass_fraction_by_head': top1_fraction.cpu().tolist(), 'per_head_topk_mean_pairwise_overlap': overlap_mean.cpu().tolist()}

    def _save_grad(self, layer_idx: int, name: str, grad: torch.Tensor) -> None:
        self.last_forward_gradients.setdefault(layer_idx, {})[name] = float(grad.detach().float().norm().item())

    def clear_bank(self) -> None:
        self._bank = None
        self._bank_ids = None
        self._compiled_bank_metrics.clear()
        self.active_bank_compile_report = {'status': 'EMPTY'}
        self._contexts.clear()
        self._prepared_values.clear()
        self._prepared_signatures.clear()
        self._residual_pending.clear()
        self.invalidate_metric_cache()
        self.last_stats.clear()
        self.last_correction_stats.clear()
        self.last_forward_gradients.clear()

    def remove(self) -> None:
        for (module, prepare, read) in self._previous:
            module._factop_prepare_callback = prepare
            module._factop_read_callback = read
        self._previous.clear()
        self._callbacks.clear()
        self.clear_directional_intervention()
        self.clear_bank()
