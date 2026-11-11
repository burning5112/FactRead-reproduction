import math
import torch
from torch import nn

def orthonormal_columns(raw):
    (q, r) = torch.linalg.qr(raw.float(), mode='reduced')
    signs = torch.where(r.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).detach()
    return q * signs.unsqueeze(-2)

def symmetric_from_packed(x, rank):
    (i, j) = torch.triu_indices(rank, rank, device=x.device)
    scale = torch.where(i == j, 1.0, 1.0 / math.sqrt(2.0)).to(x.dtype)
    out = x.new_zeros(*x.shape[:-1], rank, rank)
    out[..., i, j] = x * scale
    out[..., j, i] = x * scale
    return out

class FactMetricCompiler(nn.Module):

    def __init__(self, key_dim=384, heads=32, head_dim=128, per_head_rank=2, bound=math.log(4.0), mode='full', diag_width=1792, block_width=1664):
        super().__init__()
        (self.heads, self.head_dim, self.size) = (heads, head_dim, per_head_rank)
        (self.rank, self.bound, self.mode) = (heads * per_head_rank, bound, mode)
        rank = self.rank
        if mode == 'full':
            self.project = nn.Linear(key_dim, rank * (rank + 1) // 2, dtype=torch.float32)
            nn.init.zeros_(self.project.weight)
            nn.init.zeros_(self.project.bias)
        elif mode == 'diag_rotated':
            self.project = nn.Sequential(nn.Linear(key_dim, diag_width, dtype=torch.float32), nn.SiLU(), nn.Linear(diag_width, rank, dtype=torch.float32))
            nn.init.zeros_(self.project[-1].weight)
            nn.init.zeros_(self.project[-1].bias)
            self.raw_rotation = nn.Parameter(torch.eye(rank, dtype=torch.float32))
        elif mode == 'block':
            out_dim = heads * per_head_rank * (per_head_rank + 1) // 2
            self.project = nn.Sequential(nn.Linear(key_dim, block_width, dtype=torch.float32), nn.SiLU(), nn.Linear(block_width, out_dim, dtype=torch.float32))
            nn.init.zeros_(self.project[-1].weight)
            nn.init.zeros_(self.project[-1].bias)
        else:
            raise ValueError(mode)
        initial = orthonormal_columns(torch.randn(heads, head_dim, per_head_rank))
        self.raw_basis = nn.Parameter(initial)

    def initialize_basis_from_parent_value_weight(self, head_value_weight):
        with torch.no_grad():
            (u, _, _) = torch.linalg.svd(head_value_weight.float(), full_matrices=False)
            self.raw_basis.copy_(u[..., :self.size])

    def basis(self):
        return orthonormal_columns(self.raw_basis)

    def compile(self, raw_keys):
        if next(self.project.parameters()).dtype != torch.float32:
            raise TypeError('Attach FP32 compiler after parent dtype conversion')
        with torch.autocast(device_type=raw_keys.device.type, enabled=False):
            e = raw_keys.float()
            x = math.sqrt(e.shape[-1]) * e / e.norm(dim=-1, keepdim=True).clamp_min(1e-08)
            q = self.project(x)
            denom = (1 + q.square().sum(-1)).sqrt()
            if self.mode == 'full':
                s = self.bound * symmetric_from_packed(q, self.rank) / denom[..., None, None]
                p = torch.matrix_exp(s)
                a = p - torch.eye(self.rank, device=p.device, dtype=p.dtype)
            elif self.mode == 'diag_rotated':
                s = self.bound * q / denom[..., None]
                rotation = orthonormal_columns(self.raw_rotation)
                a = torch.einsum('ir,...r,jr->...ij', rotation, s.expm1(), rotation)
            else:
                packed = self.size * (self.size + 1) // 2
                qh = q.reshape(*q.shape[:-1], self.heads, packed)
                s = self.bound * symmetric_from_packed(qh, self.size) / denom[..., None, None, None]
                blocks = torch.matrix_exp(s) - torch.eye(self.size, device=s.device, dtype=s.dtype)
                a = q.new_zeros(*q.shape[:-1], self.rank, self.rank)
                for h in range(self.heads):
                    sl = slice(h * self.size, (h + 1) * self.size)
                    a[..., sl, sl] = blocks[..., h, :, :]
            return (a + a.transpose(-1, -2)) * 0.5

def project_values(values, basis):
    return torch.einsum('...hd,hds->...hs', values.float(), basis.float())

def read_chunk(base, projected_values, masses, metric_deltas, basis, mode='joint', metric_stats=None):
    with torch.autocast(device_type=base.device.type, enabled=False):
        (y, z, g, a, B) = (base.float(), projected_values.float(), masses.float(), metric_deltas.float(), basis.float())
        (q, heads, _) = y.shape
        size = B.shape[-1]
        rank = heads * size
        z0 = project_values(y, B).reshape(q, rank)
        residual = z.reshape(q, z.shape[1], rank) - z0[:, None]
        positive = g > 0
        root = torch.where(positive, torch.where(positive, g, torch.ones_like(g)).sqrt(), torch.zeros_like(g))
        ell = root.repeat_interleave(size, dim=-1)
        effective = ell[..., :, None] * a * ell[..., None, :]
        parts = torch.einsum('qkij,qkj->qki', effective, residual)
        eye = torch.eye(rank, device=y.device, dtype=y.dtype)
        if mode == 'joint':
            G = eye + effective.sum(1)
            chol = torch.linalg.cholesky((G + G.transpose(-1, -2)) * 0.5)
            step = torch.cholesky_solve(parts.sum(1)[..., None], chol).squeeze(-1)
        elif mode == 'linear':
            step = parts.sum(1)
        elif mode == 'linear_norm':
            linear_step = parts.sum(1)
            G = eye + effective.sum(1)
            chol = torch.linalg.cholesky((G + G.transpose(-1, -2)) * 0.5)
            joint_step = torch.cholesky_solve(parts.sum(1)[..., None], chol).squeeze(-1)
            linear_update = torch.einsum('hds,qhs->qhd', B, linear_step.reshape(q, heads, size))
            joint_update = torch.einsum('hds,qhs->qhd', B, joint_step.reshape(q, heads, size))
            linear_norm = linear_update.flatten(1).norm(dim=-1)
            joint_norm = joint_update.flatten(1).norm(dim=-1)
            valid = linear_norm > 1e-08
            ratio = torch.where(valid, joint_norm / linear_norm.clamp_min(1e-08), torch.zeros_like(linear_norm))
            update = linear_update * ratio[:, None, None]
            if metric_stats is not None:
                metric_stats['linear_norm_rows'] = metric_stats.get('linear_norm_rows', 0) + int(q)
                metric_stats['linear_norm_degenerate_rows'] = metric_stats.get('linear_norm_degenerate_rows', 0) + int((~valid).sum().item())
                metric_stats['linear_norm_min_denominator'] = min(metric_stats.get('linear_norm_min_denominator', float('inf')), float(linear_norm.min().item()))
            out = y + update
            return out.to(base.dtype)
        elif mode == 'no_reference_norm':
            G = eye + effective.sum(1)
            chol = torch.linalg.cholesky((G + G.transpose(-1, -2)) * 0.5)
            full_step = torch.cholesky_solve(parts.sum(1)[..., None], chol).squeeze(-1)
            no_reference_parts = torch.einsum('qkij,qkj->qki', effective, z.reshape(q, z.shape[1], rank))
            no_reference_step = torch.cholesky_solve(no_reference_parts.sum(1)[..., None], chol).squeeze(-1)
            full_update = torch.einsum('hds,qhs->qhd', B, full_step.reshape(q, heads, size))
            no_reference_update = torch.einsum('hds,qhs->qhd', B, no_reference_step.reshape(q, heads, size))
            target_norm = full_update.flatten(1).norm(dim=-1)
            denominator = no_reference_update.flatten(1).norm(dim=-1)
            valid = denominator > 1e-12
            ratio = torch.where(valid, target_norm / denominator.clamp_min(1e-12), torch.zeros_like(denominator))
            update = no_reference_update * ratio[:, None, None]
            out = y + update
            if metric_stats is not None:
                metric_stats['no_reference_norm_rows'] = metric_stats.get('no_reference_norm_rows', 0) + int(q)
                metric_stats['no_reference_norm_degenerate_rows'] = metric_stats.get('no_reference_norm_degenerate_rows', 0) + int((~valid).sum().item())
                positive_denominator = denominator[valid]
                if positive_denominator.numel():
                    metric_stats['no_reference_norm_min_denominator'] = min(metric_stats.get('no_reference_norm_min_denominator', float('inf')), float(positive_denominator.min().item()))
                changed = (out.to(base.dtype) != base).any(dim=(-1, -2))
                metric_stats['no_reference_norm_bf16_changed_rows'] = metric_stats.get('no_reference_norm_bf16_changed_rows', 0) + int(changed.sum().item())
            return out.to(base.dtype)
        elif mode == 'rayleigh':
            rhs = parts.sum(1)
            G = eye + effective.sum(1)
            Gb = torch.einsum('qij,qj->qi', G, rhs)
            numerator = rhs.square().sum(-1)
            denominator = (rhs * Gb).sum(-1)
            nonzero = numerator > 0
            invalid = nonzero & (denominator <= 0)
            if bool(invalid.any()):
                raise FloatingPointError('Rayleigh step has a non-positive quadratic denominator')
            step_size = torch.where(nonzero, numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny), torch.zeros_like(numerator))
            step = step_size[:, None] * rhs
            if metric_stats is not None:
                metric_stats['rayleigh_rows'] = metric_stats.get('rayleigh_rows', 0) + int(q)
                metric_stats['rayleigh_zero_rhs_rows'] = metric_stats.get('rayleigh_zero_rhs_rows', 0) + int((~nonzero).sum().item())
                valid_steps = step_size[nonzero]
                if valid_steps.numel():
                    metric_stats['rayleigh_step_size_sum'] = metric_stats.get('rayleigh_step_size_sum', 0.0) + float(valid_steps.sum().item())
                    metric_stats['rayleigh_step_size_count'] = metric_stats.get('rayleigh_step_size_count', 0) + int(valid_steps.numel())
        elif mode == 'independent':
            G = eye + effective
            chol = torch.linalg.cholesky((G + G.transpose(-1, -2)) * 0.5)
            step = torch.cholesky_solve(parts[..., None], chol).squeeze(-1).sum(1)
        else:
            raise ValueError(mode)
        update = torch.einsum('hds,qhs->qhd', B, step.reshape(q, heads, size))
        out = y + update
    return out.to(base.dtype)
