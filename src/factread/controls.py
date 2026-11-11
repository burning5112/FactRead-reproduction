import math
import torch
from torch import nn
if __package__:
    from .metric import orthonormal_columns, project_values
else:
    from metric_torch_reference import orthonormal_columns, project_values

class ValueDelta(nn.Module):

    def __init__(self, key_dim=384, heads=32, head_dim=128):
        super().__init__()
        (self.heads, self.head_dim) = (heads, head_dim)
        self.delta = nn.Linear(key_dim, heads * head_dim, dtype=torch.float32)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, raw_values, original_values):
        with torch.autocast(device_type=raw_values.device.type, enabled=False):
            d = self.delta(raw_values.float()).reshape(*raw_values.shape[:-1], self.heads, self.head_dim)
            return (original_values.float() + d).to(original_values.dtype)

class HeadSet(nn.Module):

    def __init__(self, key_dim=384, heads=32, head_dim=128, per_head_rank=2):
        super().__init__()
        (self.heads, self.size, self.rank) = (heads, per_head_rank, heads * per_head_rank)
        r = self.rank
        self.phi = nn.Sequential(nn.Linear(key_dim + 2 * r + heads, 768, dtype=torch.float32), nn.SiLU(), nn.Linear(768, 384, dtype=torch.float32))
        self.psi = nn.Sequential(nn.Linear(r + 384 + 1, 192, dtype=torch.float32), nn.SiLU(), nn.Linear(192, r, dtype=torch.float32))
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)
        self.raw_basis = nn.Parameter(orthonormal_columns(torch.randn(heads, head_dim, per_head_rank)))

    def initialize_basis_from_parent_value_weight(self, head_value_weight):
        with torch.no_grad():
            (u, _, _) = torch.linalg.svd(head_value_weight.float(), full_matrices=False)
            self.raw_basis.copy_(u[..., :self.size])

    def basis(self):
        return orthonormal_columns(self.raw_basis)

    def forward(self, base, raw_keys, projected_values, masses, basis):
        with torch.autocast(device_type=base.device.type, enabled=False):
            (y, k, z, g, B) = (base.float(), raw_keys.float(), projected_values.float(), masses.float(), basis.float())
            (q, count, _, _) = z.shape
            r = self.rank
            z = z.reshape(q, count, r)
            z0 = project_values(y, B).reshape(q, r)
            k = math.sqrt(k.shape[-1]) * k / k.norm(dim=-1, keepdim=True).clamp_min(1e-08)
            record_mass = g.mean(-1)
            relative_head_mass = g / record_mass[..., None].clamp_min(1e-08)
            x = torch.cat([k, z, z0[:, None].expand(-1, count, -1), relative_head_mass], -1)
            features = self.phi(x)
            total_mass = record_mass.sum(1, keepdim=True)
            aggregate = (record_mass[..., None] * features).sum(1) / total_mass.clamp_min(1e-08)
            step = total_mass * self.psi(torch.cat([z0, aggregate, total_mass], -1))
            update = torch.einsum('hds,qhs->qhd', B, step.reshape(q, self.heads, self.size))
            return (y + update).to(base.dtype)

class FEMBasedReader(nn.Module):

    def __init__(self, hidden_size=4096, heads=32, head_dim=128, per_head_rank=2):
        super().__init__()
        (self.heads, self.size, self.rank) = (heads, per_head_rank, heads * per_head_rank)
        self.gate = nn.Linear(hidden_size, 2 * self.rank, dtype=torch.float32)
        self.raw_beta = nn.Parameter(torch.empty(heads, per_head_rank, dtype=torch.float32))
        self.raw_basis = nn.Parameter(orthonormal_columns(torch.randn(heads, head_dim, per_head_rank)))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        inv_softplus_one = math.log(math.expm1(1.0))
        with torch.no_grad():
            self.gate.bias[self.rank:] = inv_softplus_one
            self.raw_beta.fill_(math.log(math.expm1(0.1)))

    def initialize_basis_from_parent_value_weight(self, head_value_weight):
        with torch.no_grad():
            (u, _, _) = torch.linalg.svd(head_value_weight.float(), full_matrices=False)
            self.raw_basis.copy_(u[..., :self.size])

    def basis(self):
        return orthonormal_columns(self.raw_basis)

    def forward(self, base, projected_values, masses, basis, token_hidden):
        with torch.autocast(device_type=base.device.type, enabled=False):
            (y, z, gamma, hidden, B) = (base.float(), projected_values.float(), masses.float(), token_hidden.float(), basis.float())
            (q, count, heads, rank) = z.shape
            if heads != self.heads or rank != self.size or gamma.shape != (q, count, heads):
                raise ValueError('FEM reader input shapes do not match its configured head/rank dimensions')
            total = gamma.sum(dim=1)
            live = total > 1e-08
            p = gamma / total[:, None].clamp_min(1e-08)
            uniform = torch.full_like(p, 1.0 / max(count, 1))
            p = torch.where(live[:, None, :], p, uniform)
            mu = (p[..., None] * z).sum(dim=1)
            centered = z - mu[:, None]
            logp = torch.where(p > 0, p.clamp_min(1e-30).log(), torch.full_like(p, -torch.inf))
            beta = torch.nn.functional.softplus(self.raw_beta).reshape(heads, rank)
            beta_safe = beta.clamp_min(0.0001)
            gap = torch.logsumexp(logp[..., None] + beta_safe[None, None] * centered, dim=1) / beta_safe[None]
            variance = (p[..., None] * centered.square()).sum(dim=1)
            gap = torch.where(beta[None] < 0.0001, 0.5 * beta[None] * variance, gap)
            free_energy = mu + gap
            (lambda_raw, gain_raw) = self.gate(hidden).chunk(2, dim=-1)
            lam = lambda_raw.sigmoid().reshape(q, heads, rank)
            gain = torch.nn.functional.softplus(gain_raw).reshape(q, heads, rank)
            gain = gain / gain.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-06)
            correction = total[..., None] * (gain * ((1.0 - lam) * mu + lam * free_energy) - mu)
            correction = torch.where(live[..., None], correction, torch.zeros_like(correction))
            update = torch.einsum('hdr,qhr->qhd', B, correction)
            return (y + update).to(base.dtype)
