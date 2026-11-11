from __future__ import annotations
import math
import torch

def principal_shared_axes(coefficients: torch.Tensor) -> torch.Tensor:
    values = coefficients.detach().float()
    if values.ndim != 3 or values.shape[-1] != 2 or values.shape[0] == 0:
        raise ValueError('Expected non-empty [facts, heads, 2] coefficients')
    covariance = torch.einsum('nhd,nhe->hde', values, values) / values.shape[0]
    (_, vectors) = torch.linalg.eigh(covariance)
    axes = vectors[:, :, -1]
    sign = torch.where(axes[:, 0] < 0, -1.0, 1.0)
    return axes * sign[:, None]

def project_to_shared_axes(coefficients: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    coeff = coefficients.float()
    axis = torch.nn.functional.normalize(axes.float(), dim=-1)
    if coeff.shape[-2:] != axis.shape:
        raise ValueError(f'Coefficient/axis shape mismatch: {coeff.shape} vs {axis.shape}')
    dot = (coeff * axis).sum(-1, keepdim=True)
    sign = torch.where(dot < 0, -1.0, 1.0)
    return coeff.norm(dim=-1, keepdim=True) * sign * axis

def rotate_coefficients(coefficients: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    coeff = coefficients.float()
    phi = angles.float()
    (a, b) = coeff.unbind(-1)
    while phi.ndim < a.ndim:
        phi = phi.unsqueeze(0)
    (c, s) = (torch.cos(2.0 * phi), torch.sin(2.0 * phi))
    return torch.stack((a * c + b * s, b * c - a * s), dim=-1)

def residual_direction_length_crossover(full: torch.Tensor, shared: torch.Tensor, eps: float=1e-12) -> dict[str, torch.Tensor]:
    if full.shape != shared.shape:
        raise ValueError('Full and shared residual corrections must have identical shapes')
    (cf, cc) = (full.float(), shared.float())
    rf = torch.linalg.vector_norm(cf, dim=-1, keepdim=True)
    rc = torch.linalg.vector_norm(cc, dim=-1, keepdim=True)
    uf = torch.where(rf > eps, cf / rf.clamp_min(eps), torch.zeros_like(cf))
    uc = torch.where(rc > eps, cc / rc.clamp_min(eps), torch.zeros_like(cc))
    return {'full': cf, 'shared': cc, 'direction_transfer': rc * uf, 'length_transfer': rf * uc, 'full_norm': rf.squeeze(-1), 'shared_norm': rc.squeeze(-1)}

def norm_match_residual(rotated: torch.Tensor, reference: torch.Tensor, eps: float=1e-12):
    (rot, ref) = (rotated.float(), reference.float())
    nr = torch.linalg.vector_norm(rot, dim=-1, keepdim=True)
    nf = torch.linalg.vector_norm(ref, dim=-1, keepdim=True)
    scale = torch.where(nr > eps, nf / nr.clamp_min(eps), torch.zeros_like(nr))
    return (rot * scale, scale.squeeze(-1))

def seeded_head_angles(heads: int, seed: int, device=None) -> torch.Tensor:
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    angles = (torch.rand(heads, generator=generator) - 0.5) * math.pi
    return angles.to(device=device)
