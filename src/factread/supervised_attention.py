import torch
import torch.nn.functional as F
LAYER = 15
CHUNK = 5


class Capture:

    def __init__(self):
        self.gamma = None

    def reset(self):
        self.gamma = None

    def callback(self, layer, gamma, **kwargs):
        if int(layer) != LAYER:
            raise RuntimeError(f'wrong capture layer {layer}')
        self.gamma = gamma
        return None

class SupervisedAttention:

    def __init__(self, bundle, device):
        self.bundle = bundle
        self.device = torch.device(device)
        self.model = bundle.model
        self.encoder = bundle.encoder
        self.model.eval()
        self.encoder.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        mods = {int(m.layer_idx): m for m in self.model.modules() if getattr(m, 'layer_idx', None) is not None and hasattr(m, 'q_proj_kg')}
        if LAYER not in mods:
            raise RuntimeError('A0 has no knowledge-query projection at L15')
        self.attn = mods[LAYER]
        self.modules = mods
        self.hidden = int(self.model.config.hidden_size)
        outdim = int(self.encoder.projector_k.out_features)
        if outdim != self.hidden * (1 + int(self.model.config.num_hidden_layers) // 3):
            raise ValueError(f'unexpected projector_k out dim {outdim}')
        if LAYER // 3 != CHUNK:
            raise ValueError('L15 key chunk mapping changed')
        self.start = CHUNK * self.hidden
        self.end = (CHUNK + 1) * self.hidden
        if self.end > outdim:
            raise ValueError('selected Key chunk out of bounds')
        ln = self.encoder.key_layernorm
        if getattr(ln, 'elementwise_affine', True):
            raise ValueError('expected frozen non-affine global Key LayerNorm')
        self.eps = float(ln.eps)
        self.dq_w = torch.nn.Parameter(torch.zeros_like(self.attn.q_proj_kg.weight.detach(), dtype=torch.float32, device=self.device))
        self.dq_b = torch.nn.Parameter(torch.zeros_like(self.attn.q_proj_kg.bias.detach(), dtype=torch.float32, device=self.device)) if self.attn.q_proj_kg.bias is not None else None
        self.dk_w = torch.nn.Parameter(torch.zeros_like(self.encoder.projector_k.weight.detach()[self.start:self.end], dtype=torch.float32, device=self.device))
        self.dk_b = torch.nn.Parameter(torch.zeros_like(self.encoder.projector_k.bias.detach()[self.start:self.end], dtype=torch.float32, device=self.device)) if self.encoder.projector_k.bias is not None else None
        self.capture = Capture()
        self.handles = []

        def qhook(module, args, out):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError('q_proj_kg input missing')
            return out + F.linear(args[0].float(), self.dq_w, self.dq_b).to(out.dtype)
        self.handles.append(self.attn.q_proj_kg.register_forward_hook(qhook))

        def attn_gate(module, args, kwargs):
            kwargs['output_attentions'] = int(module.layer_idx) == LAYER
            return (args, kwargs)
        for m in mods.values():
            self.handles.append(m.register_forward_pre_hook(attn_gate, with_kwargs=True))
        self.attn._factop_gamma_override_callback = self.capture.callback

    def project_keys(self, raw):
        with torch.no_grad():
            base = self.encoder.projector_k(raw).float()
            norm = self.encoder.key_layernorm(base).float()
            scale = torch.rsqrt(base.var(dim=-1, unbiased=False, keepdim=True) + self.eps)
        delta = F.linear(raw.float(), self.dk_w, self.dk_b)
        out = norm.clone()
        out[..., self.start:self.end] = out[..., self.start:self.end] + delta * scale
        return out.to(torch.bfloat16)

    def project_values(self, raw):
        with torch.no_grad():
            return self.encoder.projector_v(raw).to(torch.bfloat16)

    def remove(self):
        for h in self.handles:
            h.remove()
        if hasattr(self.attn, '_factop_gamma_override_callback'):
            delattr(self.attn, '_factop_gamma_override_callback')

def attach_key_residual(bundle, s):
    holder = {}

    def remember_raw(module, args, out):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError('projector_k raw input unavailable')
        holder['raw'] = args[0]
        return out

    def inject_after_norm(module, args, out):
        raw = holder.pop('raw', None)
        if raw is None:
            raise RuntimeError('projector_k/key_layernorm hook order mismatch')
        base = args[0].float()
        scale = torch.rsqrt(base.var(dim=-1, unbiased=False, keepdim=True) + s.eps)
        delta = F.linear(raw.float(), s.dk_w, s.dk_b)
        adjusted = out.float().clone()
        adjusted[..., s.start:s.end] = adjusted[..., s.start:s.end] + delta * scale
        return adjusted.to(out.dtype)
    h1 = bundle.encoder.projector_k.register_forward_hook(remember_raw)
    h2 = bundle.encoder.key_layernorm.register_forward_hook(inject_after_norm)
    return [h1, h2]
