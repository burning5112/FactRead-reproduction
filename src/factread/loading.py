"""Load the four methods in the common Llama and MiniLM interface."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig

from .encoder import KnowledgeEncoder, QueryLoRA
from .reader import FactReadAdapter
from .supervised_attention import SupervisedAttention, attach_key_residual


def projection_state(path):
    path = Path(path)
    state = load_file(str(path), device="cpu") if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in state.items() if k.startswith(("projector_k.", "projector_v."))}


def _copy_lm_head(model, backbone):
    from safetensors import safe_open
    backbone = Path(backbone)
    index = backbone / "model.safetensors.index.json"
    shard = backbone / (json.loads(index.read_text())["weight_map"]["lm_head.weight"] if index.exists() else "model.safetensors")
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("lm_head.weight")
    model.lm_head.weight.data.copy_(weight.to(model.lm_head.weight.dtype))
    del weight


def load(method, config, device):
    torch.manual_seed(1607)
    paths = config["models"]
    backbone = paths["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(paths.get("tokenizer", backbone), local_files_only=True, use_fast=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "^"
    handles = []
    query = readout = supervised = None
    if method == "KBLaM":
        from kblam.models.llama3_model import KblamLlamaForCausalLM
        from kblam.models.kblam_config import KBLaMConfig
        llm_config = LlamaConfig.from_pretrained(backbone, local_files_only=True)
        llm_config.base_model_name_or_path = backbone
        llm_config.torch_dtype = torch.bfloat16
        llm_config._attn_implementation = "eager"
        model = KblamLlamaForCausalLM(llm_config)
        _copy_lm_head(model, backbone)
        encoder = KnowledgeEncoder(model.config.hidden_size * (1 + model.config.num_hidden_layers // 3))
        state = load_file(paths["kblam"], device="cpu")
        expected = {f"encoder.{k}" for k in encoder.state_dict()}
        for layer_id in range(0, 31, 3):
            projection = model.model.layers[layer_id].self_attn.q_proj_new
            projection.weight.data = projection.weight.detach().float()
            key = f"llm.layer_{layer_id}.self_attn.q_proj_new.weight"
            projection.weight.data.copy_(state[key])
            expected.add(key)
            if projection.bias is not None:
                key = f"llm.layer_{layer_id}.self_attn.q_proj_new.bias"
                projection.bias.data = projection.bias.detach().float()
                projection.bias.data.copy_(state[key])
                expected.add(key)
            def cast_input(module, args):
                module._input_dtype = args[0].dtype
                return (args[0].to(module.weight.dtype), *args[1:])
            def cast_output(module, args, output):
                return output.to(module._input_dtype)
            handles.extend([projection.register_forward_pre_hook(cast_input), projection.register_forward_hook(cast_output)])
        if set(state) != expected:
            raise ValueError("KBLaM checkpoint tensor names do not match the model")
        encoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
        del state
        model.to(device)
        model.lm_head.to(dtype=next(model.model.parameters()).dtype)
        encoder.to(device)
        kb_config = KBLaMConfig(base_model_name_or_path=backbone, kb_layer_frequency=3, kb_scale_factor=100, top_k_kb=100, dynamic_sparsify=False, sep_query_head=True, attn_implementation="eager")
    else:
        from atlaskv.models.llama3_model import AtlaskvLlamaForCausalLM, set_llama_attention_classes
        from atlaskv.models.kblam_config import AtlasKVConfig
        set_llama_attention_classes(True)
        atlas = paths.get("atlaskv", backbone)
        llm_config = LlamaConfig.from_pretrained(atlas, local_files_only=True)
        llm_config.base_model_name_or_path = backbone
        llm_config._attn_implementation = "eager"
        model = AtlaskvLlamaForCausalLM.from_pretrained(atlas, config=llm_config, device_map={"": str(device)}, torch_dtype="auto", local_files_only=True)
        if paths.get("atlaskv_queries"):
            state = load_file(paths["atlaskv_queries"], device="cpu")
            named = dict(model.named_parameters())
            with torch.no_grad():
                for key, value in state.items():
                    if not ("q_proj_kg" in key or key.endswith("score_shift")) or key not in named:
                        raise ValueError("Invalid AtlasKV query checkpoint")
                    named[key].copy_(value.to(named[key].device, named[key].dtype))
            del state
        encoder = KnowledgeEncoder(model.config.hidden_size * (1 + model.config.num_hidden_layers // 3)).to(device)
        if method in ("AtlasKV", "SR-KI"):
            encoder.load_state_dict(projection_state(paths["atlaskv_encoder"]), strict=True)
        kb_config = AtlasKVConfig(sep_query_head=True, kb_layer_frequency=3, kb_scale_factor=100, use_kg=True, use_hierarchial_kv=False)
        if method == "FactRead":
            state = load_file(paths["factread"], device="cpu")
            encoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
            query = QueryLoRA(model.config.hidden_size).to(device)
            query.load_state_dict({k[11:]: v for k, v in state.items() if k.startswith("query_lora.")}, strict=True)
            query.attach(model)
            readout = FactReadAdapter(model, mode="balanced", layers=(9, 18, 27), effect_scale=0.5, heads=32, head_dim=128, selected_records=8).to(device)
            readout.initialize_bases(encoder)
            readout.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("readout.")}, strict=True)
            expected = {f"{prefix}.{k}" for prefix, module in [("encoder", encoder), ("query_lora", query), ("readout", readout)] for k in module.state_dict()}
            if set(state) != expected:
                raise ValueError("FactRead checkpoint tensor names do not match the model")
            del state
        elif method == "SR-KI":
            supervised = SupervisedAttention(SimpleNamespace(model=model, encoder=encoder), device)
            handles.extend(attach_key_residual(SimpleNamespace(encoder=encoder), supervised))
            state = load_file(paths["srki"], device="cpu")
            params = {"delta_q_weight": supervised.dq_w, "delta_key_weight": supervised.dk_w}
            if supervised.dq_b is not None:
                params["delta_q_bias"] = supervised.dq_b
            if supervised.dk_b is not None:
                params["delta_key_bias"] = supervised.dk_b
            if set(state) != set(params):
                raise ValueError("SR-KI checkpoint tensor names do not match the model")
            with torch.no_grad():
                for key, value in params.items():
                    value.copy_(state[key].to(value.device))
                    value.requires_grad_(False)
            del state
    for component in (model, encoder, query, readout):
        if component is not None:
            component.eval()
            for parameter in component.parameters():
                parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.config.use_cache = False
    return SimpleNamespace(method=method, model=model, encoder=encoder, query=query, readout=readout, supervised=supervised, handles=handles, tokenizer=tokenizer, kb_config=kb_config)


def close(bundle):
    for handle in bundle.handles:
        handle.remove()
    for module in (bundle.supervised, bundle.query, bundle.readout):
        if module is not None:
            module.remove()
