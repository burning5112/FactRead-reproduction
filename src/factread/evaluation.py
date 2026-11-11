"""Shared-prefix answer ranking and controlled Value attribution."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

METHODS = ("KBLaM", "AtlasKV", "SR-KI", "FactRead")
NEAR = ("Near4", "Near8", "Near16", "Near32")
KB = ("KB10", "KB100", "KB1000", "KB10000")
BANK_SIZE = {**{f"Near{n}": n for n in (4, 8, 16, 32)}, **{f"KB{n}": n for n in (10, 100, 1000, 10000)}}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ranking_metrics(scores, gold):
    values = [float(x) for x in scores]
    if len(values) != 8 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected eight finite candidate scores")
    g = values[gold]
    above = sum(x > g + 1e-8 for x in values)
    tied = sum(abs(x - g) <= 1e-8 for x in values)
    rank = 1 + above + (tied - 1) / 2
    return {"acc_at1": 1.0 / tied if above == 0 else 0.0, "mrr": 1.0 / rank, "gold_margin": g - max(x for i, x in enumerate(values) if i != gold)}


def value_attribution(deltas, support_index):
    if len(deltas) < 2 or not all(math.isfinite(float(x)) for x in deltas):
        raise ValueError("Expected finite support and distractor effects")
    return float(deltas[support_index] > max(abs(x) for i, x in enumerate(deltas) if i != support_index))


def resolve_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    def resolve(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (path.parent / p).resolve())
    for key in ("panel", "donors"):
        config[key] = resolve(config[key])
    config["models"] = {k: resolve(v) for k, v in config["models"].items()}
    config["features"] = {d: {k: resolve(v) for k, v in spec.items()} for d, spec in config["features"].items()}
    config["checksums"] = {resolve(k): v for k, v in config.get("checksums", {}).items()}
    return config


def validate(config, verify_hashes=True):
    import numpy as np
    if verify_hashes:
        for path, expected in config.get("checksums", {}).items():
            if file_sha(path) != expected:
                raise ValueError(f"Asset checksum mismatch: {Path(path).name}")
    panel = read_jsonl(config["panel"])
    if len(panel) != 276 or collections.Counter(q["domain"] for q in panel) != {"cc": 92, "enron": 92, "pes2o": 92}:
        raise ValueError("The formal panel must contain 92 questions per domain")
    if len({q["question_id"] for q in panel}) != len(panel):
        raise ValueError("Duplicate question IDs")
    arrays = {d: (np.load(s["keys"], mmap_mode="r"), np.load(s["values"], mmap_mode="r")) for d, s in config["features"].items()}
    donors = {}
    for row in read_jsonl(config["donors"]):
        key = (row["question_id"], row["bank"], int(row["bank_index"]))
        if key in donors:
            raise ValueError("Duplicate donor entry")
        donors[key] = row
    for q in panel:
        keys, values = arrays[q["domain"]]
        if keys.shape != values.shape or keys.ndim != 2 or keys.shape[1] != 384:
            raise ValueError("Expected aligned MiniLM features with 384 columns")
        choices = q["choices"]
        if len(choices) != 8 or not 0 <= q["gold_index"] < 8 or not q["common_prefix_token_ids"]:
            raise ValueError("Invalid candidate set")
        if len({tuple(x["content_token_ids"]) for x in choices}) != 8 or any(not c["content_token_ids"] for c in choices):
            raise ValueError("Empty or duplicate candidate tokenization")
        for bank in NEAR + KB:
            b = q["banks"][bank]
            rows, facts = list(map(int, b["source_rows"])), list(map(int, b["fact_ids"]))
            if len(rows) != BANK_SIZE[bank] or len(set(rows)) != len(rows) or len(facts) != len(rows) or len(set(facts)) != len(facts):
                raise ValueError("Invalid knowledge bank")
            if min(rows) < 0 or max(rows) >= len(keys) or q["support_fact_id"] not in facts:
                raise ValueError("Bank rows or support ID are invalid")
            if bank in NEAR:
                for i, row in enumerate(rows):
                    donor = donors[(q["question_id"], bank, i)]
                    if int(donor["target_source_row"]) != row or int(donor["donor_source_row"]) in rows or not 0 <= int(donor["donor_source_row"]) < len(keys):
                        raise ValueError("Invalid Value donor")
    if len(donors) != 276 * sum(BANK_SIZE[b] for b in NEAR):
        raise ValueError("Unexpected donor coverage")
    return panel, donors, arrays


def score(bundle, q, bank, arrays, device, donor=None):
    import numpy as np
    import torch
    tok = bundle.tokenizer
    text = "<|start_header_id|>user<|end_header_id|> " + q["question"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    condition = list(map(int, tok(text, add_special_tokens=True)["input_ids"])) + q["common_prefix_token_ids"]
    b = q["banks"][bank]
    source_rows = list(map(int, b["source_rows"]))
    key_array, value_array = arrays[q["domain"]]
    keys = torch.as_tensor(np.array(key_array[source_rows], dtype=np.float32, copy=True), device=device)
    values = torch.as_tensor(np.array(value_array[source_rows], dtype=np.float32, copy=True), device=device)
    with torch.inference_mode():
        if donor is not None:
            index = int(donor["bank_index"])
            donor_row = int(donor["donor_source_row"])
            if source_rows[index] != int(donor["target_source_row"]) or donor_row in source_rows:
                raise ValueError("Value donor must be outside the target bank")
            previous = values
            values = values.clone()
            values[index] = torch.as_tensor(np.array(value_array[donor_row], dtype=np.float32, copy=True), device=device)
            changed = torch.where(torch.any(values != previous, dim=-1))[0].cpu().tolist()
            if changed != [index]:
                raise ValueError("Exactly one raw Value must change")
        projected_keys, projected_values = bundle.encoder(keys, values)
        if bundle.readout is not None:
            bundle.readout.set_bank(keys, values, record_ids=b["fact_ids"])
        scores = []
        try:
            for choice in q["choices"]:
                target = choice["content_token_ids"]
                ids = torch.tensor([condition + target], dtype=torch.long, device=device)
                output = bundle.model(input_ids=ids, attention_mask=torch.ones_like(ids), kb_kvs=(projected_keys.unsqueeze(0), projected_values.unsqueeze(0)), kb_config=bundle.kb_config, output_attentions=True, output_hidden_states=False, use_cache=False, return_dict=True, save_attention_weights=False)
                logits = output.logits[0, len(condition) - 1:len(condition) + len(target) - 1].float()
                token_lp = logits.log_softmax(-1).gather(-1, torch.tensor(target, device=device).unsqueeze(-1)).squeeze(-1).cpu().tolist()
                if len(token_lp) != len(target) or not all(math.isfinite(x) for x in token_lp):
                    raise ValueError("Invalid candidate log probability")
                scores.append(sum(token_lp) / len(token_lp))
                del output, ids, logits
        finally:
            if bundle.readout is not None:
                bundle.readout.clear_bank()
    return scores


def run(args):
    import torch
    from .loading import load, close
    config = resolve_config(args.config)
    panel, donors, arrays = validate(config)
    if args.check_only:
        print(json.dumps({"status": "PASS", "questions": len(panel), "donors": len(donors)}))
        return
    if args.limit_per_domain is not None:
        counts = collections.Counter()
        selected = []
        for q in panel:
            if counts[q["domain"]] < args.limit_per_domain:
                selected.append(q)
                counts[q["domain"]] += 1
        panel = selected
    banks = tuple(args.banks.split(",")) if args.banks else (NEAR if args.table == "near" else KB)
    allowed = NEAR if args.table == "near" else KB
    if not banks or len(set(banks)) != len(banks) or any(b not in allowed for b in banks):
        raise ValueError("Bank selection does not match the requested table")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"method": args.method, "table": args.table, "banks": list(banks), "question_ids": [q["question_id"] for q in panel], "config_sha256": object_sha(config), "panel_sha256": file_sha(config["panel"]), "donors_sha256": file_sha(config["donors"]), "device": args.device, "protocol": "value-content-serial-v1", "source_sha256": object_sha({str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha(p) for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))})}
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output directory belongs to a different run")
    results_path = out / "scores.jsonl"
    if results_path.exists() and not identity_path.exists():
        raise ValueError("Saved scores have no run identity")
    identity_path.write_text(json.dumps(identity, indent=2))
    old = read_jsonl(results_path) if results_path.exists() else []
    done = {}
    question_map = {q["question_id"]: q for q in panel}
    for row in old:
        key = (row["question_id"], row["bank"])
        if key in done or row["identity_sha256"] != object_sha(identity):
            raise ValueError("Duplicate or incompatible saved result")
        if row["question_id"] not in question_map or row["bank"] not in banks or row["method"] != args.method:
            raise ValueError("Unexpected saved result")
        expected = ranking_metrics(row["scores"], question_map[row["question_id"]]["gold_index"])
        if any(abs(expected[k] - row[k]) > 1e-12 for k in expected):
            raise ValueError("Saved metrics disagree with scores")
        if args.table == "near" and (len(row["value_deltas"]) != BANK_SIZE[row["bank"]] or row["va_at1"] != value_attribution(row["value_deltas"], row["support_index"])):
            raise ValueError("Saved attribution disagrees with interventions")
        done[key] = row
    remaining = [(q, b) for b in banks for q in panel if (q["question_id"], b) not in done]
    torch.set_num_threads(4)
    torch.manual_seed(1607)
    torch.backends.cuda.matmul.allow_tf32 = False
    bundle = load(args.method, config, torch.device(args.device)) if remaining else None
    try:
        for index, (q, bank) in enumerate(remaining, 1):
            scores = score(bundle, q, bank, arrays, args.device)
            metrics = ranking_metrics(scores, q["gold_index"])
            row = {"question_id": q["question_id"], "domain": q["domain"], "method": args.method, "bank": bank, "gold_index": q["gold_index"], "scores": scores, **metrics, "identity_sha256": object_sha(identity)}
            if args.table == "near":
                deltas = []
                for i in range(BANK_SIZE[bank]):
                    changed = score(bundle, q, bank, arrays, args.device, donors[(q["question_id"], bank, i)])
                    deltas.append(metrics["gold_margin"] - ranking_metrics(changed, q["gold_index"])["gold_margin"])
                support = list(map(int, q["banks"][bank]["fact_ids"])).index(int(q["support_fact_id"]))
                row.update(value_deltas=deltas, support_index=support, va_at1=value_attribution(deltas, support))
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            done[(q["question_id"], bank)] = row
            print(json.dumps({"method": args.method, "bank": bank, "completed": index, "total": len(remaining)}), flush=True)
    finally:
        if bundle is not None:
            close(bundle)
    print(json.dumps({"status": "COMPLETE", "rows": len(done), "output": str(results_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--table", choices=("near", "kb"), default="kb")
    parser.add_argument("--method", choices=METHODS, default="FactRead")
    parser.add_argument("--banks")
    parser.add_argument("--output", default="outputs/check")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--limit-per-domain", type=int)
    args = parser.parse_args()
    if args.limit_per_domain is not None and not 1 <= args.limit_per_domain <= 92:
        parser.error("--limit-per-domain must be between 1 and 92")
    run(args)


if __name__ == "__main__":
    main()
