"""Aggregate complete model runs into the two main result tables."""
import argparse
import collections
import csv
import json
from pathlib import Path
from .evaluation import METHODS, NEAR, KB, BANK_SIZE, read_jsonl, ranking_metrics, value_attribution, object_sha


def aggregate(root, table, output):
    banks = NEAR if table == "near" else KB
    fields = ("acc_at1", "mrr", "va_at1") if table == "near" else ("acc_at1", "mrr")
    result = []
    shared_panel = shared_donors = None
    for method in METHODS:
        directory = Path(root) / table / method
        identity = json.loads((directory / "identity.json").read_text())
        rows = read_jsonl(directory / "scores.jsonl")
        if identity["method"] != method or identity["table"] != table or set(identity["banks"]) != set(banks):
            raise ValueError("Run identity does not match the requested full table")
        if len(identity["question_ids"]) != 276 or len(set(identity["question_ids"])) != 276:
            raise ValueError("A full table needs all 276 questions")
        panel_id = (identity["panel_sha256"], tuple(identity["question_ids"]))
        if shared_panel is not None and shared_panel != panel_id:
            raise ValueError("Methods used different evaluation panels")
        if shared_donors is not None and shared_donors != identity["donors_sha256"]:
            raise ValueError("Methods used different Value donors")
        shared_panel, shared_donors = panel_id, identity["donors_sha256"]
        seen = set()
        for row in rows:
            key = (row["question_id"], row["bank"])
            if key in seen or row["question_id"] not in identity["question_ids"] or row["bank"] not in banks or row["method"] != method or row["identity_sha256"] != object_sha(identity):
                raise ValueError("Duplicate, unexpected, or incompatible result")
            seen.add(key)
            expected = ranking_metrics(row["scores"], row["gold_index"])
            if any(abs(expected[k] - row[k]) > 1e-12 for k in expected):
                raise ValueError("Stored ranking metrics do not match scores")
            if table == "near":
                if len(row["value_deltas"]) != BANK_SIZE[row["bank"]] or row["va_at1"] != value_attribution(row["value_deltas"], row["support_index"]):
                    raise ValueError("Stored Value attribution does not match interventions")
        if len(seen) != 276 * len(banks):
            raise ValueError("Incomplete table coverage")
        for bank in banks:
            per_domain = {}
            for domain in ("enron", "pes2o", "cc"):
                group = [r for r in rows if r["bank"] == bank and r["domain"] == domain]
                if len(group) != 92:
                    raise ValueError("Expected exactly 92 questions per domain and bank")
                per_domain[domain] = {k: 100.0 * sum(r[k] for r in group) / len(group) for k in fields}
                result.append({"method": method, "domain": domain, "bank": bank, **per_domain[domain]})
            result.append({"method": method, "domain": "macro", "bank": bank, **{k: sum(v[k] for v in per_domain.values()) / 3 for k in fields}})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "domain", "bank", *fields])
        writer.writeheader()
        writer.writerows({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in row.items()} for row in result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--table", choices=("near", "kb"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(args.root, args.table, args.output)
    print(json.dumps({"status": "COMPLETE", "cells": len(result) * (3 if args.table == "near" else 2), "output": args.output}))


if __name__ == "__main__":
    main()
