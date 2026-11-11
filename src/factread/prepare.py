"""Create compact evaluation inputs from the fixed panel and candidate files."""
import argparse
import json
from pathlib import Path
from .evaluation import NEAR, KB, read_jsonl


def prepare(panel_source, candidates_source, donor_sources, output):
    candidates = {r["question_id"]: r for r in read_jsonl(candidates_source)}
    panel = read_jsonl(panel_source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destinations = (output / "panel.jsonl", output / "donors.jsonl")
    if any(p.exists() for p in destinations):
        raise FileExistsError("Use a new output directory")
    def source_bank(bank):
        return "K32" if bank == "Near32" else bank
    with destinations[0].open("w", encoding="utf-8") as handle:
        for q in panel:
            c = candidates[q["question_id"]]
            for key in ("domain", "question", "support_fact_id"):
                if c[key] != q[key]:
                    raise ValueError("Panel and candidate identities differ")
            row = {k: q[k] for k in ("question_id", "domain", "question", "support_fact_id")}
            row.update(common_prefix_token_ids=c["common_prefix_token_ids"], gold_index=int(c["gold_index"]), choices=[{"content_token_ids": x["content_token_ids"]} for x in c["choices"]])
            row["banks"] = {b: {k: q["banks"][source_bank(b)][k] for k in ("source_rows", "fact_ids")} for b in NEAR + KB}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with destinations[1].open("w", encoding="utf-8") as handle:
        for bank in NEAR:
            for donor in read_jsonl(donor_sources[bank]):
                row = {k: donor[k] for k in ("question_id", "bank_index", "target_source_row", "donor_source_row")}
                row["bank"] = bank
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--donor", action="append", required=True, metavar="BANK=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    donors = dict(x.split("=", 1) for x in args.donor)
    if set(donors) != set(NEAR):
        parser.error("Provide one --donor entry for each of Near4, Near8, Near16, Near32")
    prepare(args.panel, args.candidates, donors, args.output)


if __name__ == "__main__":
    main()
