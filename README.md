# FactRead

Code and fixed evaluation inputs for **FactRead: Augmenting LLMs with External Knowledge through Joint Readout**.

## Setup

Use Linux, Python 3.10, and a CUDA GPU. The experiments were run with PyTorch 2.9.0, Transformers 4.46.0, and an A800 80 GB GPU. Candidates are scored one at a time.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

Download the evaluation inputs and trained parameters from the anonymous mirror:

```bash
python -m factread.download_assets --output assets
```

The download is about 2 GB. Each part and extracted file is checked against its SHA-256 digest. It includes the four fitted knowledge interfaces, 276 evaluation questions, eight candidates per question, fixed knowledge banks, Value donor assignments, precomputed 384-dimensional MiniLM features, and tokenizer assets.

Download the common frozen language checkpoint from the original AtlasKV distribution:

```bash
python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="HaoyuHuang2/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000",
    revision="0a4572b85ffd818f5c3b22a407e428bb2d542c27",
    local_dir="assets/backbone",
    allow_patterns=["*.json", "*.safetensors"],
)
PYCODE
```

All methods use its frozen Llama-3.1-8B-Instruct language backbone. The supplied AtlasKV knowledge Query parameters and score shifts restore the paper's trained parent interface; FactRead and SR-KI load their respective fitted parameters on top. KBLaM uses its matched knowledge-query and encoder checkpoint. The frozen language tensors are shared.

Paths are configured in `configs/main.json` and resolved relative to that file. Validate the complete evaluation inputs before running:

```bash
python -m factread.evaluation --config configs/main.json --check-only
```

## Main table: near-neighbor interference

This table evaluates Near4, Near8, Near16, and Near32, counting the injected facts. Each domain has 92 questions. ACC@1 and MRR use eight shared-prefix answer candidates; VA@1 replaces each bank record's Value once with its fixed external donor while preserving the Keys and other Values.

```bash
for method in KBLaM AtlasKV SR-KI FactRead; do
  python -m factread.evaluation \
    --config configs/main.json --method "$method" --table near \
    --output "outputs/near/$method" --device cuda:0
done
python -m factread.aggregate \
  --root outputs --table near --output outputs/near.csv
```

The full table requires 4,416 clean evaluations and 66,240 Value interventions across the four methods; each evaluates eight candidate continuations. A short functionality check can run one question per domain:

```bash
python -m factread.evaluation \
  --config configs/main.json --method FactRead --table near \
  --banks Near4 --limit-per-domain 1 --output outputs/check-near
```

## Main table: knowledge-bank size

This table evaluates banks of 10, 100, 1,000, and 10,000 facts with ACC@1 and MRR on the same questions and candidates.

```bash
for method in KBLaM AtlasKV SR-KI FactRead; do
  python -m factread.evaluation \
    --config configs/main.json --method "$method" --table kb \
    --output "outputs/kb/$method" --device cuda:0
done
python -m factread.aggregate \
  --root outputs --table kb --output outputs/kb.csv
```

A short check uses `--banks KB10 --limit-per-domain 1` with a separate output directory. Runs resume from completed question/bank pairs when their configuration and code identity match. The table aggregator requires complete coverage of every method, bank, and domain; it rejects partial runs.

## Scoring

Each candidate conditions on the question and common entity–relation prefix. Its score is the mean natural log probability of the Value content tokens. Prefix and EOS tokens do not contribute. The exact question serialization is implemented in `factread.evaluation.score`.

ACC@1 splits credit among tied top candidates. MRR uses the average tied rank, with a score tolerance of `1e-8`. A Value intervention's effect is the clean gold-versus-best-wrong margin minus the intervened margin. VA@1 succeeds only when the signed support effect is strictly larger than every absolute distractor effect. Results are reported on a 0–100 scale; macro values average the three domain means.

The four fitted methods are evaluated with the same frozen backbone and MiniLM fact interface. SR-KI uses supervised attention in this interface, with trained knowledge Query/Key branches at layer 15 and a frozen Value branch. See `THIRD_PARTY.md` for source attribution and licenses.
