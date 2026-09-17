#!/bin/bash
#SBATCH --job-name=trellis2-step3b-prompts
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/h/hongshan/cp4281-as2/logs/step3b-prompts-%j.log
#SBATCH --error=/home/h/hongshan/cp4281-as2/logs/step3b-prompts-%j.log

set -eo pipefail

PROJECT_ROOT="$HOME/cp4281-as2"
ENV_PREFIX="$PROJECT_ROOT/envs/trellis2"
HF_HOME="$PROJECT_ROOT/hf-cache"
HF_HUB_CACHE="$HF_HOME/hub"
FLUX_REV="741f7c3ce8b383c54771c7003378a50191e9efe9"
FLUX_SNAPSHOT="$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-schnell/snapshots/$FLUX_REV"
EXPERIMENT_SPEC="$PROJECT_ROOT/inputs/step3/experiment_spec.json"
REPORT="$PROJECT_ROOT/logs/step3b-prompt-token-counts-$SLURM_JOB_ID.json"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "Start: $(date --iso-8601=seconds)"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
python --version

test -f "$FLUX_SNAPSHOT/tokenizer/tokenizer_config.json"
test -f "$FLUX_SNAPSHOT/tokenizer_2/tokenizer_config.json"
test -f "$EXPERIMENT_SPEC"
python -m json.tool "$EXPERIMENT_SPEC" > /dev/null

python - "$EXPERIMENT_SPEC" "$FLUX_SNAPSHOT" "$REPORT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


def count_tokens(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


spec_path = Path(sys.argv[1])
snapshot = Path(sys.argv[2])
report_path = Path(sys.argv[3])

with spec_path.open(encoding="utf-8") as handle:
    spec = json.load(handle)

text_spec = spec["text_conditioned"]
template = text_spec["prompt_template"]
cases = text_spec["cases"]
assert text_spec["prompt_template_revision"] == 3, text_spec
assert len(cases) == 5, len(cases)

clip_tokenizer = AutoTokenizer.from_pretrained(
    snapshot / "tokenizer", local_files_only=True
)
t5_tokenizer = AutoTokenizer.from_pretrained(
    snapshot / "tokenizer_2", local_files_only=True
)
clip_limit = int(clip_tokenizer.model_max_length)
t5_limit = int(text_spec["text_to_image"]["max_sequence_length"])
assert clip_limit == 77, clip_limit
assert t5_limit == 256, t5_limit

results = []
for case in cases:
    prompt = template.format(subject_description=case["subject_description"])
    clip_count = count_tokens(clip_tokenizer, prompt)
    t5_count = count_tokens(t5_tokenizer, prompt)
    passed = clip_count <= clip_limit and t5_count <= t5_limit
    result = {
        "case_id": case["case_id"],
        "expanded_prompt": prompt,
        "clip_token_count": clip_count,
        "clip_token_limit": clip_limit,
        "t5_token_count": t5_count,
        "t5_token_limit": t5_limit,
        "status": "passed" if passed else "failed",
    }
    results.append(result)
    print(
        f"{result['case_id']}: CLIP {clip_count}/{clip_limit}; "
        f"T5 {t5_count}/{t5_limit}; {result['status'].upper()}"
    )

report = {
    "schema_version": 1,
    "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
    "verified_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "prompt_template_revision": text_spec["prompt_template_revision"],
    "prompt_template": template,
    "prompt_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    "model_revision": "741f7c3ce8b383c54771c7003378a50191e9efe9",
    "cases": results,
}
temporary = report_path.with_suffix(report_path.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(report_path)

if report["status"] != "passed":
    raise SystemExit("One or more expanded prompts exceed a tokenizer limit")
PY

test -s "$REPORT"
echo "Report: $REPORT"
echo "STEP 3B PROMPT TOKEN VERIFICATION: PASSED"
echo "Finish: $(date --iso-8601=seconds)"
