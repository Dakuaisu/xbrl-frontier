#!/usr/bin/env bash
# Owner-run on the RTX 3050 laptop under WSL2, from the repo root (a commit containing data/frontier_latency_ids.txt).
# Usage: serving/run_rtx3050_load.sh /path/to/export-Qwen3-0.6B /path/to/export-Qwen3-1.7B
# Runs the card's planned variants (serving.hardware.rtx3050_wsl2.variants) with its vllm_flags; no $ is attached.
set -euo pipefail
E06="${1:?export dir of the 0.6B (merged-fp16/, awq-int4/)}"
E17="${2:?export dir of the 1.7B (awq-int4/)}"
python3 -m venv .venv-serve
. .venv-serve/bin/activate
pip install -q "pyyaml==6.0.3" "pyarrow==25.0.1"
VLLM_PIN=$(python -c "import yaml; print(yaml.safe_load(open('docs/config/thresholds.yaml'))['pins']['software']['vllm'])")
[ -n "$VLLM_PIN" ] || { echo "pins.software.vllm is empty" >&2; exit 1; }
pip install -q "vllm==${VLLM_PIN}"
OUT=data/phase4/rtx3050_wsl2
python -m serving.load_test --card rtx3050_wsl2 --model Qwen/Qwen3-0.6B --variant bf16 --checkpoint "$E06/merged-fp16" --out "$OUT"
python -m serving.load_test --card rtx3050_wsl2 --model Qwen/Qwen3-0.6B --variant awq-int4 --checkpoint "$E06/awq-int4" --out "$OUT"
python -m serving.load_test --card rtx3050_wsl2 --model Qwen/Qwen3-1.7B --variant awq-int4 --checkpoint "$E17/awq-int4" --out "$OUT"
