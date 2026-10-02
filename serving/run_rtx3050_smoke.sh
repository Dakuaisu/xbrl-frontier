#!/usr/bin/env bash
# Owner-run on the RTX 3050 laptop under WSL2, from the repo root. Usage:
#   serving/run_rtx3050_smoke.sh /path/to/smoke_checkpoints
# The checkpoints directory is the unpacked smoke_checkpoints.tar.gz from the Kaggle notebook.
set -euo pipefail
CKPT="${1:?path to smoke_checkpoints}"
python3 -m venv .venv-serve
. .venv-serve/bin/activate
pip install -q "pyyaml==6.0.3" "pyarrow==25.0.1"
VLLM_PIN=$(python -c "import yaml; print(yaml.safe_load(open('docs/config/thresholds.yaml'))['pins']['software']['vllm'])")
[ -n "$VLLM_PIN" ] || { echo "pins.software.vllm is empty; owner approval needed first" >&2; exit 1; }
pip install -q "vllm==${VLLM_PIN}"
python -m serving.smoke_test --card rtx3050_wsl2 --checkpoints "$CKPT" --out data/phase0/rtx3050_wsl2
