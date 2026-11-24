#!/bin/sh
set -u
prefix=${1:?usage: collect_d3_curve.sh PREFIX}
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
name=d3-$prefix
kernel=adnanrashid29/xbrl-frontier-$name-dev/1
log=data/kaggle/$name-watch.log
while :; do
  state=$(.venv/bin/kaggle kernels status "$kernel" 2>/dev/null) || state=STATUS_UNAVAILABLE
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" >> "$log"
  case "$state" in
    *KernelWorkerStatus.COMPLETE*|*KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL_ACKNOWLEDGED*) break ;;
  esac
  sleep 120
done
while ! .venv/bin/kaggle kernels output "$kernel" -p "data/kaggle/$name-output" --page-size 200 --file-pattern "^$name/.*" >> "$log" 2>&1; do
  sleep 120
done
.venv/bin/kaggle quota >> "$log" 2>&1
case "$state" in
  *KernelWorkerStatus.COMPLETE*) PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m scripts.d3_curve_dev score --prefix "$prefix" >> "$log" 2>&1 ;;
  *) printf '%s\n' 'Terminal failure: outputs retained, scoring not attempted.' >> "$log" ;;
esac
