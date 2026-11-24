#!/bin/sh
set -u
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
kernel=adnanrashid29/xbrl-frontier-d3-20k-dev/1
while :; do
  state=$(.venv/bin/kaggle kernels status "$kernel" 2>/dev/null) || state=STATUS_UNAVAILABLE
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" >> data/kaggle/d3-20k-watch.log
  case "$state" in
    *KernelWorkerStatus.COMPLETE*|*KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL_ACKNOWLEDGED*) break ;;
  esac
  sleep 120
done
while ! .venv/bin/kaggle kernels output "$kernel" -p data/kaggle/d3-20k-output --page-size 200 --file-pattern '^d3-20k/.*' >> data/kaggle/d3-20k-watch.log 2>&1; do
  sleep 120
done
.venv/bin/kaggle quota >> data/kaggle/d3-20k-watch.log 2>&1
case "$state" in
  *KernelWorkerStatus.COMPLETE*) PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m scripts.d3_20k_score >> data/kaggle/d3-20k-watch.log 2>&1 ;;
  *) printf '%s\n' 'Terminal failure: outputs retained, scoring not attempted.' >> data/kaggle/d3-20k-watch.log ;;
esac
