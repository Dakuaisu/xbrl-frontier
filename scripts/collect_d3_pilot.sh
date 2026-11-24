#!/bin/sh
set -u
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
kernel=adnanrashid29/xbrl-frontier-d3-final-dev-pilot/1
while :; do
  state=$(.venv/bin/kaggle kernels status "$kernel" 2>/dev/null) || state=STATUS_UNAVAILABLE
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" >> data/kaggle/d3-pilot-watch.log
  case "$state" in
    *KernelWorkerStatus.COMPLETE*|*KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL_ACKNOWLEDGED*) break ;;
  esac
  sleep 120
done
printf '%s\n' "$state" > data/kaggle/d3-pilot-terminal.txt
while ! .venv/bin/kaggle kernels output "$kernel" -p data/kaggle/d3-pilot-output --page-size 200 --file-pattern '^d3-pilot/.*' >> data/kaggle/d3-pilot-watch.log 2>&1; do
  sleep 120
done
mkdir -p data/kaggle/d3-pilot-output/d3-pilot
.venv/bin/kaggle quota >> data/kaggle/d3-pilot-watch.log 2>&1
PYTHONPATH=. .venv/bin/python scripts/d3_pilot_summary.py >> data/kaggle/d3-pilot-watch.log 2>&1
