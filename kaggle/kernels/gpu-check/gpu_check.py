"""xbrl-frontier: confirm the account gets a T4 with internet on (T-033, T-035). No secrets, no data."""
import json
import platform
import subprocess
import time
import urllib.request

out = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version()}
try:
    out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip()
except Exception as e:
    out["nvidia_smi"] = f"error: {e}"
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    out["cuda"] = torch.version.cuda
    out["device_count"] = torch.cuda.device_count()
except Exception as e:
    out["torch"] = f"error: {e}"
for name, url in (("huggingface", "https://huggingface.co/api/models/Qwen/Qwen3-0.6B"), ("pypi", "https://pypi.org/pypi/vllm/json"),
                  ("sec", "https://www.sec.gov/robots.txt")):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "xbrl-frontier gpu-check"})
        with urllib.request.urlopen(req, timeout=30) as r:
            out[f"internet_{name}"] = r.status
    except Exception as e:
        out[f"internet_{name}"] = f"error: {e}"
print(json.dumps(out, indent=1))
open("/kaggle/working/gpu_check.json", "w").write(json.dumps(out, indent=1))
