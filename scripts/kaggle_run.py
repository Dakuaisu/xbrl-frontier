"""Builder-run Kaggle steps (T-033): datasets, notebook kernels, status, outputs and GPU-hour accounting.

  dataset  --slug S --dir D [--message M]     create the private dataset adnanrashid29/S from D, or add a version
  push     --name N --notebook P [--param-cell CODE] [--dataset S ...] [--kernel-source K ...] [--timeout SEC]
           builds kernel adnanrashid29/xbrl-frontier-N from notebook P (the first code cell replaced by CODE when given),
           T4 accelerator, internet on, private; logs the push with the GPU quota used before it
  wait     --name N [--minutes M]             polls `kaggle kernels status` until complete/error/cancel or M minutes
  fetch    --name N --to DIR                  downloads the outputs and logs the GPU quota used after (GPU-hours = delta)
Every action appends a row to data/kaggle/runs.jsonl. The token comes from KAGGLE_API_TOKEN and is never printed.
"""
import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from xbrl_frontier import config

OWNER = "adnanrashid29"
KAGGLE = str(config.REPO_ROOT / ".venv" / "bin" / "kaggle")
RUNS = config.REPO_ROOT / "data" / "kaggle" / "runs.jsonl"


def sh(*args: str) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed: {out.stdout[-500:]} {out.stderr[-500:]}")
    return out.stdout


def gpu_used_hours() -> float:
    for line in sh("quota").splitlines():
        if line.startswith("GPU"):
            return float(re.findall(r"([\d.]+)h", line)[0])
    raise SystemExit("could not read the GPU quota")


def log(row: dict) -> None:
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS, "a") as f:
        f.write(json.dumps({"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **row}) + "\n")


def dataset(slug: str, d: Path, message: str) -> None:
    meta = {"title": slug, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{OWNER}/{slug}"], capture_output=True, text=True).returncode == 0
    out = sh("datasets", "version", "-p", str(d), "-m", message, "-r", "skip") if exists else sh("datasets", "create", "-p", str(d), "-r", "skip")
    log({"action": "dataset", "slug": slug, "version": exists, "files": sorted(p.name for p in d.iterdir()), "out": out.strip()[-300:]})
    print(out.strip())


def push(name: str, notebook: Path, param_cell: str | None, datasets: list[str], kernels: list[str], timeout: int | None) -> None:
    nb = json.loads(notebook.read_text())
    if param_cell is not None:
        cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
        cell["source"] = param_cell.splitlines(keepends=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "notebook.ipynb").write_text(json.dumps(nb))
        meta = {"id": f"{OWNER}/xbrl-frontier-{name}", "title": f"xbrl-frontier-{name}", "code_file": "notebook.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
                "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [f"{OWNER}/{s}" for s in datasets], "competition_sources": [],
                "kernel_sources": [f"{OWNER}/xbrl-frontier-{k}" for k in kernels], "model_sources": []}
        (t / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        before = gpu_used_hours()
        args = ["kernels", "push", "-p", str(t), "--accelerator", "NvidiaTeslaT4"]
        if timeout:
            args += ["-t", str(timeout)]
        out = sh(*args)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.REPO_ROOT).stdout.strip()
    log({"action": "push", "name": name, "kernel": meta["id"], "notebook": str(notebook.resolve().relative_to(config.REPO_ROOT)),
         "param_cell": param_cell, "datasets": datasets, "kernel_sources": kernels, "git_head": git,
         "gpu_used_hours_before": before, "out": out.strip()[-300:]})
    print(out.strip())


def status(name: str) -> str:
    return sh("kernels", "status", f"{OWNER}/xbrl-frontier-{name}").strip()


def wait(name: str, minutes: float) -> str:
    import time
    end = time.time() + minutes * 60
    s = status(name)
    while not re.search(r"COMPLETE|ERROR|CANCEL", s, re.I) and time.time() < end:
        time.sleep(30)
        s = status(name)
    print(s)
    return s


def fetch(name: str, to: Path, pattern: str | None = None) -> None:
    to.mkdir(parents=True, exist_ok=True)
    out = sh("kernels", "output", f"{OWNER}/xbrl-frontier-{name}", "-p", str(to), *(["--file-pattern", pattern] if pattern else []))
    pushes = [json.loads(l) for l in open(RUNS) if json.loads(l).get("action") == "push" and json.loads(l)["name"] == name]
    after = gpu_used_hours()
    before = pushes[-1]["gpu_used_hours_before"] if pushes else None
    row = {"action": "fetch", "name": name, "status": status(name), "to": str(to.resolve().relative_to(config.REPO_ROOT)),
           "gpu_used_hours_after": after, "gpu_hours_this_run": None if before is None else round(after - before, 2),
           "files": sorted(str(p.relative_to(to)) for p in to.rglob("*") if p.is_file())[:50]}
    log(row)
    print(json.dumps(row, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["dataset", "push", "wait", "fetch", "status"])
    ap.add_argument("--slug")
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--message", default="update")
    ap.add_argument("--name")
    ap.add_argument("--notebook", type=Path)
    ap.add_argument("--param-cell")
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--kernel-source", action="append", default=[])
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--minutes", type=float, default=110)
    ap.add_argument("--to", type=Path)
    ap.add_argument("--pattern", help="regex passed to --file-pattern; skip multi-GB weights, which hang the CLI download")
    a = ap.parse_args()
    if a.action == "dataset":
        dataset(a.slug, a.dir, a.message)
    elif a.action == "push":
        push(a.name, a.notebook, a.param_cell, a.dataset, a.kernel_source, a.timeout)
    elif a.action == "wait":
        wait(a.name, a.minutes)
    elif a.action == "status":
        print(status(a.name))
    else:
        fetch(a.name, a.to, a.pattern)


if __name__ == "__main__":
    main()
