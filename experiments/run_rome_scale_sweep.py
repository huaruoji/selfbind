#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MODEL = "/home/yuno/.cache/modelscope/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
SCALES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def main():
    rows = []
    for scale in SCALES:
        out = ROOT / "data" / f"rome_scale_{str(scale).replace('.', 'p')}.json"
        cmd = [
            PY,
            str(ROOT / "experiments" / "rome_identity_baseline.py"),
            "--model", MODEL,
            "--layer", "20",
            "--n-ctx", "30",
            "--no-cov",
            "--edit-scale", str(scale),
            "--clean-limit", "3",
            "--contam-limit", "6",
            "--output", str(out),
        ]
        print("[run]", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if result.returncode == 0 else "fail"
        row = {"scale": scale, "status": status, "output": str(out)}
        if out.exists():
            try:
                payload = json.loads(out.read_text())
                row["clean_hits"] = payload["post_rome"]["summary"]["n_clean_ok"]
                row["contam_hits"] = payload["post_rome"]["summary"]["n_contam_ok"]
            except Exception:
                pass
        if result.returncode != 0:
            row["stderr_tail"] = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-20:])
        rows.append(row)
        print(row)
    report = ROOT / "data" / "rome_scale_sweep_report.json"
    report.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"-> {report}")


if __name__ == "__main__":
    main()
