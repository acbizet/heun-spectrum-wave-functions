#!/usr/bin/env python3
"""Run one or all numerical experiments in this repository."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = {
    "basic": ROOT / "experiments/01_basic_shooting/heun_shooting.py",
    "large-t": ROOT / "experiments/02_large_t_regions/heun_large_t_shooting.py",
    "t10000": ROOT / "experiments/03_t10000_mode_matching/mode_matching_t10000.py",
    "first-correction": ROOT / "experiments/04_first_correction/heun_first_correction_test.py",
}

def run(script: Path) -> None:
    print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case", nargs="?", choices=["all", *CASES], default="all")
    args = p.parse_args()
    if args.case == "all":
        for script in CASES.values():
            run(script)
    else:
        run(CASES[args.case])
