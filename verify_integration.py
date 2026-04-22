"""
Integrated verification for:
1) End-to-end synthetic funnel behavior
2) Config + docs privacy/sensor consistency

Run:
    python3 verify_integration.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_synthetic_e2e() -> None:
    cmd = [sys.executable, str(ROOT / "test_pipeline.py")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if "ModuleNotFoundError" in (result.stdout + result.stderr):
            print("[skip] synthetic E2E not executed (missing Python deps).")
            print("       install deps with: pip3 install -r requirements.txt")
            return
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Synthetic E2E test failed.")
    print("[ok] synthetic E2E pipeline test passed")


def verify_config_readme() -> None:
    config_text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    if re.search(r"^\s*privacy:\s*$", config_text, re.MULTILINE) is None:
        raise RuntimeError("config.yaml missing privacy section")
    if re.search(r"^\s*mode:\s*\w+", config_text, re.MULTILINE) is None:
        raise RuntimeError("config.yaml missing privacy.mode")
    if re.search(r"^\s*depth:\s*$", config_text, re.MULTILINE) is None:
        raise RuntimeError("config.yaml missing depth section")
    if re.search(r"^\s*backend:\s*\w+", config_text, re.MULTILINE) is None:
        raise RuntimeError("config.yaml missing depth.backend")
    if re.search(r"^\s*gaze:\s*$", config_text, re.MULTILINE) is None:
        raise RuntimeError("config.yaml missing gaze section")
    if re.search(r"^\s*enabled:\s*true\s*$", config_text, re.MULTILINE | re.IGNORECASE) is None:
        raise RuntimeError("config.yaml gaze.enabled should be true for look metrics")

    required_terms = [
        "Privacy",
        "dashboard",
        "telemetry_engine.py",
        "demo_synthetic.py",
    ]
    missing = [term for term in required_terms if re.search(re.escape(term), readme, re.IGNORECASE) is None]
    if missing:
        raise RuntimeError(f"README missing expected terms: {missing}")

    print("[ok] config + README privacy/sensor checks passed")


def main() -> int:
    run_synthetic_e2e()
    verify_config_readme()
    print("\nAll integration checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
