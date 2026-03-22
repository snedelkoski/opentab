#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
autoimprove eval harness — runs all evaluators, computes composite score.
DO NOT MODIFY after baseline is established.

Usage: uv run .autoimprove/eval_harness.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def run_evaluator(script: Path, cwd: Path, timeout: int = 300, log_file=None) -> dict:
    """Run an evaluator, streaming stderr to terminal and log file."""
    try:
        # Use Popen to stream stderr in real-time while capturing stdout for JSON
        proc = subprocess.Popen(
            ["uv", "run", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
        )

        # Read stderr line by line and forward to terminal + log
        stderr_lines = []
        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_lines.append(line)
                if log_file:
                    log_file.write(line)
                    log_file.flush()

        stdout, remaining_stderr = proc.stdout.read(), proc.stderr.read()
        if remaining_stderr:
            sys.stderr.write(remaining_stderr)
            stderr_lines.append(remaining_stderr)
            if log_file:
                log_file.write(remaining_stderr)
                log_file.flush()

        if proc.returncode != 0:
            return {
                "name": script.stem,
                "score": 0.0,
                "details": {
                    "error": f"exit {proc.returncode}",
                    "stderr": "".join(stderr_lines)[-2000:],
                },
            }
        for line in reversed(stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                return json.loads(line)
        return {
            "name": script.stem,
            "score": 0.0,
            "details": {"error": "no JSON output", "stdout": stdout[-500:]},
        }
    except subprocess.TimeoutExpired:
        proc.kill()
        return {
            "name": script.stem,
            "score": 0.0,
            "details": {"error": f"timeout after {timeout}s"},
        }
    except Exception as e:
        return {"name": script.stem, "score": 0.0, "details": {"error": str(e)}}


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent
    evaluators_dir = here / "evaluators"
    log_path = here / "run.log"

    # Load weights from config if it exists
    config_path = here / "config.yaml"
    weights = {}
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            for e in cfg.get("evaluators", []):
                weights[e["name"]] = e.get("weight", 1.0)
        except ImportError:
            # Parse YAML manually for simple structure
            with open(config_path) as f:
                content = f.read()
            # Simple weight extraction
            import re

            for match in re.finditer(
                r'name:\s*"?(\w+)"?\s*\n.*?weight:\s*([\d.]+)', content, re.DOTALL
            ):
                weights[match.group(1)] = float(match.group(2))

    t0 = time.time()
    results = []

    # Open log file for the entire run
    with open(log_path, "w") as log_file:
        for script in sorted(evaluators_dir.glob("*.py")):
            if script.name.startswith("_"):
                continue
            header = f"=== evaluator: {script.name} ===\n"
            print(header, end="", file=sys.stderr)
            log_file.write(header)
            log_file.flush()

            r = run_evaluator(script, repo, log_file=log_file)
            r["weight"] = weights.get(r["name"], 1.0)
            results.append(r)

            footer = f"--- {r['name']}: score={r['score']} ---\n\n"
            print(footer, end="", file=sys.stderr)
            log_file.write(footer)
            log_file.flush()

        total_w = sum(r["weight"] for r in results)
        composite = sum(r["score"] * r["weight"] for r in results) / total_w if total_w else 0.0

        output = json.dumps(
            {
                "composite_score": round(composite, 6),
                "elapsed_seconds": round(time.time() - t0, 1),
                "evaluators": results,
            },
            indent=2,
        )
        print(output)

        # Append final JSON to the log
        log_file.write("\n=== result ===\n")
        log_file.write(output + "\n")


if __name__ == "__main__":
    main()
