"""Optional Nsight profiles (Nsight Compute and Nsight Systems) of the template kernels.

torch.profiler (make profile) is the primary profiler. Nsight is a bonus, because
under WSL2 the tools are frequently missing, or present but unable to collect
(Nsight Compute needs access to GPU performance counters, which WSL2 may not give).
So this runner never fails because of the tools: for each tool it records whether
it was found, whether it ran, and the end of its output, in results/nsight/STATUS.json.
A tool that did not produce a profile is reported as such; nothing is claimed from it.

Usage:
    python -m kernelgen.nsight --out results/nsight
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from kernelgen.benchmark import reference_candidates

TOOLS = ("ncu", "nsys")
TAIL_CHARS = 1500


def available_tools() -> dict[str, str | None]:
    """Path to each Nsight command line tool, or None if it is not installed."""
    return {tool: shutil.which(tool) for tool in TOOLS}


def kernel_names(source: str) -> list[str]:
    """Names of the @triton.jit functions in `source`, which are the kernel names in a profile."""
    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "jit":
                    names.append(node.name)
    return names


def build_command(tool: str, tool_path: str, out_base: Path, run_cmd: list[str], kernel: str | None) -> list[str]:
    """The profiler command line for `tool` wrapped around `run_cmd`."""
    # The unambiguous --flag=value spelling is used throughout. These command lines
    # could not be run where this was written (no Nsight tools), so they are unverified.
    if tool == "ncu":
        cmd = [tool_path, "--set=full", "--force-overwrite", f"--export={out_base}"]
        if kernel:
            cmd.append(f"--kernel-name=regex:{kernel}")
        return cmd + ["--launch-count=3"] + run_cmd
    if tool == "nsys":
        return [tool_path, "profile", "--force-overwrite=true", f"--output={out_base}", "--stats=false"] + run_cmd
    raise ValueError(f"unknown tool: {tool!r}")


def run_tool(tool: str, tool_path: str, out_base: Path, run_cmd: list[str], kernel: str | None, timeout_s: float) -> dict[str, Any]:
    cmd = build_command(tool, tool_path, out_base, run_cmd, kernel)
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        output = (proc.stdout + proc.stderr)[-TAIL_CHARS:]
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        output, returncode = f"timed out after {timeout_s:g}s", None
    produced = sorted(str(p) for p in out_base.parent.glob(out_base.name + ".*"))
    return {
        "tool": tool,
        "command": cmd,
        "returncode": returncode,
        "seconds": round(time.monotonic() - started, 1),
        "profile_files": produced,
        "produced_profile": returncode == 0 and bool(produced),
        "output_tail": output,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/nsight")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--all-families", action="store_true", help="one task per family instead of just the first")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tools = available_tools()
    status: dict[str, Any] = {"tools_found": tools, "runs": []}

    targets = reference_candidates(sample_only=True)
    if not args.all_families:
        targets = targets[:1]

    if not any(tools.values()):
        status["note"] = "neither ncu nor nsys was found on PATH; no Nsight profile was produced"
    for task, candidate_id, source in targets:
        names = kernel_names(source)
        run_cmd = [sys.executable, "-m", "kernelgen.run_once", "--family", task.family, "--params", json.dumps(task.params)]
        for tool, path in tools.items():
            if path is None:
                continue
            result = run_tool(tool, path, out / f"{task.name}.{tool}", run_cmd, names[0] if names else None, args.timeout)
            result["task"] = task.name
            status["runs"].append(result)
            print(f"{tool} on {task.name}: {'wrote ' + ', '.join(result['profile_files']) if result['produced_profile'] else 'no profile produced (see STATUS.json)'}")

    (out / "STATUS.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    if "note" in status:
        print(status["note"])
    print(f"wrote {out / 'STATUS.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
