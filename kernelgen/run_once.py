"""Run one candidate kernel a few times, for an external profiler to attach to.

Nsight Compute (ncu) and Nsight Systems (nsys) profile a whole command, so this is
the command: it builds the task, loads the candidate, warms it up, and runs it. It
applies the same candidate policy as the benchmark before running anything.

Usage (normally started by kernelgen.nsight):
    python -m kernelgen.run_once --family layernorm --params '{"M": 4096, "N": 2048}'
    python -m kernelgen.run_once --family softmax --params '{"M": 1024, "N": 4096}' --source kernel.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

from kernelgen.candidate import load_candidate
from kernelgen.families import get_family
from kernelgen.policy import check_source
from kernelgen.tasks import TIMING_SEED


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--family", required=True)
    parser.add_argument("--params", required=True, help="shape parameters as JSON, e.g. '{\"M\": 4096, \"N\": 2048}'")
    parser.add_argument("--source", default=None, help="candidate file; default is the family's template")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args(argv)

    fam = get_family(args.family)
    params = json.loads(args.params)
    source = Path(args.source).read_text() if args.source else fam.render(params, fam.default_config(params))
    violations = check_source(source)
    if violations:
        print("candidate rejected by the policy: " + "; ".join(violations), file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = fam.task(params)
    with load_candidate(source) as fn:
        for i in range(args.warmup + args.iters):
            inputs = task.make_inputs(TIMING_SEED + i, device)
            fn(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
    print(f"ran {task.name} on {device}: {args.warmup} warmup + {args.iters} measured calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
