#!/usr/bin/env python3
"""Dump EvalPlus prompts to plain JSON so the mlx generator (Python 3.9) can read them.

EvalPlus lives in its own 3.14 venv because mlx-lm is installed against the system
3.9. Generation and evaluation are decoupled in EvalPlus by design, so the only thing
that has to cross the version boundary is prompt text in, samples.jsonl out.

Run with: ~/evalplus-env/bin/python dump_prompts.py <outdir>
"""
import json
import sys
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/evalplus_prompts")
out.mkdir(parents=True, exist_ok=True)

for name, loader in [("humaneval", get_human_eval_plus), ("mbpp", get_mbpp_plus)]:
    ds = loader()
    slim = {
        tid: {"prompt": d["prompt"], "entry_point": d["entry_point"]}
        for tid, d in ds.items()
    }
    path = out / f"{name}.json"
    path.write_text(json.dumps(slim, indent=1))
    print(f"{name}: {len(slim)} tasks -> {path}")
