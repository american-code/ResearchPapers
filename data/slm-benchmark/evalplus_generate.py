#!/usr/bin/env python3
"""Generate EvalPlus samples with mlx-lm. Runs on system Python 3.9 (where mlx lives).

Emits samples.jsonl in EvalPlus's expected shape ({"task_id", "solution"}) plus a
sidecar metrics JSON with tokens/sec and peak memory, which EvalPlus does not track
but the quantization audit needs.

Scoring is a separate step:
  ~/evalplus-env/bin/evalplus.evaluate --dataset humaneval --samples <samples.jsonl>

Usage:
  evalplus_generate.py --model <mlx-id-or-path> --dataset humaneval|mbpp \
      --out <dir> [--limit N] [--tag NAME]
"""
import argparse
import ast
import json
import re
import time
from pathlib import Path

PROMPTS = Path("/tmp/evalplus_prompts")
MAX_GEN_TOKENS = 768

SYS_MSG = (
    "You are an expert Python programmer. "
    "Write the complete function, including the `def` line and any imports it needs. "
    "Return only code—no explanation."
)


def extract_code(response: str) -> str:
    """Return the first fenced block that actually parses as Python.

    Pattern order alone is not enough: the non-greedy patterns truncate any answer
    that itself contains a ``` delimiter, so a correct answer can become a
    SyntaxError. The greedy variants recover those and the ast check arbitrates.
    """
    texts = []
    # Every fenced block, individually and in order. A model that emits prose, then a
    # helper block, then the real answer needs each candidate tried separately — taking
    # only the first block or greedily spanning to the last both fail on that shape.
    blocks = [m.group(1).strip()
              for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)]
    texts.extend(blocks)
    # All blocks concatenated, for an answer split across fences.
    if len(blocks) > 1:
        texts.append("\n\n".join(blocks))
    # Greedy span, for an answer that itself contains a ``` delimiter.
    for pat in (r"```python\s*\n(.*)```", r"```\s*\n(.*)```"):
        m = re.search(pat, response, re.DOTALL)
        if m:
            texts.append(m.group(1).strip())
    # Unterminated fence: generation hit the token cap before the closing ```, so no
    # complete block exists. Strip the opening fence so the failure is reported as the
    # truncation it is, rather than as a bogus "```python is invalid syntax" at line 1.
    m = re.search(r"```(?:python)?\s*\n(.*)", response, re.DOTALL)
    if m and "```" not in m.group(1):
        texts.append(m.group(1).strip())
    texts.append(response.strip())

    parseable = []
    for t in texts:
        try:
            ast.parse(t)
        except SyntaxError:
            continue
        parseable.append(t)
    if parseable:
        # Prefer a candidate that actually defines something over a bare expression.
        for t in parseable:
            if re.search(r"^\s*(def|class|import|from)\s", t, re.M):
                return t
        return parseable[0]
    return texts[0] if texts else response.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--out", default="/tmp/evalplus_runs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--tag", default="", help="label for this config, e.g. bf16 / 4bit-rtn")
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    tasks = json.loads((PROMPTS / f"{args.dataset}.json").read_text())
    ids = sorted(tasks, key=lambda t: int(t.split("/")[1]))
    if args.limit:
        ids = ids[: args.limit]

    tag = args.tag or Path(args.model).name
    outdir = Path(args.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    samples_path = outdir / f"{args.dataset}-samples.jsonl"

    print(f"[{tag}] loading {args.model}", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model)
    load_sec = time.time() - t0

    sampler = make_sampler(temp=0.0)  # greedy; verified bit-reproducible on this host
    tps_list, peak_mem = [], 0.0

    with samples_path.open("w") as fh:
        for i, tid in enumerate(ids):
            t = tasks[tid]
            user = f"Complete the following Python function:\n\n{t['prompt']}"
            try:
                chat = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYS_MSG},
                     {"role": "user", "content": user}],
                    add_generation_prompt=True, tokenize=False)
            except Exception:
                chat = user

            chunks, tps = [], 0.0
            for resp in stream_generate(model, tokenizer, chat,
                                        max_tokens=MAX_GEN_TOKENS, sampler=sampler):
                chunks.append(resp.text)
                if resp.generation_tps and resp.generation_tps > 0:
                    tps = resp.generation_tps
                if resp.peak_memory:
                    peak_mem = max(peak_mem, resp.peak_memory)
            tps_list.append(tps)

            code = extract_code("".join(chunks))
            # EvalPlus accepts a full standalone solution; prepend the stub only when
            # the model omitted the def line entirely.
            if f"def {t['entry_point']}" not in code:
                code = t["prompt"] + "\n" + code
            fh.write(json.dumps({"task_id": tid, "solution": code}) + "\n")

            if (i + 1) % 25 == 0 or i == len(ids) - 1:
                mean = sum(tps_list) / max(len(tps_list), 1)
                print(f"  [{tag}/{args.dataset}] {i+1}/{len(ids)} "
                      f"tps={mean:.1f} peak={peak_mem:.2f}GB", flush=True)

    metrics = {
        "tag": tag, "model": args.model, "dataset": args.dataset,
        "n_tasks": len(ids), "load_sec": round(load_sec, 1),
        "mean_tokens_per_sec": round(sum(tps_list) / max(len(tps_list), 1), 1),
        "peak_memory_gb": round(peak_mem, 2),
        "max_gen_tokens": MAX_GEN_TOKENS, "sampler": "greedy temp=0.0", "n_samples": 1,
    }
    (outdir / f"{args.dataset}-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{tag}] wrote {samples_path}")
    print(f"[{tag}] {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
