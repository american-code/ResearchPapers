#!/usr/bin/env python3
"""One cell of the MLX quantization-audit matrix: all arms for a single model.

Idempotent by design. Every (arm, dataset) that already has an eval-results file is
skipped, so a task killed at hour three resumes instead of restarting. That matters
because the daemon retries tasks and the big models take hours.

Arms:
  bf16      original HF weights, no quantization — the baseline
  4bit-rtn  self-quantized with mlx_lm.convert (naive round-to-nearest, MLX default)
  4bit-awq  self-quantized with mlx_lm.awq   (activation-aware, calibrated)
  4bit-gptq self-quantized with mlx_lm.gptq  (Hessian-based, calibrated)
  4bit-dwq  self-quantized with mlx_lm.dwq   (distilled)

We quantize ourselves rather than pulling mlx-community uploads so that every arm is
provably the same base weights. A community quant of unknown provenance would confound
the one variable this whole study is trying to isolate.

Usage:
  run_matrix.py --model qwen3b [--arms bf16,4bit-rtn] [--datasets humaneval,mbpp,realtasks]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
RESULTS = HOME / "ResearchPapers/data/slm-benchmark/matrix"
QUANT_DIR = Path("/tmp/mlx_quants")          # ephemeral by design: weights are huge
RUNS = Path("/tmp/evalplus_runs")            # regenerable; results are copied to RESULTS
PROMPTS = Path("/tmp/evalplus_prompts")      # regenerated on demand, see ensure_prompts()
SELF_DIR = Path(__file__).resolve().parent
GEN = SELF_DIR / "evalplus_generate.py"
EVALPLUS = HOME / "evalplus-env/bin/evalplus.evaluate"
REAL_TASKS = SELF_DIR / "real_tasks.py"

# These functions are longer than HumanEval stubs; 768 truncated several mid-body,
# which surfaces as an unterminated code fence rather than an honest failure.
REALTASK_MAX_TOKENS = 1536

# bf16_gb is the safetensors footprint; anything near 32 GB cannot run bf16 on lab-02.
MODELS = {
    "qwen1.5b": {"hf": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "bf16_gb": 3.1},
    "qwen3b":   {"hf": "Qwen/Qwen2.5-Coder-3B-Instruct",   "bf16_gb": 6.2},
    "qwen7b":   {"hf": "Qwen/Qwen2.5-Coder-7B-Instruct",   "bf16_gb": 15.2},
    "phi4mini": {"hf": "microsoft/Phi-4-mini-instruct",    "bf16_gb": 7.7},
    "granite8b": {"hf": "ibm-granite/granite-8b-code-instruct-4k", "bf16_gb": 16.1},
    # 31.4 GB in bf16 — will not fit in 32 GB alongside the OS. 4-bit arms only;
    # excluded from the bf16-vs-4bit delta analysis, and that must be stated in results.
    "deepseek16b": {"hf": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                    "bf16_gb": 31.4, "no_bf16": True},
}

BF16_RAM_CEILING_GB = 24.0  # leave headroom for OS + eval subprocesses on a 32 GB box


def ensure_prompts() -> None:
    """Regenerate the EvalPlus prompt dump if /tmp was cleared since the last run."""
    if (PROMPTS / "humaneval.json").exists() and (PROMPTS / "mbpp.json").exists():
        return
    dumper = SELF_DIR / "dump_prompts.py"
    evalpy = HOME / "evalplus-env/bin/python"
    print(f"[setup] regenerating prompt dump at {PROMPTS}", flush=True)
    if subprocess.call([str(evalpy), str(dumper), str(PROMPTS)]) != 0:
        raise RuntimeError("failed to regenerate EvalPlus prompts")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list, **kw) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, **kw)


def resolve_arm(key: str, arm: str) -> str:
    """Return a model path for this arm, quantizing from the HF weights if needed."""
    hf = MODELS[key]["hf"]
    if arm == "bf16":
        return hf
    out = QUANT_DIR / f"{key}-{arm}"
    if (out / "config.json").exists():
        log(f"reusing existing quant at {out}")
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    method = arm.split("-", 1)[1]
    if method == "rtn":
        cmd = [sys.executable, "-m", "mlx_lm.convert", "--hf-path", hf,
               "--mlx-path", str(out), "-q", "--q-bits", "4"]
    else:
        # Calibrated quantizers ship as their own CLIs and fetch a calibration corpus.
        cmd = [sys.executable, "-m", f"mlx_lm.{method}", "--model", hf,
               "--mlx-path", str(out), "--bits", "4"]
    if sh(cmd) != 0:
        raise RuntimeError(f"quantization failed for {key}/{arm}")
    return str(out)


def eval_results_path(tag: str, ds: str) -> Path:
    return Path(f"/tmp/evalplus_runs/{tag}/{ds}-samples_eval_results.json")


def run_evalplus(model_path: str, tag: str, ds: str) -> dict:
    if eval_results_path(tag, ds).exists():
        log(f"SKIP {tag}/{ds} — already evaluated")
        return {"skipped": True}
    if sh([sys.executable, str(GEN), "--model", model_path, "--dataset", ds,
           "--tag", tag, "--out", "/tmp/evalplus_runs"]) != 0:
        raise RuntimeError(f"generation failed {tag}/{ds}")
    env = dict(os.environ, EVALPLUS_MAX_MEMORY_BYTES="-1")  # Darwin rejects RLIMIT_AS
    samples = f"/tmp/evalplus_runs/{tag}/{ds}-samples.jsonl"
    if sh([str(EVALPLUS), "--dataset", ds, "--samples", samples], env=env) != 0:
        raise RuntimeError(f"evaluation failed {tag}/{ds}")
    return {"skipped": False}


def run_real_tasks(model_path: str, tag: str) -> dict:
    """Run the 26-task domain suite. Uses the same greedy protocol as EvalPlus."""
    out = Path(f"/tmp/evalplus_runs/{tag}/realtasks.json")
    if out.exists():
        log(f"SKIP {tag}/realtasks — already run")
        return json.loads(out.read_text())

    sys.path.insert(0, str(REAL_TASKS.parent))
    sys.path.insert(0, str(GEN.parent))
    import real_tasks as RT
    # real_tasks.build_program executes the solution string verbatim — it does no
    # markdown-fence stripping. Passing raw model output makes every program start
    # with ```python and die at line 1, scoring a uniform 0.
    from evalplus_generate import extract_code
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.0)
    results, tps_all = [], []
    for task in RT.REAL_TASKS:
        user = f"Complete the following Python function:\n\n{task['prompt']}"
        try:
            chat = tokenizer.apply_chat_template(
                [{"role": "system", "content":
                  "You are an expert Python programmer. Write the complete function, "
                  "including the def line and any imports it needs. Return only code."},
                 {"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            chat = user
        chunks, tps = [], 0.0
        for r in stream_generate(model, tokenizer, chat, max_tokens=REALTASK_MAX_TOKENS, sampler=sampler):
            chunks.append(r.text)
            if r.generation_tps:
                tps = r.generation_tps
        tps_all.append(tps)
        # check_solution returns (ok, detail). Unpack it — bool() on the tuple is
        # always True, which silently scores every task as a pass.
        code = extract_code("".join(chunks))
        ok, detail = RT.check_solution(task["id"], code)
        results.append({"id": task["id"], "name": task["name"], "tier": task["tier"],
                        "trap_family": task["trap_family"], "passed": bool(ok),
                        "tps": round(tps, 1),
                        "detail": "" if ok else detail[:400]})
        log(f"  {task['id']} {task['tier']:6} {task['name']:26} "
            f"{'PASS' if ok else 'FAIL'}")

    by_tier = {}
    for t in ("EASY", "MEDIUM", "HARD"):
        sub = [r for r in results if r["tier"] == t]
        by_tier[t] = {"passed": sum(r["passed"] for r in sub), "total": len(sub)}
    payload = {"tag": tag, "model": model_path,
               "passed": sum(r["passed"] for r in results), "total": len(results),
               "by_tier": by_tier,
               "mean_tokens_per_sec": round(sum(tps_all) / max(len(tps_all), 1), 1),
               "detail": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log(f"realtasks {tag}: {payload['passed']}/{payload['total']} {by_tier}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--arms", default="bf16,4bit-rtn")
    ap.add_argument("--datasets", default="humaneval,mbpp,realtasks")
    ap.add_argument("--keep-weights", action="store_true")
    args = ap.parse_args()

    key = args.model
    spec = MODELS[key]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    ensure_prompts()

    if spec.get("no_bf16") and "bf16" in arms:
        log(f"EXCLUDING bf16 arm for {key}: {spec['bf16_gb']} GB exceeds the "
            f"{BF16_RAM_CEILING_GB} GB ceiling on this host. 4-bit arms only; this model "
            f"cannot contribute to the bf16-vs-4bit delta.")
        arms = [a for a in arms if a != "bf16"]

    summary = {"model": key, "hf": spec["hf"], "arms": {}}
    for arm in arms:
        tag = f"{key}-{arm}"
        log(f"=== ARM {tag} ===")
        try:
            path = resolve_arm(key, arm)
        except Exception as e:
            log(f"ARM FAILED {tag}: {e}")
            summary["arms"][arm] = {"error": str(e)}
            continue
        arm_res = {}
        for ds in datasets:
            try:
                if ds == "realtasks":
                    run_real_tasks(path, tag)
                else:
                    run_evalplus(path, tag, ds)
                arm_res[ds] = "ok"
            except Exception as e:
                log(f"FAILED {tag}/{ds}: {e}")
                arm_res[ds] = f"error: {e}"
        summary["arms"][arm] = arm_res
        # Free the SSD between arms, but never touch results.
        if not args.keep_weights and arm != "bf16":
            q = QUANT_DIR / f"{key}-{arm}"
            if q.exists():
                shutil.rmtree(q, ignore_errors=True)
                log(f"deleted quantized weights {q}")

    (RESULTS / f"{key}-summary.json").write_text(json.dumps(summary, indent=2))
    log(f"WROTE {RESULTS / f'{key}-summary.json'}")
    failed = [f"{a}/{d}" for a, r in summary["arms"].items()
              if isinstance(r, dict) for d, v in r.items() if str(v).startswith("error")]
    if failed:
        log(f"COMPLETED WITH FAILURES: {failed}")
        return 1
    log("ALL ARMS COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
