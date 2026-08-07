#!/usr/bin/env python3
"""
MLX quantization audit — matrix runner.

Usage:
  python3 run_matrix.py --model qwen3b --arms bf16,4bit-rtn \\
      --datasets humaneval,mbpp,realtasks

Idempotent: skips any (arm, dataset) cell whose eval-results marker already
exists before generating or evaluating anything.

File layout (per arm):
  /tmp/evalplus_runs/<model>-<arm>/<dataset>-samples.jsonl        (generation)
  /tmp/evalplus_runs/<model>-<arm>/<dataset>-samples_eval_results.json  (marker: he/mbpp)
  /tmp/evalplus_runs/<model>-<arm>/<dataset>-metrics.json         (tps, memory)
  /tmp/evalplus_runs/<model>-<arm>/realtasks.json                 (marker: realtasks)
  /tmp/evalplus_runs/<model>-<arm>/realtasks-metrics.json

Summary:
  <script_dir>/matrix/<model>-summary.json

Sentinel (written on full completion):
  /tmp/evalplus_runs/<model>.done
"""

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SLMBENCH_DIR = Path(__file__).parent
MATRIX_DIR = SLMBENCH_DIR / "matrix"
EVALPLUS_RUNS_BASE = Path("/tmp/evalplus_runs")
EVALPLUS_PROMPTS = Path("/tmp/evalplus_prompts")
EVALPLUS_VENV_PYTHON = Path.home() / "evalplus-env" / "bin" / "python"

MAX_GEN_TOKENS = 768
SYS_MSG = (
    "You are an expert Python programmer. "
    "Write the complete function, including the `def` line and any imports it needs. "
    "Return only code—no explanation."
)

# ── Model registry ────────────────────────────────────────────────────────────
# `mlx_id` is deliberately None on every arm.
#
# Pointing an arm at an mlx-community upload makes the comparison rest on an artifact
# of unknown provenance: we cannot verify which base revision it was converted from or
# with what settings. With mlx_id None, bf16 loads the HF weights directly and 4-bit is
# produced from those same weights by mlx_lm.convert here, so the pair differs in
# exactly one variable — which is the entire point of the audit.
#
# qwen3b was run before this was corrected and used community uploads for both arms;
# its numbers are sound but not provenance-matched, and it should be re-run for
# consistency before the results are written up.
def _pair(hf_id: str) -> dict:
    return {
        "bf16":     {"mlx_id": None, "hf_id": hf_id, "q_bits": None},
        "4bit-rtn": {"mlx_id": None, "hf_id": hf_id, "q_bits": 4},
    }


MODEL_REGISTRY = {
    "qwen1.5b": {
        "display_name": "Qwen2.5-Coder-1.5B-Instruct",
        "hf": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "param_b": 1.5,
        "arms": _pair("Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    },
    "qwen3b": {
        "display_name": "Qwen2.5-Coder-3B-Instruct",
        "hf": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "param_b": 3.0,
        "arms": _pair("Qwen/Qwen2.5-Coder-3B-Instruct"),
    },
    "qwen7b": {
        "display_name": "Qwen2.5-Coder-7B-Instruct",
        "hf": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "param_b": 7.0,
        "arms": _pair("Qwen/Qwen2.5-Coder-7B-Instruct"),
    },
    "phi4mini": {
        "display_name": "Phi-4-mini-instruct",
        "hf": "microsoft/Phi-4-mini-instruct",
        "param_b": 3.8,
        "arms": _pair("microsoft/Phi-4-mini-instruct"),
    },
    "granite8b": {
        # The 4k variant specifically — the 128k variant has a different published
        # baseline and an earlier sweep silently substituted it.
        "display_name": "Granite-8B-Code-Instruct-4k",
        "hf": "ibm-granite/granite-8b-code-instruct-4k",
        "param_b": 8.0,
        "arms": _pair("ibm-granite/granite-8b-code-instruct-4k"),
    },
    "deepseek16b": {
        # 31.4 GB in bf16 does not fit in 32 GB alongside the OS, so there is no bf16
        # arm at all. This model cannot contribute to the bf16-vs-4bit delta and must
        # be reported as 4-bit-only rather than quietly folded in.
        "display_name": "DeepSeek-Coder-V2-Lite-Instruct",
        "hf": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "param_b": 16.0,
        "bf16_excluded": "31.4 GB bf16 exceeds the 32 GB host; 4-bit arms only",
        "arms": {
            "4bit-rtn": {"mlx_id": None,
                         "hf_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                         "q_bits": 4},
        },
    },
}

# ── Global log file handle ────────────────────────────────────────────────────
_log_fh = None

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ── Code extraction (AST-validated, matches evalplus_generate.py) ─────────────
def extract_code(response: str) -> str:
    """Return first fenced block that parses as Python, with fallback chain."""
    texts = []
    blocks = [m.group(1).strip()
              for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)]
    texts.extend(blocks)
    if len(blocks) > 1:
        texts.append("\n\n".join(blocks))
    for pat in (r"```python\s*\n(.*)```", r"```\s*\n(.*)```"):
        m = re.search(pat, response, re.DOTALL)
        if m:
            texts.append(m.group(1).strip())
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
        for t in parseable:
            if re.search(r"^\s*(def|class|import|from)\s", t, re.M):
                return t
        return parseable[0]
    return texts[0] if texts else response.strip()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_arm(arm_cfg: dict):
    """Load model for the given arm. Returns (model, tokenizer)."""
    from mlx_lm import load as mlx_load

    mlx_id = arm_cfg.get("mlx_id")
    hf_id = arm_cfg["hf_id"]
    q_bits = arm_cfg.get("q_bits")

    if mlx_id:
        log(f"  Trying mlx-community: {mlx_id}")
        try:
            return mlx_load(mlx_id)
        except Exception as e:
            log(f"  mlx-community load failed ({e}), falling back")

    if q_bits:
        local_path = EVALPLUS_RUNS_BASE / f"_cvt_{hf_id.replace('/', '_')}_{q_bits}bit"
        if local_path.exists():
            log(f"  Loading existing converted model at {local_path}")
        else:
            log(f"  Converting {hf_id} → {q_bits}-bit RTN")
            local_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "mlx_lm", "convert",
                 "--hf-path", hf_id, "--mlx-path", str(local_path),
                 "--quantize", "--q-bits", str(q_bits)],
                check=True, timeout=7200,
            )
        return mlx_load(str(local_path))

    log(f"  Loading {hf_id} in native precision (bf16)")
    return mlx_load(hf_id)


# ── Generation (HumanEval / MBPP) ─────────────────────────────────────────────
def generate_samples(model, tokenizer, dataset: str, arm_dir: Path) -> tuple[Path, dict]:
    """Generate samples for humaneval or mbpp. Returns (samples_file, metrics)."""
    prompts_file = EVALPLUS_PROMPTS / f"{dataset}.json"
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Missing prompt file: {prompts_file}\n"
            "Run: python3 -c \"from evalplus.data import get_human_eval_plus, write_jsonl; "
            "write_jsonl('/tmp/evalplus_prompts/humaneval.json', get_human_eval_plus())\" "
            "(or equivalent) to regenerate."
        )

    tasks = json.loads(prompts_file.read_text())
    task_ids = sorted(tasks, key=lambda t: int(t.split("/")[1]))
    samples_file = arm_dir / f"{dataset}-samples.jsonl"

    tps_list, peak_mem = [], 0.0
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    with samples_file.open("w") as fh:
        for i, tid in enumerate(task_ids):
            t = tasks[tid]
            user = f"Complete the following Python function:\n\n{t['prompt']}"
            try:
                chat = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYS_MSG},
                     {"role": "user", "content": user}],
                    add_generation_prompt=True, tokenize=False,
                )
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
            if f"def {t['entry_point']}" not in code:
                code = t["prompt"] + "\n" + code
            fh.write(json.dumps({"task_id": tid, "solution": code}) + "\n")

            if (i + 1) % 25 == 0 or i == len(task_ids) - 1:
                mean_tps = sum(tps_list) / max(len(tps_list), 1)
                log(f"    [{dataset}] {i+1}/{len(task_ids)} "
                    f"tps={mean_tps:.1f} peak={peak_mem:.2f}GB")

    mean_tps = sum(tps_list) / max(len(tps_list), 1)
    metrics = {
        "dataset": dataset,
        "n_tasks": len(task_ids),
        "mean_tokens_per_sec": round(mean_tps, 1),
        "peak_memory_gb": round(peak_mem, 2),
        "max_gen_tokens": MAX_GEN_TOKENS,
        "sampler": "greedy temp=0.0",
    }
    (arm_dir / f"{dataset}-metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"  {dataset}: {len(task_ids)} samples written → {samples_file}")
    return samples_file, metrics


# ── EvalPlus evaluation ───────────────────────────────────────────────────────
def evalplus_evaluate(dataset: str, samples_file: Path) -> dict:
    """Run evalplus.evaluate; returns parsed pass@1 result."""
    env = {**os.environ, "EVALPLUS_MAX_MEMORY_BYTES": "-1"}
    venv_python = EVALPLUS_VENV_PYTHON

    if not venv_python.exists():
        # System python fallback (evalplus may be installed globally)
        venv_python = Path(sys.executable)
        log(f"  WARNING: ~/evalplus-env not found, using {venv_python}")

    log(f"  Running evalplus.evaluate on {samples_file.name}...")
    result = subprocess.run(
        [str(venv_python), "-m", "evalplus.evaluate",
         "--dataset", dataset,
         "--samples", str(samples_file)],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    if result.returncode != 0:
        log(f"  evalplus evaluate stderr (tail):\n{result.stderr[-800:]}")
        raise RuntimeError(f"evalplus.evaluate failed (rc={result.returncode})")

    # evalplus writes <stem>_eval_results.json next to the samples file
    results_file = samples_file.parent / (samples_file.stem + "_eval_results.json")
    if not results_file.exists():
        raise FileNotFoundError(f"Expected results file at {results_file}")

    return score_results(dataset, results_file)


def score_results(dataset: str, results_file: Path) -> dict:
    """Score a dataset from EvalPlus's per-problem verdicts file."""
    data = json.loads(results_file.read_text())

    # Score from EvalPlus's per-problem verdicts.
    #
    # This previously read `v.get("base", [False])[0]` behind an `isinstance(v, dict)`
    # test. Both are wrong for this format: each value is a LIST of records, so the
    # isinstance test excluded every entry, and the field is "base_status", not "base".
    # It scored 0/164 and recorded 0.0 — and because 0.0 is not None, the "could not
    # parse" warning never fired. qwen3b was written to disk as pass@1 = 0.0 while its
    # real score, sitting in this very file, was 0.841.
    #
    # A scoring function must never invent a number it could not compute. Anything
    # unparseable raises now.
    evals = data.get("eval", data)
    if not isinstance(evals, dict) or not evals:
        raise RuntimeError(
            f"cannot score {dataset}: no per-problem verdicts in {results_file} "
            f"(top-level keys: {list(data)[:10]})")

    n = len(evals)
    base_passed = plus_passed = 0
    unparsed = 0
    for recs in evals.values():
        rec = recs[0] if isinstance(recs, list) and recs else recs
        if not isinstance(rec, dict) or "base_status" not in rec:
            unparsed += 1
            continue
        if rec.get("base_status") == "pass":
            base_passed += 1
        if rec.get("plus_status") == "pass":
            plus_passed += 1

    if unparsed:
        raise RuntimeError(
            f"cannot score {dataset}: {unparsed}/{n} records lacked a base_status "
            f"field in {results_file}; refusing to report a partial rate")

    pass_at_1 = base_passed / n
    plus_at_1 = plus_passed / n

    # A real coding model scoring zero on a whole benchmark means the harness broke,
    # not that the model failed every problem. Fail loudly rather than banking a zero.
    if base_passed == 0 and n >= 20:
        raise RuntimeError(
            f"{dataset} scored 0/{n} — treating as a harness fault, not a result. "
            f"Inspect {results_file}.")

    log(f"  {dataset} pass@1 = {pass_at_1:.4f} ({base_passed}/{n})  "
        f"plus = {plus_at_1:.4f} ({plus_passed}/{n})")
    return {"pass_at_1": round(pass_at_1, 4),
            "plus_pass_at_1": round(plus_at_1, 4),
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "total": n,
            "results_file": str(results_file)}


# ── Real-tasks evaluation ─────────────────────────────────────────────────────
def run_realtasks_cell(model, tokenizer, arm_dir: Path) -> dict:
    """Evaluate model on real_tasks.py benchmark. Saves realtasks.json."""
    sys.path.insert(0, str(SLMBENCH_DIR))
    try:
        from real_tasks import REAL_TASKS, check_solution
    except ImportError as e:
        log(f"  ERROR: cannot import real_tasks: {e}")
        return {"error": str(e)}

    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    task_results, tps_list, peak_mem = [], [], 0.0

    for task in REAL_TASKS:
        user = f"Complete the following Python function:\n\n{task['prompt']}"
        try:
            chat = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYS_MSG},
                 {"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False,
            )
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

        solution = extract_code("".join(chunks))
        ok, detail = check_solution(task["id"], solution)
        task_results.append({
            "id": task["id"], "tier": task["tier"], "name": task["name"],
            "passed": ok, "tps": round(tps, 1),
            "detail": detail if not ok else "",
        })
        log(f"    {task['id']} [{task['tier']}] {task['name']}: "
            f"{'PASS' if ok else 'FAIL'} @ {tps:.1f} tps")

    n_passed = sum(r["passed"] for r in task_results)
    n_total = len(REAL_TASKS)
    by_tier = {}
    for r in task_results:
        t = r["tier"]
        by_tier.setdefault(t, {"passed": 0, "total": 0})
        by_tier[t]["total"] += 1
        by_tier[t]["passed"] += r["passed"]

    log(f"  Real tasks: {n_passed}/{n_total} passed")
    for tier, counts in by_tier.items():
        log(f"    {tier}: {counts['passed']}/{counts['total']}")

    result = {
        "pass_rate": round(n_passed / n_total, 4),
        "passed": n_passed,
        "total": n_total,
        "mean_tps": round(float(np.mean(tps_list)), 1),
        "peak_memory_gb": round(peak_mem, 2),
        "by_tier": by_tier,
        "detail": task_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (arm_dir / "realtasks.json").write_text(json.dumps(result, indent=2))
    (arm_dir / "realtasks-metrics.json").write_text(json.dumps({
        "pass_rate": result["pass_rate"],
        "passed": n_passed, "total": n_total,
        "mean_tokens_per_sec": result["mean_tps"],
        "peak_memory_gb": result["peak_memory_gb"],
        "by_tier": by_tier,
    }, indent=2))
    return result


# ── Cell idempotency marker ───────────────────────────────────────────────────
def done_marker(arm_dir: Path, dataset: str) -> Path:
    if dataset == "realtasks":
        return arm_dir / "realtasks.json"
    return arm_dir / f"{dataset}-samples_eval_results.json"


def cached_cell_summary(arm_dir: Path, dataset: str) -> dict:
    """Rebuild a cell's summary from an already-complete run.

    The resume path used to store the marker file's raw contents, which for
    HumanEval/MBPP is EvalPlus's 100 KB of per-problem verdicts — not a score. Any
    resumed run therefore wrote a summary with no usable pass@1 in it. Score the
    verdicts here instead, through the same function the fresh path uses.
    """
    marker = done_marker(arm_dir, dataset)
    if dataset == "realtasks":
        return json.loads(marker.read_text())
    out = score_results(dataset, marker)
    metrics_file = arm_dir / f"{dataset}-metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        out["mean_tps"] = m.get("mean_tokens_per_sec")
        out["peak_memory_gb"] = m.get("peak_memory_gb")
    return out


# ── Per-arm orchestrator ──────────────────────────────────────────────────────
def run_arm(model_key: str, arm_key: str, arm_cfg: dict, datasets: list) -> dict:
    arm_dir = EVALPLUS_RUNS_BASE / f"{model_key}-{arm_key}"
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Skip entire arm if all cells are done
    if all(done_marker(arm_dir, ds).exists() for ds in datasets):
        log(f"\nSKIP arm={arm_key} — all {len(datasets)} cells already complete")
        return _read_arm_summary(arm_dir, datasets)

    log(f"\n{'='*64}")
    log(f"ARM: {arm_key}")
    log(f"{'='*64}")

    import mlx.core as mx

    t_load = time.time()
    model, tokenizer = load_arm(arm_cfg)
    mx.eval(model.parameters())
    load_sec = round(time.time() - t_load, 1)
    log(f"  Model loaded in {load_sec}s")

    arm_summary = {}

    for dataset in datasets:
        marker = done_marker(arm_dir, dataset)
        if marker.exists():
            log(f"  SKIP {dataset} — {marker.name} exists")
            arm_summary[dataset] = cached_cell_summary(arm_dir, dataset)
            continue

        log(f"\n  ── Cell: arm={arm_key}  dataset={dataset} ──")
        t_cell = time.time()
        try:
            if dataset == "realtasks":
                cell_result = run_realtasks_cell(model, tokenizer, arm_dir)
            else:
                samples_file, gen_metrics = generate_samples(model, tokenizer, dataset, arm_dir)
                ep_result = evalplus_evaluate(dataset, samples_file)
                cell_result = {
                    "pass_at_1": ep_result["pass_at_1"],
                    "plus_pass_at_1": ep_result.get("plus_pass_at_1"),
                    "base_passed": ep_result.get("base_passed"),
                    "plus_passed": ep_result.get("plus_passed"),
                    "total": ep_result.get("total"),
                    "mean_tps": gen_metrics["mean_tokens_per_sec"],
                    "peak_memory_gb": gen_metrics["peak_memory_gb"],
                    "results_file": ep_result["results_file"],
                }
        except Exception as exc:
            import traceback
            log(f"  ERROR in {dataset}: {exc}")
            traceback.print_exc()
            cell_result = {"error": str(exc)}

        cell_result["elapsed_sec"] = round(time.time() - t_cell, 1)
        arm_summary[dataset] = cell_result

    log(f"\n  Freeing model (arm={arm_key})...")
    del model, tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    arm_summary["load_sec"] = load_sec
    return arm_summary


def _read_arm_summary(arm_dir: Path, datasets: list) -> dict:
    result = {}
    for ds in datasets:
        if done_marker(arm_dir, ds).exists():
            result[ds] = cached_cell_summary(arm_dir, ds)
    return result


# ── Summary ───────────────────────────────────────────────────────────────────
def write_summary(model_key: str, model_cfg: dict, all_arm_results: dict):
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model_key,
        "display_name": model_cfg["display_name"],
        "hf": model_cfg["hf"],
        "param_b": model_cfg["param_b"],
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arms": all_arm_results,
    }
    summary_file = MATRIX_DIR / f"{model_key}-summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    log(f"\nSummary → {summary_file}")
    return summary_file


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MLX quantization audit matrix runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", required=True, help="e.g. bf16,4bit-rtn")
    parser.add_argument("--datasets", required=True, help="e.g. humaneval,mbpp,realtasks")
    args = parser.parse_args()

    model_key = args.model.lower()
    arms = [a.strip() for a in args.arms.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    if model_key not in MODEL_REGISTRY:
        sys.exit(f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY)}")

    model_cfg = MODEL_REGISTRY[model_key]

    EVALPLUS_RUNS_BASE.mkdir(parents=True, exist_ok=True)
    log_file = EVALPLUS_RUNS_BASE / f"{model_key}-run.log"
    global _log_fh
    _log_fh = open(log_file, "a")

    sentinel = EVALPLUS_RUNS_BASE / f"{model_key}.done"

    log("=" * 64)
    log(f"MLX Quantization Audit: {model_cfg['display_name']}")
    log(f"Arms: {arms}")
    log(f"Datasets: {datasets}")
    log(f"Log: {log_file}")
    log("=" * 64)

    all_arm_results = {}

    for arm_key in arms:
        if arm_key not in model_cfg["arms"]:
            log(f"WARNING: arm '{arm_key}' not in registry, skipping")
            continue
        try:
            arm_result = run_arm(model_key, arm_key, model_cfg["arms"][arm_key], datasets)
        except Exception as exc:
            import traceback
            log(f"FATAL ERROR in arm {arm_key}: {exc}")
            traceback.print_exc()
            arm_result = {"error": str(exc)}
        all_arm_results[arm_key] = arm_result

    summary_file = write_summary(model_key, model_cfg, all_arm_results)

    sentinel.write_text(json.dumps({
        "done": True,
        "model": model_key,
        "arms": arms,
        "datasets": datasets,
        "summary": str(summary_file),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    log(f"Sentinel → {sentinel}")
    log(f"Run complete — {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    if _log_fh:
        _log_fh.close()


if __name__ == "__main__":
    main()
