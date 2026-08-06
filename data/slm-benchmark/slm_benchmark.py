#!/usr/bin/env python3
"""
SLM Coding Efficiency Benchmark Phase 1 — lab-02
Sequential evaluation: HumanEval pass@1, MBPP pass@1, 10 real coding tasks.
Deletes each model from cache before downloading the next.
"""

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR = Path(os.path.expanduser("~/ResearchPapers/data/slm-benchmark"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "results.json"
LOG_FILE = RESULTS_DIR / "benchmark.log"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))

MAX_GEN_TOKENS = 512
EXEC_TIMEOUT = 30  # seconds per subprocess code execution

MODELS = [
    {
        "name": "Qwen2.5-Coder-1.5B-Instruct",
        "mlx_id": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "hf_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "param_b": 1.5,
    },
    {
        "name": "Qwen2.5-Coder-3B-Instruct",
        "mlx_id": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "hf_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "param_b": 3.0,
    },
    {
        "name": "Qwen2.5-Coder-7B-Instruct",
        "mlx_id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "param_b": 7.0,
    },
    {
        "name": "Phi-4-Mini-Instruct",
        "mlx_id": "mlx-community/phi-4-mini-instruct-4bit",
        "hf_id": "microsoft/phi-4-mini-instruct",
        "param_b": 3.8,
    },
    {
        "name": "DeepSeek-Coder-V2-Lite-Instruct",
        "mlx_id": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
        "hf_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "param_b": 16.0,
    },
    {
        "name": "Granite-Code-8B-Instruct",
        "mlx_id": "mlx-community/granite-code-8b-instruct-4bit",
        "hf_id": "ibm-granite/granite-code-8b-instruct",
        "param_b": 8.0,
    },
]

# 10 real coding tasks derived from our research codebase patterns
REAL_TASKS = [
    {
        "id": "rt_01",
        "name": "l2_normalize_rows",
        "prompt": """\
import numpy as np

def l2_normalize_rows(W: np.ndarray) -> np.ndarray:
    \"\"\"
    Normalize each row of 2D array W to unit L2 norm.
    Rows with norm < 1e-12 are returned as-is (zero vectors stay zero).

    Args:
        W: shape (n, d) float array
    Returns:
        shape (n, d) array with unit-norm rows
    \"\"\"
""",
        "tests": [
            "import numpy as np",
            "W = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0], [-0.6, 0.8]])",
            "R = l2_normalize_rows(W)",
            "assert R.shape == W.shape",
            "assert abs(np.linalg.norm(R[0]) - 1.0) < 1e-6",
            "assert abs(np.linalg.norm(R[1]) - 1.0) < 1e-6",
            "assert np.allclose(R[2], [0.0, 0.0])",
            "assert np.allclose(R[0], [0.6, 0.8])",
            "assert np.allclose(R[3], [-0.6, 0.8])",
        ],
    },
    {
        "id": "rt_02",
        "name": "batched_cosine_similarity",
        "prompt": """\
import numpy as np

def batched_cosine_similarity(A: np.ndarray, B: np.ndarray, batch_size: int = 512) -> np.ndarray:
    \"\"\"
    Compute full cosine similarity matrix between L2-normalized rows of A and B.
    Processes A in chunks of batch_size to limit peak memory.

    Args:
        A: shape (m, d), rows already L2-normalized
        B: shape (n, d), rows already L2-normalized
        batch_size: number of A rows per batch
    Returns:
        sim: shape (m, n), values in [-1, 1]
    \"\"\"
""",
        "tests": [
            "import numpy as np",
            "A = np.array([[1.0, 0.0], [0.0, 1.0], [1/np.sqrt(2), 1/np.sqrt(2)]])",
            "B = np.array([[1.0, 0.0], [0.0, 1.0]])",
            "sim = batched_cosine_similarity(A, B, batch_size=2)",
            "assert sim.shape == (3, 2)",
            "assert abs(sim[0, 0] - 1.0) < 1e-6",
            "assert abs(sim[0, 1]) < 1e-6",
            "assert abs(sim[1, 0]) < 1e-6",
            "assert abs(sim[1, 1] - 1.0) < 1e-6",
            "assert abs(sim[2, 0] - 1/np.sqrt(2)) < 1e-6",
            "assert abs(sim[2, 1] - 1/np.sqrt(2)) < 1e-6",
        ],
    },
    {
        "id": "rt_03",
        "name": "bootstrap_confidence_interval",
        "prompt": """\
import numpy as np

def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple:
    \"\"\"
    Bootstrap confidence interval for the mean of 1D data.

    Args:
        data: 1D float array of observations
        n_bootstrap: number of resamples
        confidence: CI level (e.g. 0.95)
        seed: random seed for reproducibility
    Returns:
        (lower, upper) tuple of floats
    \"\"\"
""",
        "tests": [
            "import numpy as np",
            "data = np.arange(1.0, 11.0)",
            "lo, hi = bootstrap_confidence_interval(data, n_bootstrap=5000, confidence=0.95, seed=0)",
            "assert lo < 5.5 < hi, f'Mean 5.5 must be inside CI [{lo:.3f}, {hi:.3f}]'",
            "assert lo > 2.0, f'Lower {lo:.3f} too low'",
            "assert hi < 9.0, f'Upper {hi:.3f} too high'",
            "assert lo < hi",
        ],
    },
    {
        "id": "rt_04",
        "name": "top_k_per_row",
        "prompt": """\
import numpy as np

def top_k_per_row(sim: np.ndarray, k: int) -> np.ndarray:
    \"\"\"
    For each row of sim return the k column indices with the highest values,
    ordered from highest to lowest.

    Args:
        sim: shape (m, n) float array
        k: number of top indices per row
    Returns:
        indices: shape (m, k) int array, descending by value per row
    \"\"\"
""",
        "tests": [
            "import numpy as np",
            "sim = np.array([[0.1, 0.9, 0.5, 0.3], [0.8, 0.2, 0.7, 0.4]])",
            "idx = top_k_per_row(sim, k=2)",
            "assert idx.shape == (2, 2)",
            "assert set(idx[0].tolist()) == {1, 2}",
            "assert set(idx[1].tolist()) == {0, 2}",
            "assert idx[0, 0] == 1, 'row 0 best should be col 1'",
            "assert idx[1, 0] == 0, 'row 1 best should be col 0'",
        ],
    },
    {
        "id": "rt_05",
        "name": "ioi_logit_difference",
        "prompt": """\
import numpy as np

def ioi_logit_difference(logits: np.ndarray, io_tokens: list, s_tokens: list) -> float:
    \"\"\"
    Mean (IO - S) logit difference at the final sequence position.
    Used in Indirect Object Identification (IOI) circuit analysis.

    Args:
        logits: shape (batch, seq_len, vocab_size) unnormalized logits
        io_tokens: list of int, indirect-object token id per example
        s_tokens: list of int, subject token id per example
    Returns:
        float: mean over batch of logit[IO] - logit[S] at last position
    \"\"\"
""",
        "tests": [
            "import numpy as np",
            "logits = np.zeros((3, 10, 100))",
            "logits[0, -1, 5] = 2.0; logits[0, -1, 10] = 1.0",
            "logits[1, -1, 20] = 3.0; logits[1, -1, 30] = 1.5",
            "logits[2, -1, 7] = 0.5; logits[2, -1, 15] = 1.5",
            "result = ioi_logit_difference(logits, [5, 20, 7], [10, 30, 15])",
            "expected = ((2.0-1.0) + (3.0-1.5) + (0.5-1.5)) / 3.0",
            "assert abs(result - expected) < 1e-6, f'Expected {expected}, got {result}'",
        ],
    },
    {
        "id": "rt_06",
        "name": "procrustes_alignment",
        "prompt": """\
import numpy as np

def procrustes_alignment(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    \"\"\"
    Orthogonal Procrustes: find R minimizing ||A @ R - B||_F with R^T R = I.
    Solution: compute SVD of A^T B = U S Vt, return R = U @ Vt.

    Args:
        A: shape (n, d1) source space
        B: shape (n, d2) target space
    Returns:
        R: shape (d1, d2) orthonormal alignment matrix
    \"\"\"
""",
        "tests": [
            "import numpy as np",
            "A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)",
            "R_true = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)",
            "B = A @ R_true",
            "R_out = procrustes_alignment(A, B)",
            "assert R_out.shape == (2, 2)",
            "assert np.allclose(A @ R_out, B, atol=1e-5), 'Reconstruction error too large'",
            "assert np.allclose(R_out.T @ R_out, np.eye(2), atol=1e-5), 'R not orthonormal'",
        ],
    },
    {
        "id": "rt_07",
        "name": "merge_json_metrics",
        "prompt": """\
def merge_json_metrics(records: list) -> dict:
    \"\"\"
    Merge a list of metric dicts by averaging numeric values.
    Only keys present in ALL records with numeric (int/float) values
    in every record are included; others are silently dropped.

    Args:
        records: list of dict
    Returns:
        dict mapping key -> mean float

    Example:
        merge_json_metrics([{'loss': 0.5, 'tag': 'a'}, {'loss': 0.3, 'tag': 'b'}])
        -> {'loss': 0.4}
    \"\"\"
""",
        "tests": [
            "records = [{'loss': 0.5, 'acc': 0.8, 'epoch': 1}, {'loss': 0.3, 'acc': 0.9, 'epoch': 2}]",
            "result = merge_json_metrics(records)",
            "assert abs(result['loss'] - 0.4) < 1e-9",
            "assert abs(result['acc'] - 0.85) < 1e-9",
            "assert abs(result['epoch'] - 1.5) < 1e-9",
            "r2 = merge_json_metrics([{'a': 1, 'label': 'x'}, {'a': 3, 'label': 'y'}])",
            "assert 'label' not in r2, 'non-numeric keys must be dropped'",
            "assert abs(r2['a'] - 2.0) < 1e-9",
        ],
    },
    {
        "id": "rt_08",
        "name": "extract_code_block",
        "prompt": """\
import re

def extract_code_block(response: str) -> str:
    \"\"\"
    Extract Python code from an LLM markdown response.
    Priority: ```python fence > generic ``` fence > full stripped response.
    Returns content of the FIRST matching fence, or the full response if none found.

    Args:
        response: raw LLM output string
    Returns:
        extracted code string
    \"\"\"
""",
        "tests": [
            "import re",
            "r1 = 'Here is the solution:\\n```python\\ndef foo(): return 1\\n```\\nDone.'",
            "out1 = extract_code_block(r1)",
            "assert 'def foo' in out1",
            "assert '```' not in out1",
            "r2 = 'Solution:\\n```\\nx = 1 + 2\\n```'",
            "out2 = extract_code_block(r2)",
            "assert 'x = 1 + 2' in out2",
            "r3 = 'def bar(): pass'",
            "assert 'def bar' in extract_code_block(r3)",
        ],
    },
    {
        "id": "rt_09",
        "name": "parse_training_log",
        "prompt": """\
import re

def parse_training_log(log_text: str) -> list:
    \"\"\"
    Parse key=value pairs from training log output.
    Returns a list of dicts, one per line containing at least one key=value.
    Numeric values (int, float, scientific notation) are cast to float.
    Non-numeric values are kept as str. Lines with no pairs are skipped.

    Example:
        Input:  'step=100 loss=0.5 lr=1e-4'
        Output: [{'step': 100.0, 'loss': 0.5, 'lr': 0.0001}]
    \"\"\"
""",
        "tests": [
            "import re",
            "log = 'step=100 loss=0.5 lr=1e-4\\nstep=200 loss=0.4 lr=9e-5\\nNo pairs here\\nstep=300 loss=0.3 tag=best'",
            "records = parse_training_log(log)",
            "assert len(records) == 3, f'Expected 3 records, got {len(records)}'",
            "assert records[0]['step'] == 100.0",
            "assert abs(records[0]['lr'] - 1e-4) < 1e-15",
            "assert records[2]['tag'] == 'best'",
            "assert 'step' in records[1]",
        ],
    },
    {
        "id": "rt_10",
        "name": "power_required_n",
        "prompt": """\
import math

def power_required_n(effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
    \"\"\"
    Required per-group n for two-sample two-tailed t-test (normal approximation).
    Formula: n = ceil(2 * ((z_{alpha/2} + z_{beta}) / effect_size)^2)
    Uses scipy-free normal ppf via the rational approximation from Abramowitz & Stegun.

    Args:
        effect_size: Cohen's d
        power: desired power (1 - beta), e.g. 0.8
        alpha: significance level, e.g. 0.05
    Returns:
        int: per-group sample size (ceiling)

    Example: d=0.5, power=0.8, alpha=0.05 -> ~64
    \"\"\"
""",
        "tests": [
            "import math",
            "n = power_required_n(0.5, power=0.8, alpha=0.05)",
            "assert isinstance(n, int)",
            "assert 60 <= n <= 70, f'Expected ~64 for d=0.5, got {n}'",
            "n2 = power_required_n(0.8, power=0.8, alpha=0.05)",
            "assert 22 <= n2 <= 32, f'Expected ~26 for d=0.8, got {n2}'",
            "n3 = power_required_n(0.5, power=0.9, alpha=0.05)",
            "assert n3 > n, 'Higher power requires more samples'",
        ],
    },
]

# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─── Code extraction ──────────────────────────────────────────────────────────

def extract_code(response: str) -> str:
    """Pull Python code out of an LLM response, handling markdown fences."""
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some models use triple-backtick without newline after language tag
    m = re.search(r"```python(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


# ─── Execution harness ────────────────────────────────────────────────────────

def run_code(full_code: str) -> bool:
    """Execute full_code in a subprocess; return True on exit 0."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=EXEC_TIMEOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ─── Generation ───────────────────────────────────────────────────────────────

def make_chat_prompt(tokenizer, user_text: str) -> str:
    """Apply the model's chat template. Falls back to bare text if it errors."""
    sys_msg = (
        "You are an expert Python programmer. "
        "Complete the function. Return only the function body with correct indentation—no explanation."
    )
    try:
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        try:
            msgs = [{"role": "user", "content": user_text}]
            return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            return user_text


def generate_timed(model, tokenizer, prompt_text: str) -> tuple:
    """
    Generate a completion, return (text, generation_tps, peak_memory_gb).
    GenerationResponse carries generation_tps and peak_memory natively.
    """
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    chunks = []
    gen_tps = 0.0
    peak_mem_gb = 0.0
    for resp in stream_generate(
        model, tokenizer, prompt_text,
        max_tokens=MAX_GEN_TOKENS,
        sampler=sampler,
    ):
        chunks.append(resp.text)
        if resp.generation_tps and resp.generation_tps > 0:
            gen_tps = resp.generation_tps
        if resp.peak_memory:
            peak_mem_gb = max(peak_mem_gb, resp.peak_memory)
    text = "".join(chunks)
    return text, gen_tps, peak_mem_gb


# ─── Benchmark suites ─────────────────────────────────────────────────────────

def run_humaneval(model, tokenizer) -> dict:
    from datasets import load_dataset

    log("  HumanEval: loading dataset...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    log(f"  HumanEval: {len(ds)} problems")

    passed = 0
    tps_list = []
    peak_mem = 0.0

    for i, prob in enumerate(ds):
        user_msg = f"Complete the following Python function:\n\n{prob['prompt']}"
        chat_prompt = make_chat_prompt(tokenizer, user_msg)
        response, tps, pmem = generate_timed(model, tokenizer, chat_prompt)
        tps_list.append(tps)
        peak_mem = max(peak_mem, pmem)

        code = extract_code(response)

        # If model reproduced the full function, use code directly; otherwise append to prompt
        if f"def {prob['entry_point']}" in code:
            full_code = code
        else:
            full_code = prob["prompt"] + "\n" + code
        full_code += f"\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"

        ok = run_code(full_code)
        passed += ok

        if (i + 1) % 20 == 0 or i == len(ds) - 1:
            log(f"    HumanEval [{i+1}/{len(ds)}] pass={passed} tps={np.mean(tps_list):.1f}")

    total = len(ds)
    pass_at_1 = passed / total
    log(f"  HumanEval pass@1 = {pass_at_1:.4f} ({passed}/{total})")
    return {"pass_at_1": round(pass_at_1, 4), "passed": passed, "total": total,
            "mean_tps": round(float(np.mean(tps_list)), 1), "peak_memory_gb": round(peak_mem, 2)}


def run_mbpp(model, tokenizer) -> dict:
    from datasets import load_dataset

    log("  MBPP: loading dataset (sanitized)...")
    try:
        ds = load_dataset("mbpp", "sanitized", split="test")
    except Exception:
        log("  MBPP sanitized not found, using default split...")
        ds = load_dataset("mbpp", split="test")
    log(f"  MBPP: {len(ds)} problems")

    passed = 0
    tps_list = []
    peak_mem = 0.0

    for i, prob in enumerate(ds):
        user_msg = (
            f"Write a Python function for the following task:\n\n{prob["prompt"]}\n\n"
            "Return only the complete function definition."
        )
        chat_prompt = make_chat_prompt(tokenizer, user_msg)
        response, tps, pmem = generate_timed(model, tokenizer, chat_prompt)
        tps_list.append(tps)
        peak_mem = max(peak_mem, pmem)

        code = extract_code(response)
        tests = "\n".join(prob["test_list"])
        full_code = f"{code}\n\n{tests}\n"

        ok = run_code(full_code)
        passed += ok

        if (i + 1) % 50 == 0 or i == len(ds) - 1:
            log(f"    MBPP [{i+1}/{len(ds)}] pass={passed} tps={np.mean(tps_list):.1f}")

    total = len(ds)
    pass_at_1 = passed / total
    log(f"  MBPP pass@1 = {pass_at_1:.4f} ({passed}/{total})")
    return {"pass_at_1": round(pass_at_1, 4), "passed": passed, "total": total,
            "mean_tps": round(float(np.mean(tps_list)), 1), "peak_memory_gb": round(peak_mem, 2)}


def run_real_tasks(model, tokenizer) -> dict:
    task_results = []
    tps_list = []

    peak_mem = 0.0
    for task in REAL_TASKS:
        user_msg = f"Complete the following Python function:\n\n{task['prompt']}"
        chat_prompt = make_chat_prompt(tokenizer, user_msg)
        response, tps, pmem = generate_timed(model, tokenizer, chat_prompt)
        tps_list.append(tps)
        peak_mem = max(peak_mem, pmem)

        code = extract_code(response)
        # Prepend the prompt (contains the def line) then append generated body + tests
        full_code = task["prompt"] + "\n" + code + "\n\n" + "\n".join(task["tests"]) + "\n"

        ok = run_code(full_code)
        task_results.append({"id": task["id"], "name": task["name"], "passed": ok,
                              "tps": round(tps, 1)})
        log(f"    {task['name']}: {'PASS' if ok else 'FAIL'} @ {tps:.1f} tps")

    n_passed = sum(r["passed"] for r in task_results)
    log(f"  Real tasks: {n_passed}/10 passed")
    return {"pass_rate": round(n_passed / 10, 2), "passed": n_passed, "total": 10,
            "mean_tps": round(float(np.mean(tps_list)), 1),
            "peak_memory_gb": round(peak_mem, 2), "detail": task_results}


# ─── Model management ─────────────────────────────────────────────────────────

def delete_model_cache(mlx_id: str):
    """Remove model directory from HuggingFace hub cache."""
    cache_name = "models--" + mlx_id.replace("/", "--")
    cache_path = HF_CACHE / cache_name
    if cache_path.exists():
        shutil.rmtree(cache_path)
        log(f"  Deleted: {cache_path}")
    else:
        log(f"  Cache not found (already clean?): {cache_path}")


def load_model_with_fallback(model_cfg: dict):
    """
    Try mlx-community 4-bit first; if that fails, convert from HF with --q-bits 4.
    Returns (model, tokenizer, local_converted_path_or_None).
    """
    from mlx_lm import load as mlx_load

    mlx_id = model_cfg["mlx_id"]
    log(f"  Trying mlx-community 4-bit: {mlx_id}")
    try:
        model, tokenizer = mlx_load(mlx_id)
        return model, tokenizer, None
    except Exception as e1:
        log(f"  mlx-community load failed: {e1}")

    hf_id = model_cfg["hf_id"]
    local_path = str(RESULTS_DIR / "converted_model_tmp")
    log(f"  Falling back to mlx_lm.convert from {hf_id} -> {local_path}")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.convert",
                "--hf-path", hf_id,
                "--mlx-path", local_path,
                "--quantize", "--q-bits", "4",
            ],
            check=True, timeout=7200,
        )
        model, tokenizer = mlx_load(local_path)
        return model, tokenizer, local_path
    except Exception as e2:
        log(f"  Conversion also failed: {e2}")
        raise RuntimeError(f"Cannot load {model_cfg['name']}: {e1} | {e2}")


# ─── Per-model benchmark ──────────────────────────────────────────────────────

def benchmark_model(model_cfg: dict) -> dict:
    import mlx.core as mx

    name = model_cfg["name"]
    log(f"\n{'='*64}")
    log(f"  MODEL: {name}  ({model_cfg['param_b']}B params, 4-bit)")
    log(f"{'='*64}")

    t_load = time.time()
    model, tokenizer, converted_path = load_model_with_fallback(model_cfg)
    mx.eval(model.parameters())  # ensure weights are materialized
    t_load_elapsed = time.time() - t_load
    log(f"  Loaded in {t_load_elapsed:.1f}s")

    humaneval = run_humaneval(model, tokenizer)
    mbpp = run_mbpp(model, tokenizer)
    real = run_real_tasks(model, tokenizer)

    # Peak memory across all suites (from GenerationResponse.peak_memory — unified MLX memory)
    peak_memory_gb = round(max(
        humaneval.get("peak_memory_gb", 0),
        mbpp.get("peak_memory_gb", 0),
        real.get("peak_memory_gb", 0),
    ), 2)

    all_tps = [humaneval["mean_tps"], mbpp["mean_tps"], real["mean_tps"]]
    mean_tps = round(float(np.mean(all_tps)), 1)

    log(f"  Freeing model...")
    del model, tokenizer
    gc.collect()

    # Delete from cache
    delete_model_cache(model_cfg["mlx_id"])
    if converted_path and Path(converted_path).exists():
        shutil.rmtree(converted_path)
        log(f"  Deleted converted model: {converted_path}")
    # Also remove HF cache for the HF base model if we converted
    if converted_path:
        hf_cache_name = "models--" + model_cfg["hf_id"].replace("/", "--")
        hf_cache_path = HF_CACHE / hf_cache_name
        if hf_cache_path.exists():
            shutil.rmtree(hf_cache_path)
            log(f"  Deleted HF cache: {hf_cache_path}")

    return {
        "model": name,
        "mlx_id": model_cfg["mlx_id"],
        "param_b": model_cfg["param_b"],
        "load_time_sec": round(t_load_elapsed, 1),
        "peak_memory_gb": peak_memory_gb,
        "mean_tokens_per_sec": mean_tps,
        "humaneval": humaneval,
        "mbpp": mbpp,
        "real_tasks": real,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ─── Markdown table ───────────────────────────────────────────────────────────

def write_markdown_table(results: list):
    md_path = RESULTS_DIR / "quality_per_gb.md"
    rows = []
    for r in results:
        if "error" in r:
            rows.append(f"| {r['model']} | {r['param_b']}B | — | ERROR | — | — | — | — | — |")
            continue
        name = r["model"]
        pb = r["param_b"]
        mem = r.get("peak_memory_gb", 0) or 1.0
        he = r["humaneval"]["pass_at_1"]
        mb = r["mbpp"]["pass_at_1"]
        rt = r["real_tasks"]["pass_rate"]
        tps = r.get("mean_tokens_per_sec", 0)
        he_gb = round(he / mem, 4)
        mb_gb = round(mb / mem, 4)
        rows.append(
            f"| {name} | {pb}B | {mem:.1f} | {he:.3f} | {mb:.3f} | {rt:.1f} | {tps:.0f} | {he_gb:.4f} | {mb_gb:.4f} |"
        )

    lines = [
        "# SLM Coding Efficiency — Phase 1 Benchmark",
        "",
        f"*Run: {time.strftime('%Y-%m-%d')} | Hardware: Apple Silicon (lab-02, 32 GB) | Quant: 4-bit MLX*",
        "",
        "## Results",
        "",
        "| Model | Params | Mem GB | HumanEval | MBPP | Real/10 | Tok/s | HE/GB | MBPP/GB |",
        "|-------|--------|--------|-----------|------|---------|-------|-------|---------|",
        *rows,
        "",
        "## Notes",
        "- **HumanEval**: pass@1 greedy, 164 problems (`openai/openai_humaneval`)",
        "- **MBPP**: pass@1 greedy, MBPP-sanitized test split",
        "- **Real/10**: 10 research-codebase tasks (l2_normalize, cosine_sim, bootstrap CI, top-k,",
        "  IOI logit diff, Procrustes, JSON merge, code extraction, log parsing, power analysis)",
        "- **Tok/s**: mean generation throughput across all three suites",
        "- **HE/GB / MBPP/GB**: pass@1 ÷ peak model memory (quality-per-GB efficiency)",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    log(f"\nMarkdown table: {md_path}")
    return md_path


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"results": []}


def save_checkpoint(data: dict):
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log(f"SLM Benchmark Phase 1 — start {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    log(f"Output dir: {RESULTS_DIR}")
    log(f"Models: {len(MODELS)}")

    ckpt = load_checkpoint()
    all_results = ckpt.get("results", [])
    done = {r["model"] for r in all_results}

    for model_cfg in MODELS:
        name = model_cfg["name"]
        if name in done:
            log(f"\nSKIP {name} (already in checkpoint)")
            continue

        try:
            result = benchmark_model(model_cfg)
        except Exception as exc:
            import traceback as tb
            log(f"\nERROR on {name}: {exc}")
            tb.print_exc()
            result = {
                "model": name,
                "mlx_id": model_cfg["mlx_id"],
                "param_b": model_cfg["param_b"],
                "error": str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        all_results.append(result)
        ckpt["results"] = all_results
        save_checkpoint(ckpt)
        RESULTS_FILE.write_text(json.dumps({"benchmark": "slm-phase1", "results": all_results}, indent=2))
        log(f"\nCheckpoint saved after {name}")

    RESULTS_FILE.write_text(json.dumps({"benchmark": "slm-phase1", "results": all_results}, indent=2))
    write_markdown_table(all_results)
    log(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
