#!/usr/bin/env python3
"""Generate HumanEval completions through one of two backends, recording raw bytes.

This is the generator half of the cluster determinism gate. The question it serves:
does running a model across two machines change what the model writes?

The existing quantization harness answers a narrower version of that question for a
single machine (section 3.4: greedy MLX reproduced 40/40 identical completions on
repeat runs). Distributed inference is untested, and the MoE study depends on it --
a bf16 arm that only fits across two nodes is worthless as a baseline if the act of
distributing it perturbs the output.

Three arms are compared, not two, because "single node vs cluster" is really two
changes stacked:

    A  mlx    lab-02 alone, in-process mlx_lm      <- the existing paper's protocol
    B  exo    exo serving, ONE node in the cluster
    C  exo    exo serving, TWO nodes in the cluster

    A vs B  isolates the serving stack (does exo's own plumbing change anything?)
    B vs C  isolates distribution (does sharding across machines change anything?)
    A vs C  the end-to-end question the MoE arm actually needs answered

Comparing only A and C would confound the two, and a difference there would be
uninterpretable -- which is the same mistake the quantization paper argues against
when it refuses to subtract vendor-reported bf16 numbers.

Prompt construction and code extraction are shared verbatim with
data/slm-benchmark/evalplus_generate.py, imported rather than copied, so the arms
cannot drift from each other or from the published protocol.

Sampling is pinned exhaustively rather than left to defaults. exo resolves unset
sampling parameters from the model card (see with_card_sampling_defaults in
exo/shared/types/text_generation.py), and a card-supplied repetition_penalty would
change the logits -- producing an A/B difference caused by sampling policy rather
than by the cluster. Every knob is therefore sent explicitly.

Usage:
  cluster_gate_generate.py --backend mlx --model mlx-community/Llama-3.2-3B-bf16 \
      --tag A-mlx-single --out data/cluster-gate/runs
  cluster_gate_generate.py --backend exo --model mlx-community/Llama-3.2-3B-bf16 \
      --endpoint http://lab-02.local:52415 --tag C-exo-2node --out data/cluster-gate/runs
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "slm-benchmark"))
from evalplus_generate import MAX_GEN_TOKENS, SYS_MSG, extract_code  # noqa: E402

PROMPTS = Path("/tmp/evalplus_prompts")

# Pinned so that no card default can differ between backends. Greedy argmax makes
# top_p/top_k/min_p inert, but they are sent anyway: an inert-but-explicit value
# cannot be silently replaced, whereas None can.
SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42,
}


def build_messages(task):
    user = f"Complete the following Python function:\n\n{task['prompt']}"
    return [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user}]


def gen_mlx(model_id, tasks, ids):
    """In-process mlx_lm generation -- the protocol the quantization paper used."""
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_id)
    sampler = make_sampler(temp=0.0)
    for tid in ids:
        msgs = build_messages(tasks[tid])
        try:
            chat = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            chat = msgs[-1]["content"]
        chunks = []
        for resp in stream_generate(model, tokenizer, chat,
                                    max_tokens=MAX_GEN_TOKENS, sampler=sampler):
            chunks.append(resp.text)
        yield tid, "".join(chunks)


def gen_exo(endpoint, model_id, tasks, ids):
    """Generation through exo's OpenAI-compatible endpoint.

    Non-streaming: streaming would interleave chunk boundaries with network timing,
    and chunk boundaries are not part of what is being tested. The concatenated text
    is what matters and it is what the comparison hashes.
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    for tid in ids:
        body = dict(SAMPLING)
        body.update({
            "model": model_id,
            "messages": build_messages(tasks[tid]),
            "max_tokens": MAX_GEN_TOKENS,
            "stream": False,
        })
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                payload = json.load(r)
        except urllib.error.HTTPError as e:
            # Fail loudly and immediately. A gate that silently records an error
            # string as a completion would report a spurious mismatch and send the
            # whole MoE study down a false trail.
            sys.exit(f"ABORT: {tid} HTTP {e.code}: {e.read()[:500].decode(errors='replace')}")
        yield tid, payload["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mlx", "exo"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="http://localhost:52415")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="data/cluster-gate/runs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 164")
    ap.add_argument("--nodes", type=int, default=1, help="recorded in metadata only")
    args = ap.parse_args()

    tasks = json.loads((PROMPTS / "humaneval.json").read_text())
    ids = sorted(tasks, key=lambda t: int(t.split("/")[1]))
    if args.limit:
        ids = ids[: args.limit]

    outdir = Path(args.out) / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    src = (gen_mlx(args.model, tasks, ids) if args.backend == "mlx"
           else gen_exo(args.endpoint, args.model, tasks, ids))

    t0 = time.time()
    raw_path = outdir / "raw.jsonl"
    samples_path = outdir / "humaneval-samples.jsonl"
    n = 0
    with raw_path.open("w") as rf, samples_path.open("w") as sf:
        for tid, text in src:
            # The raw completion is the primary artifact. Extraction is lossy and
            # normalising -- two genuinely different generations can extract to the
            # same code -- so the byte comparison must happen before it.
            rf.write(json.dumps({
                "task_id": tid,
                "raw": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "n_chars": len(text),
            }) + "\n")
            code = extract_code(text)
            if f"def {tasks[tid]['entry_point']}" not in code:
                code = tasks[tid]["prompt"] + "\n" + code
            sf.write(json.dumps({"task_id": tid, "solution": code}) + "\n")
            n += 1
            if n % 25 == 0 or n == len(ids):
                print(f"  [{args.tag}] {n}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    (outdir / "meta.json").write_text(json.dumps({
        "tag": args.tag, "backend": args.backend, "model": args.model,
        "endpoint": args.endpoint if args.backend == "exo" else None,
        "nodes": args.nodes, "n_tasks": n, "elapsed_sec": round(time.time() - t0, 1),
        "max_gen_tokens": MAX_GEN_TOKENS, "sampling": SAMPLING,
    }, indent=2))
    print(f"[{args.tag}] wrote {raw_path} and {samples_path}")


if __name__ == "__main__":
    main()
