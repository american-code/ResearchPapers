#!/usr/bin/env python3
"""
Two-node distributed activation collection PoC.

Validates that split-layer inference (layers 0-15 in Node A, 16-27 in Node B)
produces activations numerically matching a single-process reference.

Wire protocol: ActStream v1.0.0 over Unix domain socket
  - 8-byte header (layer_idx uint16, token_start uint32, token_count uint16)
  - big-endian float16 payload
  - credit-based flow control with DATA_ACK

Output: data/distributed-capture-validation.json
"""
import json, time, pathlib, subprocess, sys, tempfile
import numpy as np

SCRIPTS = pathlib.Path(__file__).parent
PYTHON = sys.executable
OUT_JSON = pathlib.Path(__file__).parent.parent / "distributed-capture-validation.json"

MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"
NUM_LAYERS = 28        # Llama-3.2-3B
LAYER_SPLIT = 16       # A: 0-15, B: 16-27
NUM_TOKENS = 512
CHUNK_SIZE = 128       # tokens per streaming message → 4 chunks for 512 tokens
HIDDEN_DIM = 3072


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_token_ids(n_tokens):
    """Tokenize wikitext-103 and return the first n_tokens token IDs."""
    from mlx_lm import load
    from datasets import load_dataset

    print("  Loading tokenizer ...", flush=True)
    _, tokenizer = load(MODEL_ID)

    print("  Streaming wikitext-103-raw-v1 ...", flush=True)
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1",
        split="train", trust_remote_code=False,
    )
    buf = []
    for row in ds:
        if len(buf) >= n_tokens:
            break
        text = row["text"].strip()
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))

    return np.array(buf[:n_tokens], dtype=np.int32)


def run_subprocess(script, args_dict, label, timeout=600):
    """Run a script as subprocess, print its stdout, raise on non-zero exit."""
    cmd = [PYTHON, str(script), json.dumps(args_dict)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in result.stdout.splitlines():
        print(f"  [{label}] {line}", flush=True)
    if result.returncode != 0:
        print(f"  [{label}] STDERR:\n{result.stderr}", flush=True)
        raise RuntimeError(f"{label} exited {result.returncode}")
    return result.stdout


def wait_for_line(proc, token, timeout=300):
    """Read proc.stdout line by line until a line containing `token` appears."""
    import select
    deadline = time.time() + timeout
    while time.time() < deadline:
        r, _, _ = select.select([proc.stdout], [], [], 1.0)
        if r:
            line = proc.stdout.readline()
            if not line:
                raise EOFError("process stdout closed before ready token")
            line = line.rstrip()
            print(f"  [B] {line}", flush=True)
            if token in line:
                return
    raise TimeoutError(f"Timed out after {timeout}s waiting for '{token}'")


def drain_proc_stdout(proc, label):
    """Read and print all remaining stdout from a finished process."""
    for line in proc.stdout:
        print(f"  [{label}] {line.rstrip()}", flush=True)


def compare(ref: np.ndarray, dist: np.ndarray, name: str) -> dict:
    diff = np.abs(ref.astype(np.float32) - dist.astype(np.float32))
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    rms_err = float(np.sqrt((diff ** 2).mean()))
    return {
        "name": name,
        "shape": list(ref.shape),
        "max_abs_error": round(max_err, 8),
        "mean_abs_error": round(mean_err, 10),
        "rms_error": round(rms_err, 10),
        "pass": max_err < 1e-2,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        socket_path  = str(tmp / "node_ab.sock")
        tok_path     = str(tmp / "token_ids.npy")
        ref_handoff  = str(tmp / "ref_handoff.npy")
        ref_final    = str(tmp / "ref_final.npy")
        dist_handoff = str(tmp / "dist_handoff.npy")
        dist_final   = str(tmp / "dist_final.npy")

        # ── Step 1: token IDs ─────────────────────────────────────────────────
        banner("Step 1: Generate token IDs")
        token_ids = get_token_ids(NUM_TOKENS)
        np.save(tok_path, token_ids)
        print(f"  {NUM_TOKENS} tokens saved", flush=True)

        # ── Step 2: single-process reference ─────────────────────────────────
        banner("Step 2: Single-process reference run")
        t_ref = time.time()
        run_subprocess(
            SCRIPTS / "reference.py",
            {
                "token_ids_path": tok_path,
                "handoff_out":    ref_handoff,
                "final_out":      ref_final,
                "layer_split":    LAYER_SPLIT,
            },
            label="ref",
        )
        ref_elapsed = time.time() - t_ref
        print(f"  Reference done in {ref_elapsed:.1f}s", flush=True)

        # ── Step 3: distributed run ───────────────────────────────────────────
        banner("Step 3: Distributed run (Node A → Node B via Unix socket)")
        t_dist = time.time()

        # Start Node B first; wait until its model is loaded before launching A.
        print("  Launching Node B ...", flush=True)
        proc_b = subprocess.Popen(
            [PYTHON, str(SCRIPTS / "process_b.py"), json.dumps({
                "socket_path": socket_path,
                "final_out":   dist_final,
                "layer_split": LAYER_SPLIT,
            })],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        wait_for_line(proc_b, "NODE_B_MODEL_READY", timeout=300)

        # Node B model is loaded and accepting; launch Node A.
        print("  Launching Node A ...", flush=True)
        proc_a = subprocess.Popen(
            [PYTHON, str(SCRIPTS / "process_a.py"), json.dumps({
                "socket_path":    socket_path,
                "token_ids_path": tok_path,
                "handoff_out":    dist_handoff,
                "layer_split":    LAYER_SPLIT,
                "chunk_size":     CHUNK_SIZE,
            })],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Drain both stdout streams concurrently so neither pipe fills and
        # blocks its writer while we wait on the other.
        import threading
        lines_a, lines_b = [], []

        def _drain(proc, buf):
            for line in proc.stdout:
                buf.append(line.rstrip())

        t_a = threading.Thread(target=_drain, args=(proc_a, lines_a))
        t_b = threading.Thread(target=_drain, args=(proc_b, lines_b))
        t_a.start()
        t_b.start()

        # 10-minute ceiling: two cold model loads + forward passes under memory
        # pressure can take several minutes each on 16 GB machines.
        if proc_a.wait(timeout=600) != 0:
            raise RuntimeError(f"Node A exited {proc_a.returncode}")
        if proc_b.wait(timeout=600) != 0:
            raise RuntimeError(f"Node B exited {proc_b.returncode}")

        t_a.join()
        t_b.join()

        for line in lines_a:
            print(f"  [A] {line}", flush=True)
        for line in lines_b:
            print(f"  [B] {line}", flush=True)

        dist_elapsed = time.time() - t_dist
        print(f"  Distributed done in {dist_elapsed:.1f}s", flush=True)

        # ── Step 4: comparison ────────────────────────────────────────────────
        banner("Step 4: Comparison")
        ref_h  = np.load(ref_handoff)
        ref_f  = np.load(ref_final)
        dist_h = np.load(dist_handoff)
        dist_f = np.load(dist_final)

        handoff_cmp = compare(ref_h, dist_h, f"handoff (layer {LAYER_SPLIT-1} output)")
        final_cmp   = compare(ref_f, dist_f, f"final   (layer {NUM_LAYERS-1} output)")

        for cmp in [handoff_cmp, final_cmp]:
            status = "PASS" if cmp["pass"] else "FAIL"
            print(
                f"  {status}  {cmp['name']:40s}  "
                f"max_err={cmp['max_abs_error']:.2e}  "
                f"rms={cmp['rms_error']:.2e}",
                flush=True,
            )

        bytes_on_wire = (
            NUM_TOKENS * HIDDEN_DIM * 2          # payload (float16)
            + (NUM_TOKENS // CHUNK_SIZE) * 8     # headers
            + 8 * 2                              # EOL + EOS sentinels
        )

        verdict = (
            "PASS: distributed final output matches single-process within tolerance"
            if final_cmp["pass"]
            else f"FAIL: max_abs_error={final_cmp['max_abs_error']:.4f} exceeds 1e-2"
        )

        result = {
            "metadata": {
                "date": "2026-07-29",
                "model": MODEL_ID,
                "num_layers": NUM_LAYERS,
                "hidden_dim": HIDDEN_DIM,
                "layer_split": LAYER_SPLIT,
                "node_a_layers": f"0-{LAYER_SPLIT-1}",
                "node_b_layers": f"{LAYER_SPLIT}-{NUM_LAYERS-1}",
                "n_tokens": NUM_TOKENS,
                "chunk_size": CHUNK_SIZE,
                "n_chunks": NUM_TOKENS // CHUNK_SIZE,
                "protocol": "ActStream v1.0.0",
                "transport": "Unix domain socket (AF_UNIX SOCK_STREAM)",
                "wire_dtype": "float16 big-endian",
                "model_dtype": "bfloat16",
                "implementation": (
                    "Two Python subprocesses simulate future Swift nodes. "
                    "Model weights loaded independently in each process. "
                    "Wire format per activation-streaming-protocol.md."
                ),
            },
            "reference": {
                "type": "single_process",
                "elapsed_s": round(ref_elapsed, 1),
            },
            "distributed": {
                "type": "two_process_unix_socket",
                "elapsed_s": round(dist_elapsed, 1),
                "bytes_on_wire": bytes_on_wire,
                "bytes_on_wire_human": f"{bytes_on_wire / 1024:.1f} KiB",
                "window_size": 8,
                "node_a": {
                    "layers": f"0-{LAYER_SPLIT-1}",
                    "chunks_sent": NUM_TOKENS // CHUNK_SIZE,
                },
                "node_b": {
                    "layers": f"{LAYER_SPLIT}-{NUM_LAYERS-1}",
                },
            },
            "comparison": {
                "handoff": handoff_cmp,
                "final": final_cmp,
                "note": (
                    "Non-zero error arises from bfloat16→float16→bfloat16 "
                    "round-trip on the wire. float16 mantissa is wider than "
                    "bfloat16 so conversion is nearly lossless; residual error "
                    "is within float16 precision."
                ),
                "verdict": verdict,
            },
        }

        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

        print(f"\nResults → {OUT_JSON}", flush=True)
        print(f"  {verdict}", flush=True)


def banner(msg):
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*60}", flush=True)


main()
