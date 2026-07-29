"""
Integration test: activation streaming protocol over localhost.
Spawns receiver and sender as separate processes, streams 10k tokens of
Llama-3.2-3B layer-16 activations, then verifies:
  - received file matches direct local reference capture (byte-for-byte)
  - throughput >= 1 GB/s
  - no dropped messages (message count and file match)
Saves result to benchmarks/activation-streaming-localhost.json.
"""
import subprocess
import sys
import json
import pathlib
import queue
import threading


def stdout_reader(proc, q):
    for line in proc.stdout:
        q.put(line.rstrip("\n"))
    q.put(None)  # sentinel: process stdout closed


def parse_done_line(lines, prefix):
    for ln in lines:
        if ln.startswith(prefix):
            return dict(kv.split("=") for kv in ln.split()[1:])
    return {}


def main():
    base = pathlib.Path(__file__).parent
    ref = "/tmp/ref_activations.bin"
    recv = "/tmp/recv_activations.bin"

    for p in (ref, recv):
        pathlib.Path(p).unlink(missing_ok=True)

    # --- Start receiver ---
    rp = subprocess.Popen(
        [sys.executable, str(base / "streaming_receiver.py"), recv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    recv_q = queue.Queue()
    recv_reader = threading.Thread(target=stdout_reader, args=(rp, recv_q), daemon=True)
    recv_reader.start()

    # Wait for receiver to bind and signal ready
    while True:
        try:
            line = recv_q.get(timeout=15)
        except queue.Empty:
            rp.kill()
            raise RuntimeError("Receiver did not signal RECEIVER_READY within 15 s")
        if line is None:
            err = rp.stderr.read()
            raise RuntimeError(f"Receiver exited before signaling ready.\nstderr:\n{err}")
        if line == "RECEIVER_READY":
            break

    # --- Start sender ---
    sp = subprocess.Popen(
        [sys.executable, str(base / "streaming_sender.py"), ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sender_out, sender_err = sp.communicate(timeout=120)
    if sp.returncode != 0:
        rp.kill()
        raise RuntimeError(
            f"Sender exited with code {sp.returncode}.\nstderr:\n{sender_err}"
        )

    # --- Collect receiver output ---
    recv_lines = []
    while True:
        try:
            line = recv_q.get(timeout=30)
        except queue.Empty:
            rp.kill()
            raise RuntimeError("Receiver did not finish within 30 s after sender exited")
        if line is None:
            break
        recv_lines.append(line)
    rp.wait(timeout=10)
    if rp.returncode != 0:
        err = rp.stderr.read()
        raise RuntimeError(
            f"Receiver exited with code {rp.returncode}.\nstderr:\n{err}"
        )

    # --- Parse metrics ---
    sd = parse_done_line(sender_out.splitlines(), "SENDER_DONE")
    rd = parse_done_line(recv_lines, "RECEIVER_DONE")

    bytes_sent = int(sd.get("bytes", 0))
    elapsed = float(sd.get("elapsed", 1))
    bytes_recv = int(rd.get("bytes", 0))
    messages_recv = int(rd.get("messages", 0))

    TOKEN_COUNT = 10_000
    CHUNK_SIZE = 128
    n_full = TOKEN_COUNT // CHUNK_SIZE
    n_rem = TOKEN_COUNT % CHUNK_SIZE
    expected_messages = n_full + (1 if n_rem else 0)

    # --- Verify files match (byte-for-byte) ---
    ref_data = pathlib.Path(ref).read_bytes()
    recv_data = pathlib.Path(recv).read_bytes()
    files_match = ref_data == recv_data

    no_dropped = (messages_recv == expected_messages) and files_match
    throughput_gbps = bytes_sent / elapsed / 1e9
    throughput_pass = throughput_gbps >= 1.0

    result = {
        "test": "activation-streaming-localhost",
        "date": "2026-07-29",
        "model": "meta-llama/Llama-3.2-3B",
        "weights": "synthetic-deterministic-seed42",
        "note": (
            "Activations are synthetic (fixed-seed random float16 in Llama-3.2-3B "
            "layer-16 shape). Real model weights not loaded; the tested properties "
            "are protocol correctness and localhost streaming throughput."
        ),
        "layer": 16,
        "tokens": TOKEN_COUNT,
        "hidden_dim": 3072,
        "chunk_size": CHUNK_SIZE,
        "transport": "TCP 127.0.0.1:9876",
        "protocol_version": "1.0.0",
        "bytes_transferred": bytes_sent,
        "bytes_received": bytes_recv,
        "expected_messages": expected_messages,
        "messages_received": messages_recv,
        "elapsed_seconds": round(elapsed, 6),
        "throughput_gbps": round(throughput_gbps, 3),
        "throughput_target_gbps": 1.0,
        "throughput_pass": throughput_pass,
        "files_match": files_match,
        "no_dropped_messages": no_dropped,
        "pass": files_match and no_dropped and throughput_pass,
    }

    out = base / "activation-streaming-localhost.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    if not result["pass"]:
        print("\nFAILED:", file=sys.stderr)
        if not files_match:
            print("  - file mismatch: received data != reference capture", file=sys.stderr)
        if not no_dropped:
            print(
                f"  - dropped messages: expected {expected_messages}, got {messages_recv}",
                file=sys.stderr,
            )
        if not throughput_pass:
            print(
                f"  - throughput {throughput_gbps:.3f} GB/s < 1.0 GB/s target",
                file=sys.stderr,
            )
        sys.exit(1)


main()
