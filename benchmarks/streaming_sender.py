"""
Activation streaming protocol v1.0.0 — sender process.
Generates deterministic synthetic float16 activations matching Llama-3.2-3B
layer 16 shape (10k tokens × 3072 hidden dim), saves a big-endian reference
file, then streams over TCP with credit-based backpressure.

Single-threaded design: ACKs are drained non-blocking between sends via
select(); only blocks on a direct recv() when credit is fully depleted.
"""
import socket
import select
import struct
import json
import sys
import time
import uuid
import pathlib
import numpy as np

HOST = "127.0.0.1"
PORT = 9876
HIDDEN_DIM = 3072
TOKEN_COUNT = 10_000
CHUNK_SIZE = 128
LAYER_IDX = 16
SEED = 42
# 128 > 79 total messages → credit never depleted → no blocking recv during send
WINDOW = 128


def recv_exact(sock, n):
    buf = bytearray(n)
    v = memoryview(buf)
    pos = 0
    while pos < n:
        got = sock.recv_into(v[pos:], n - pos)
        if not got:
            raise EOFError
        pos += got
    return bytes(buf)


def drain_acks(sock, credit):
    """Non-blocking: read all pending ACKs and return updated credit."""
    while select.select([sock], [], [], 0)[0]:
        try:
            raw = sock.recv(256)   # up to 64 ACKs × 4 bytes per call
        except OSError:
            break
        if not raw:
            break
        for i in range(0, len(raw) - 3, 4):
            ack_type, c = struct.unpack(">HH", raw[i : i + 4])
            if ack_type == 0x0001:
                credit += c
    return credit


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref_activations.bin"

    # Synthetic but deterministic activations — same shape as Llama-3.2-3B layer 16
    rng = np.random.default_rng(SEED)
    activations = rng.standard_normal((TOKEN_COUNT, HIDDEN_DIM)).astype(np.float16)

    # Pre-convert to big-endian once; use a memoryview for zero-copy chunk slicing
    act_be = activations.astype(">f2").tobytes()
    act_mem = memoryview(act_be)

    # Save reference in big-endian (matches wire format, so comparison is direct)
    pathlib.Path(ref_path).write_bytes(act_be)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    sock.connect((HOST, PORT))

    # Handshake
    session_id = str(uuid.uuid4())
    hello = json.dumps({
        "protocol_version": "1.0.0",
        "session_id": session_id,
        "model_id": "meta-llama/Llama-3.2-3B",
        "num_layers": 28,
        "hidden_dim": HIDDEN_DIM,
        "dtype": "float16",
        "byte_order": "big",
        "window_size": WINDOW,
    }).encode()
    sock.sendall(struct.pack(">I", len(hello)) + hello)

    length = struct.unpack(">I", recv_exact(sock, 4))[0]
    reply = json.loads(recv_exact(sock, length).decode())
    assert reply.get("accepted"), f"Receiver rejected session: {reply}"
    window = reply["window_size"]

    credit = window
    bytes_sent = 0
    t0 = time.perf_counter()

    token_start = 0
    while token_start < TOKEN_COUNT:
        n = min(CHUNK_SIZE, TOKEN_COUNT - token_start)
        hdr = struct.pack(">HIH", LAYER_IDX, token_start, n)
        sb = token_start * HIDDEN_DIM * 2
        eb = (token_start + n) * HIDDEN_DIM * 2

        # Opportunistically drain pending ACKs (non-blocking)
        credit = drain_acks(sock, credit)

        if credit <= 0:
            # Window depleted: block until at least one ACK arrives
            raw = recv_exact(sock, 4)
            ack_type, c = struct.unpack(">HH", raw)
            if ack_type == 0x0001:
                credit += c

        credit -= 1
        sock.sendmsg([hdr, act_mem[sb:eb]])
        bytes_sent += 8 + (eb - sb)
        token_start += n

    # EOL sentinel for layer LAYER_IDX
    sock.sendall(struct.pack(">HIH", 0xFFFF, LAYER_IDX, 0xFFFF))
    # EOS sentinel
    sock.sendall(struct.pack(">HIH", 0xFFFF, 0xFFFFFFFF, 0xFFFF))

    elapsed = time.perf_counter() - t0

    # Half-close write side; drain remaining ACKs so receiver's sendall doesn't
    # hit BrokenPipeError before it finishes processing the last few messages.
    sock.shutdown(socket.SHUT_WR)
    while True:
        try:
            data = sock.recv(256)
            if not data:
                break
        except OSError:
            break
    sock.close()

    print(f"SENDER_DONE bytes={bytes_sent} elapsed={elapsed:.6f}", flush=True)


main()
