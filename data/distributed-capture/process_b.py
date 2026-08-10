#!/usr/bin/env python3
"""
Node B — Llama-3.2-3B layers LAYER_SPLIT to num_layers-1.

Binds Unix socket, loads model, then accepts one connection from Node A.
Receives the handoff activation stream (ActStream wire protocol), runs the
second half of the model, saves final-layer activations.

Args (JSON, sys.argv[1]):
  socket_path   Unix socket path to bind and listen on
  final_out     .npy output path for final-layer activations
  layer_split   first layer run by B (B runs layer_split..num_layers-1)
"""
import json, struct, socket, sys, pathlib, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"
HIDDEN_DIM = 3072
ACK_BATCH = 4


def main():
    args = json.loads(sys.argv[1])
    socket_path = args["socket_path"]
    final_out = args["final_out"]
    layer_split = args["layer_split"]

    t0 = time.time()

    # Bind socket before loading model so the OS starts queuing the connection
    # from Node A; data will buffer in the kernel until we call accept().
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    sock_path = pathlib.Path(socket_path)
    sock_path.unlink(missing_ok=True)
    srv.bind(socket_path)
    srv.listen(1)
    # Coordinator starts Node A only after seeing NODE_B_MODEL_READY, but the
    # socket is already listening in case of any race.

    print(f"NODE_B: loading {MODEL_ID} ...", flush=True)
    model, _ = load(MODEL_ID)
    num_layers = len(model.model.layers)
    print(f"NODE_B_MODEL_READY layers={layer_split}-{num_layers-1}", flush=True)

    conn, _ = srv.accept()
    srv.close()
    print(f"NODE_B: accepted connection from Node A", flush=True)

    # ── Handshake ─────────────────────────────────────────────────────────────

    def recv_exact(n):
        buf = bytearray(n)
        mv = memoryview(buf)
        pos = 0
        while pos < n:
            got = conn.recv_into(mv[pos:], n - pos)
            if not got:
                raise EOFError("connection closed")
            pos += got
        return bytes(buf)

    length = struct.unpack(">I", recv_exact(4))[0]
    hello = json.loads(recv_exact(length).decode())
    n_tokens = hello["n_tokens"]
    hidden_dim = hello["hidden_dim"]
    window = min(hello["window_size"], 8)

    reply = json.dumps({
        "protocol_version": "1.0.0",
        "session_id": hello["session_id"],
        "accepted": True,
        "window_size": window,
    }).encode()
    conn.sendall(struct.pack(">I", len(reply)) + reply)

    # ── Receive activation stream ─────────────────────────────────────────────

    stream = conn.makefile("rb", buffering=4 * 1024 * 1024)

    def read_exact(n):
        data = stream.read(n)
        if len(data) < n:
            raise EOFError(f"stream closed after {len(data)}/{n} bytes")
        return data

    chunks = []
    expected = {}     # layer_idx → expected next token_start
    pending_acks = 0

    while True:
        hdr = read_exact(8)
        layer_idx, token_start, token_count = struct.unpack(">HIH", hdr)

        if layer_idx == 0xFFFF and token_start == 0xFFFFFFFF:
            break   # end-of-stream sentinel
        if layer_idx == 0xFFFF:
            continue  # end-of-layer sentinel

        exp = expected.get(layer_idx, 0)
        if token_start != exp:
            raise ValueError(f"gap layer={layer_idx}: expected {exp}, got {token_start}")
        expected[layer_idx] = token_start + token_count

        payload = read_exact(token_count * hidden_dim * 2)
        chunks.append(payload)

        pending_acks += 1
        if pending_acks >= ACK_BATCH:
            try:
                conn.sendall(struct.pack(">HH", 0x0001, pending_acks))
            except OSError:
                pass
            pending_acks = 0

    if pending_acks:
        try:
            conn.sendall(struct.pack(">HH", 0x0001, pending_acks))
        except OSError:
            pass

    stream.close()
    conn.close()
    print(f"NODE_B: received {len(chunks)} chunks, {n_tokens} tokens", flush=True)

    # ── Reconstruct handoff tensor ────────────────────────────────────────────

    raw = b"".join(chunks)
    # Big-endian float16 → native float32 → MLX bfloat16  [1, n_tokens, hidden_dim]
    handoff_np = np.frombuffer(raw, dtype=">f2").reshape(n_tokens, hidden_dim).astype(np.float32)
    h = mx.array(handoff_np, dtype=mx.bfloat16).reshape(1, n_tokens, hidden_dim)

    # ── Run layers layer_split..num_layers-1 ──────────────────────────────────

    llama = model.model
    fa_mask = create_attention_mask(h, None)
    swa_mask = None
    if hasattr(llama, "swa_idx") and llama.swa_idx is not None:
        swa_mask = create_attention_mask(h, None, window_size=llama.sliding_window)

    for i, layer in enumerate(llama.layers):
        if i < layer_split:
            continue
        mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
        h = layer(h, mask, cache=None)

    # Materialise output: bfloat16 → float32 → float16 numpy
    h32 = h[0].astype(mx.float32)
    mx.eval(h32)
    final_np = np.array(h32, dtype=np.float16)   # [n_tokens, hidden_dim]
    np.save(final_out, final_np)

    elapsed = time.time() - t0
    print(f"NODE_B_DONE elapsed={elapsed:.1f}s n_tokens={n_tokens} layers={layer_split}-{num_layers-1}", flush=True)


main()
