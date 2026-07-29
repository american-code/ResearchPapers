#!/usr/bin/env python3
"""
Node A — Llama-3.2-3B layers 0 to LAYER_SPLIT-1.

Reads token IDs, runs forward pass through first half of layers, saves the
handoff activations locally, then streams them to Node B via Unix domain socket
using the mlxMesh wire protocol (8-byte header + big-endian float16 payload).

Args (JSON, sys.argv[1]):
  socket_path     Unix socket path Node B is listening on
  token_ids_path  .npy file of int32 token IDs
  handoff_out     .npy output path for layer-(LAYER_SPLIT-1) activations
  layer_split     first layer NOT run by A (A runs 0..layer_split-1)
  chunk_size      tokens per streaming message (default 128)
"""
import json, struct, socket, sys, time, uuid
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"
HIDDEN_DIM = 3072
WINDOW_SIZE = 8


def recv_exact(sock, n):
    buf = bytearray(n)
    mv = memoryview(buf)
    pos = 0
    while pos < n:
        got = sock.recv_into(mv[pos:], n - pos)
        if not got:
            raise EOFError("connection closed")
        pos += got
    return bytes(buf)


def main():
    args = json.loads(sys.argv[1])
    socket_path = args["socket_path"]
    token_ids_path = args["token_ids_path"]
    handoff_out = args["handoff_out"]
    layer_split = args["layer_split"]
    chunk_size = args.get("chunk_size", 128)

    t0 = time.time()
    token_ids = np.load(token_ids_path)
    n_tokens = len(token_ids)
    print(f"NODE_A: {n_tokens} tokens, layer_split={layer_split}", flush=True)

    print(f"NODE_A: loading {MODEL_ID} ...", flush=True)
    model, _ = load(MODEL_ID)
    num_layers = len(model.model.layers)
    assert layer_split <= num_layers, f"layer_split={layer_split} > num_layers={num_layers}"
    print(f"NODE_A: model loaded ({num_layers} layers), running 0-{layer_split-1}", flush=True)

    # Forward pass: embed + layers 0..layer_split-1
    llama = model.model
    inp = mx.array([token_ids.tolist()])       # [1, n_tokens]
    h = llama.embed_tokens(inp)                # [1, n_tokens, hidden_dim]

    fa_mask = create_attention_mask(h, None)
    swa_mask = None
    if hasattr(llama, "swa_idx") and llama.swa_idx is not None:
        swa_mask = create_attention_mask(h, None, window_size=llama.sliding_window)

    for i, layer in enumerate(llama.layers):
        if i >= layer_split:
            break
        mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
        h = layer(h, mask, cache=None)

    # Materialise: bfloat16 → float32 → float16 numpy  [n_tokens, hidden_dim]
    h32 = h[0].astype(mx.float32)
    mx.eval(h32)
    handoff_np = np.array(h32, dtype=np.float16)
    np.save(handoff_out, handoff_np)
    print(f"NODE_A: handoff saved ({handoff_np.shape})", flush=True)

    # Connect to Node B
    print(f"NODE_A: connecting to {socket_path} ...", flush=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    # Handshake (length-prefixed JSON, big-endian uint32 length)
    session_id = str(uuid.uuid4())
    hello = json.dumps({
        "protocol_version": "1.0.0",
        "session_id": session_id,
        "model_id": MODEL_ID,
        "num_layers": num_layers,
        "hidden_dim": HIDDEN_DIM,
        "dtype": "float16",
        "byte_order": "big",
        "window_size": WINDOW_SIZE,
        "handoff_layer": layer_split - 1,
        "n_tokens": n_tokens,
    }).encode()
    sock.sendall(struct.pack(">I", len(hello)) + hello)

    length = struct.unpack(">I", recv_exact(sock, 4))[0]
    reply = json.loads(recv_exact(sock, length).decode())
    assert reply.get("accepted"), f"Node B rejected session: {reply}"
    credit = reply["window_size"]

    # Stream: big-endian float16 payload
    act_be = handoff_np.astype(">f2").tobytes()
    act_mem = memoryview(act_be)
    layer_idx = layer_split - 1
    token_offset = 0

    while token_offset < n_tokens:
        n = min(chunk_size, n_tokens - token_offset)
        hdr = struct.pack(">HIH", layer_idx, token_offset, n)
        sb = token_offset * HIDDEN_DIM * 2
        eb = (token_offset + n) * HIDDEN_DIM * 2

        if credit <= 0:
            raw = recv_exact(sock, 4)
            ack_type, c = struct.unpack(">HH", raw)
            if ack_type == 0x0001:
                credit += c

        credit -= 1
        sock.sendall(hdr)
        sock.sendall(bytes(act_mem[sb:eb]))
        token_offset += n

    # End-of-layer sentinel, then end-of-stream sentinel
    sock.sendall(struct.pack(">HIH", 0xFFFF, layer_idx, 0xFFFF))
    sock.sendall(struct.pack(">HIH", 0xFFFF, 0xFFFFFFFF, 0xFFFF))

    # Drain remaining ACKs before close
    sock.shutdown(socket.SHUT_WR)
    while True:
        try:
            data = sock.recv(256)
            if not data:
                break
        except OSError:
            break
    sock.close()

    elapsed = time.time() - t0
    print(f"NODE_A_DONE elapsed={elapsed:.1f}s n_tokens={n_tokens}", flush=True)


main()
