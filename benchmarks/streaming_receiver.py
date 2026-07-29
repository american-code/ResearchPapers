"""
Activation streaming protocol v1.0.0 — receiver process.
Listens on 127.0.0.1:9876, performs handshake, receives float16 payloads,
validates sequence numbers, sends DATA_ACKs, saves raw payload bytes to file.
"""
import socket
import struct
import json
import sys
import time
import pathlib

HOST = "127.0.0.1"
PORT = 9876

# Batch N messages per ACK; reduces ACK syscall count while keeping backpressure
ACK_BATCH = 4


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/recv_activations.bin"

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print("RECEIVER_READY", flush=True)

    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    srv.close()

    # Handshake (small messages: use conn directly)
    def recv_exact_raw(n):
        buf = bytearray(n)
        v = memoryview(buf)
        pos = 0
        while pos < n:
            got = conn.recv_into(v[pos:], n - pos)
            if not got:
                raise EOFError
            pos += got
        return bytes(buf)

    length = struct.unpack(">I", recv_exact_raw(4))[0]
    hello = json.loads(recv_exact_raw(length).decode())
    hidden_dim = hello["hidden_dim"]
    window_size = min(hello["window_size"], 128)

    reply = json.dumps({
        "protocol_version": "1.0.0",
        "session_id": hello["session_id"],
        "accepted": True,
        "window_size": window_size,
    }).encode()
    conn.sendall(struct.pack(">I", len(reply)) + reply)

    # Wrap the socket in a large buffered reader so multiple messages can be
    # served from a single kernel recv call (4 MB = ~5 messages of 786 KB each).
    stream = conn.makefile("rb", buffering=4 * 1024 * 1024)

    def read_exact(n):
        data = stream.read(n)
        if len(data) < n:
            raise EOFError(f"stream closed after {len(data)}/{n} bytes")
        return data

    # Receive data messages
    chunks = []
    expected = {}   # layer_idx -> expected next token_start
    bytes_rx = 0
    messages = 0
    pending_acks = 0   # messages received but not yet ACK'd
    t0 = time.perf_counter()

    while True:
        hdr = read_exact(8)
        layer_idx, token_start, token_count = struct.unpack(">HIH", hdr)

        # EOS sentinel: layer_idx=0xFFFF, token_start=0xFFFFFFFF
        if layer_idx == 0xFFFF and token_start == 0xFFFFFFFF:
            break
        # EOL sentinel: layer_idx=0xFFFF, token_start=<closed layer>
        if layer_idx == 0xFFFF:
            continue

        exp = expected.get(layer_idx, 0)
        if token_start != exp:
            raise ValueError(
                f"Sequence gap layer={layer_idx}: expected {exp}, got {token_start}"
            )
        expected[layer_idx] = token_start + token_count

        payload = read_exact(token_count * hidden_dim * 2)
        bytes_rx += 8 + len(payload)
        messages += 1
        chunks.append(payload)

        # Batch ACKs: send one DATA_ACK with accumulated credit every ACK_BATCH messages
        pending_acks += 1
        if pending_acks >= ACK_BATCH:
            try:
                conn.sendall(struct.pack(">HH", 0x0001, pending_acks))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            pending_acks = 0

    # Flush remaining ACKs
    if pending_acks:
        try:
            conn.sendall(struct.pack(">HH", 0x0001, pending_acks))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    elapsed = time.perf_counter() - t0
    stream.close()
    conn.close()

    pathlib.Path(out_path).write_bytes(b"".join(chunks))
    print(
        f"RECEIVER_DONE bytes={bytes_rx} elapsed={elapsed:.6f} messages={messages}",
        flush=True,
    )


main()
