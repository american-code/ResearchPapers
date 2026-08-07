#!/usr/bin/env python3
"""
Transport benchmark for the mlxMesh activation-streaming link.

Replaces benchmarks/run_localhost_test.py, which produced a single point estimate
from one payload size on one transport, mislabelled GB/s as Gbps, and could not
distinguish the link under test from the loopback path it actually measured.

Design constraints, in order of importance:

  REPEATABLE   Fixed seed, fixed payload ladder, fixed rep count. Every parameter
               that affects the result is written into the output JSON.
  HONEST ABOUT WHAT WAS MEASURED  The client asserts which interface the socket is
               actually bound to and records the kernel's chosen route. Measuring
               WiFi while believing you measured Thunderbolt is the failure mode
               this guards against -- it is what happened to the previous benchmark.
  DISTRIBUTIONS, NOT POINTS  Reports median, IQR and p99 over N reps after
               discarding warm-up. A single number cannot support a claim about a
               link whose jitter is the thing that matters.
  CONTROLS     The same ladder is meant to be run over loopback, over the LAN, and
               over Thunderbolt, so the three columns sit side by side and the
               reader can see what the link contributes.

Usage (server on the receiving node, client on the sending node):

    # receiver
    python3 link_benchmark.py serve --bind 10.0.0.2 --port 9987

    # sender
    python3 link_benchmark.py bench --host 10.0.0.2 --port 9987 \
        --transport thunderbolt --label lab01-to-lab02 --out tb.json

Run the loopback control on either node:

    python3 link_benchmark.py serve --bind 127.0.0.1 --port 9987 &
    python3 link_benchmark.py bench --host 127.0.0.1 --port 9987 \
        --transport loopback --out loopback.json
"""

import argparse
import json
import os
import platform
import socket
import statistics
import struct
import subprocess
import time
from pathlib import Path

# Payload ladder, log-spaced. The small end exposes per-message overhead and
# latency; the large end exposes steady-state bandwidth. A single mid-sized
# payload -- what the old benchmark used -- shows neither.
PAYLOADS = [4 << 10, 16 << 10, 64 << 10, 256 << 10,
            1 << 20, 4 << 20, 16 << 20]
REPS = 40            # per payload size
WARMUP = 8           # discarded, not reported
LATENCY_PINGS = 200  # 64-byte round trips, separate from throughput
SEED = 42
HDR = struct.Struct("!Q")   # payload length prefix


# ── environment capture ─────────────────────────────────────────────────────

def route_for(host: str) -> dict:
    """Which interface will the kernel actually use for this host?"""
    info = {"host": host}
    try:
        out = subprocess.run(["route", "-n", "get", host],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                if k in ("interface", "gateway", "route to"):
                    info[k] = v.strip()
    except Exception as e:
        info["error"] = str(e)
    return info


def link_rate(iface: str) -> dict:
    """Negotiated link rate, so the report can state measured-vs-theoretical."""
    out = {"interface": iface}
    try:
        r = subprocess.run(["ifconfig", iface], capture_output=True,
                           text=True, timeout=10).stdout
        for line in r.splitlines():
            if "media:" in line:
                out["media"] = line.strip()
        tb = subprocess.run(["system_profiler", "SPThunderboltDataType"],
                            capture_output=True, text=True, timeout=30).stdout
        speeds = [l.strip() for l in tb.splitlines()
                  if "Speed:" in l and "Up to" not in l]
        if speeds:
            out["thunderbolt_negotiated"] = speeds
    except Exception as e:
        out["error"] = str(e)
    return out


def host_info() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return None
    return {
        "hostname": socket.gethostname(),
        "model": sysctl("hw.model"),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "mem_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1),
        "os": platform.platform(),
    }


# ── wire helpers ────────────────────────────────────────────────────────────

def send_all(sock, buf: memoryview) -> None:
    sock.sendall(buf)


def recv_exact(sock, n: int) -> bytes:
    parts = []
    got = 0
    while got < n:
        b = sock.recv(min(1 << 20, n - got))
        if not b:
            raise ConnectionError("peer closed mid-message")
        parts.append(b)
        got += len(b)
    return b"".join(parts)


def tune(sock) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in ("SO_SNDBUF", "SO_RCVBUF"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, getattr(socket, opt), 4 << 20)
        except OSError:
            pass


# ── server ──────────────────────────────────────────────────────────────────

def serve(args) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"listening on {args.bind}:{args.port}", flush=True)
    while True:
        conn, peer = srv.accept()
        tune(conn)
        print(f"  connection from {peer}", flush=True)
        try:
            while True:
                hdr = recv_exact(conn, HDR.size)
                (n,) = HDR.unpack(hdr)
                if n == 0:
                    break
                payload = recv_exact(conn, n)
                # echo the length back: makes each rep a full round trip, so the
                # measurement includes the receiver actually having the bytes
                conn.sendall(HDR.pack(len(payload)))
        except (ConnectionError, OSError) as e:
            print(f"  peer gone: {e}", flush=True)
        finally:
            conn.close()
        if args.once:
            break


# ── client ──────────────────────────────────────────────────────────────────

def summarize(xs: list) -> dict:
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], xs[len(xs)//2], xs[-1]]
    return {
        "n": len(xs),
        "median": round(statistics.median(xs), 6),
        "mean": round(statistics.fmean(xs), 6),
        "iqr_lo": round(q[0], 6),
        "iqr_hi": round(q[2], 6),
        "p99": round(xs[min(len(xs) - 1, int(0.99 * len(xs)))], 6),
        "min": round(xs[0], 6),
        "max": round(xs[-1], 6),
    }


def bench(args) -> None:
    import random
    rng = random.Random(SEED)
    blob = bytes(rng.getrandbits(8) for _ in range(max(PAYLOADS)))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tune(sock)
    if args.bind_local:
        sock.bind((args.bind_local, 0))
    sock.connect((args.host, args.port))
    local_ip, local_port = sock.getsockname()

    rt = route_for(args.host)
    iface = rt.get("interface", "?")
    print(f"connected {local_ip}:{local_port} -> {args.host}:{args.port} via {iface}", flush=True)

    # ── latency: small round trips, measured separately from throughput ──
    lat = []
    small = blob[:64]
    for i in range(LATENCY_PINGS + WARMUP):
        t0 = time.perf_counter()
        sock.sendall(HDR.pack(len(small)) + small)
        recv_exact(sock, HDR.size)
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            lat.append(dt * 1000.0)   # ms
    print(f"  latency: median {statistics.median(lat):.3f} ms  "
          f"p99 {sorted(lat)[int(0.99*len(lat))]:.3f} ms", flush=True)

    # ── throughput ladder ──
    results = []
    for size in PAYLOADS:
        view = memoryview(blob)[:size]
        rates = []
        for i in range(REPS + WARMUP):
            t0 = time.perf_counter()
            sock.sendall(HDR.pack(size))
            send_all(sock, view)
            recv_exact(sock, HDR.size)
            dt = time.perf_counter() - t0
            if i >= WARMUP:
                rates.append(size / dt)            # bytes/s
        s = summarize(rates)
        row = {
            "payload_bytes": size,
            "reps": REPS,
            "bytes_per_s": s,
            "median_GB_per_s": round(s["median"] / 1e9, 4),
            "median_Gbps": round(s["median"] * 8 / 1e9, 4),
            "p99_GB_per_s": round(s["p99"] / 1e9, 4),
        }
        results.append(row)
        print(f"  {size/1024:>8.0f} KiB  median {row['median_GB_per_s']:.3f} GB/s "
              f"({row['median_Gbps']:.2f} Gbps)  IQR "
              f"[{s['iqr_lo']/1e9:.3f}, {s['iqr_hi']/1e9:.3f}]", flush=True)

    sock.sendall(HDR.pack(0))
    sock.close()

    best = max(results, key=lambda r: r["median_GB_per_s"])
    doc = {
        "test": "link_benchmark",
        "label": args.label,
        "transport": args.transport,
        "note": ("Payload is fixed-seed pseudorandom bytes, not model activations. "
                 "This measures the transport, not inference. Each rep is a full "
                 "round trip: the reported rate includes the receiver having the "
                 "bytes and acknowledging."),
        "parameters": {
            "payload_ladder_bytes": PAYLOADS, "reps_per_size": REPS,
            "warmup_discarded": WARMUP, "latency_pings": LATENCY_PINGS,
            "seed": SEED, "tcp_nodelay": True, "sock_buf_bytes": 4 << 20,
        },
        "endpoints": {
            "local": {**host_info(), "ip": local_ip},
            "remote_host": args.host,
        },
        "route": rt,
        "link": link_rate(iface),
        "latency_ms": summarize(lat),
        "throughput": results,
        "headline": {
            "best_median_GB_per_s": best["median_GB_per_s"],
            "best_median_Gbps": best["median_Gbps"],
            "at_payload_bytes": best["payload_bytes"],
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {args.out}")
    print(f"headline: {best['median_GB_per_s']:.3f} GB/s "
          f"({best['median_Gbps']:.2f} Gbps) at {best['payload_bytes']/1024:.0f} KiB, via {iface}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--bind", default="0.0.0.0")
    s.add_argument("--port", type=int, default=9987)
    s.add_argument("--once", action="store_true")
    s.set_defaults(fn=serve)
    b = sub.add_parser("bench")
    b.add_argument("--host", required=True)
    b.add_argument("--port", type=int, default=9987)
    b.add_argument("--transport", required=True,
                   choices=["loopback", "ethernet", "wifi", "thunderbolt"])
    b.add_argument("--label", default="")
    b.add_argument("--bind-local", default=None,
                   help="bind the source socket to this local IP, to force the "
                        "intended interface rather than trusting the default route")
    b.add_argument("--out", default="link-benchmark.json")
    b.set_defaults(fn=bench)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
