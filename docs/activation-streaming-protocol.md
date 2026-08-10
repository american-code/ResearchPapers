# ActStream Activation Streaming Wire Protocol

Version: 1.0.0  
Status: Draft  
Date: 2026-07-29

---

## 1. Overview

The ActStream activation streaming protocol transmits intermediate layer activations between distributed inference nodes in real time. It is designed for low-latency, high-throughput delivery of float16 tensor payloads across a local network or shared-memory IPC channel, with explicit backpressure and clean failure semantics.

The protocol is message-oriented and operates over a reliable, ordered byte stream (e.g. TCP, Unix domain socket, or RDMA reliable connection). Message delivery ordering is guaranteed by the transport; the protocol adds sequence numbers for gap detection and session integrity checks only.

---

## 2. Message Structure

Every message consists of a fixed 8-byte header followed by a variable-length float16 payload.

### 2.1 Header Layout

```
Offset  Size  Type     Field
------  ----  -------  ----------------------------------------
0       2     uint16   layer_idx       (big-endian)
2       4     uint32   token_start     (big-endian)
6       2     uint16   token_count     (big-endian)
```

Total header size: **8 bytes**.

#### Field Semantics

| Field | Range | Description |
|---|---|---|
| `layer_idx` | 0–65534 | Zero-based transformer layer index. Value `0xFFFF` is reserved as the End-of-Layer sentinel (see §4). |
| `token_start` | 0–4294967294 | Absolute token offset within the current sequence. Identifies which position in the token stream this chunk starts at. Value `0xFFFFFFFF` is reserved. |
| `token_count` | 1–65534 | Number of tokens in this chunk. Zero is invalid. Value `0xFFFF` is reserved. |

### 2.2 Payload Layout

Immediately following the 8-byte header is a contiguous float16 tensor payload:

```
payload_bytes = token_count × hidden_dim × 2
```

where `hidden_dim` is established during session handshake (see §3). The tensor is laid out in row-major order: token 0 row first, then token 1, etc. Each float16 value is encoded in IEEE 754 half-precision, big-endian byte order.

**Total message size** = 8 + (token_count × hidden_dim × 2) bytes.

### 2.3 Example

For a model with `hidden_dim = 4096`, streaming tokens 0–127 of layer 16:

```
header:  layer_idx=0x0010  token_start=0x00000000  token_count=0x0080
payload: 128 × 4096 × 2 = 1,048,576 bytes  (1 MiB)
total:   1,048,584 bytes
```

---

## 3. Session Handshake

Before streaming begins, sender and receiver exchange a JSON handshake over the same transport channel. The handshake frame is length-prefixed with a 4-byte big-endian uint32 indicating the JSON byte length.

### 3.1 Sender Hello

```json
{
  "protocol_version": "1.0.0",
  "session_id": "<uuid4>",
  "model_id": "meta-llama/Llama-3.2-3B",
  "num_layers": 28,
  "hidden_dim": 3072,
  "dtype": "float16",
  "byte_order": "big",
  "window_size": 8
}
```

`window_size`: maximum number of in-flight (unacknowledged) messages the sender will issue before blocking (see §6).

### 3.2 Receiver Hello

```json
{
  "protocol_version": "1.0.0",
  "session_id": "<same uuid4>",
  "accepted": true,
  "window_size": 4
}
```

The negotiated window size is `min(sender.window_size, receiver.window_size)`.

If `accepted` is `false`, the receiver includes an `"error"` string and closes the connection.

---

## 4. Sequence Numbering

Each data message carries an implicit sequence number derived from the `(layer_idx, token_start)` pair. Within a session:

- Messages for a given layer MUST be sent in ascending `token_start` order.
- Consecutive chunks for the same layer MUST be contiguous: `next.token_start == prev.token_start + prev.token_count`.
- Layers MUST be sent in ascending `layer_idx` order (layer N fully sent before layer N+1 begins).

### 4.1 Sequence Number Wire Field

A monotonically increasing 32-bit session sequence number is appended to the header in the extended header format (v1.1+). In v1.0, the `(layer_idx, token_start)` tuple is the identity; receivers detect gaps by checking the expected `token_start` after each chunk.

### 4.2 Gap Detection

The receiver tracks `expected_token_start[layer_idx]`. On each arriving message:

```
if message.token_start != expected_token_start[layer_idx]:
    → protocol error: send NAK, close session
expected_token_start[layer_idx] += message.token_count
```

Since the transport is ordered and reliable, a gap indicates a framing bug or session corruption, not packet loss; the correct response is session termination rather than retransmission.

---

## 5. End-of-Layer Sentinel

After all chunks for a layer have been sent, the sender MUST emit a sentinel message to signal layer completion. The sentinel uses the reserved field values:

```
layer_idx   = 0xFFFF   (reserved sentinel marker)
token_start = layer_idx being closed, encoded as uint32
token_count = 0xFFFF   (sentinel marker)
payload     = empty (0 bytes)
```

To close layer 16:
```
header: 0xFFFF  0x00000010  0xFFFF
```

The receiver MUST NOT process subsequent messages for a layer after receiving its sentinel. Receiving a sentinel for a layer that has already been closed is a protocol error.

A session-level end-of-stream sentinel uses `token_start = 0xFFFFFFFF`, signaling that all layers have been sent and the sender will close the connection.

---

## 6. Backpressure

The protocol uses a credit-based window to prevent the sender from overwhelming the receiver's buffer or downstream compute pipeline.

### 6.1 Acknowledgment Messages

The receiver sends ACK messages upstream after consuming (or buffering) each data message. ACK messages travel in the reverse direction on the same connection and are 4 bytes:

```
Offset  Size  Type     Field
------  ----  -------  ----------------------------------------
0       2     uint16   ack_type   0x0001 = DATA_ACK, 0x0002 = CREDIT_PAUSE, 0x0003 = CREDIT_RESUME
2       2     uint16   credit     number of additional messages sender may now issue (for DATA_ACK)
```

### 6.2 Flow Control Rules

1. **Sender rule**: At session start, the sender's credit counter equals the negotiated `window_size`. For each message sent, decrement the counter by 1. Block (do not send) when the counter reaches 0. Resume when a DATA_ACK arrives and increments the counter.

2. **DATA_ACK semantics**: The `credit` field indicates how many new messages the receiver is granting. Typically `credit = 1` (one-for-one acking), but the receiver MAY batch ACKs and return `credit > 1` to indicate it has freed multiple slots.

3. **CREDIT_PAUSE**: Receiver signals it cannot accept any new messages (e.g. GPU OOM, downstream backpressure). Sender MUST stop sending immediately and hold the current credit at 0 regardless of any previous DATA_ACK credits not yet applied.

4. **CREDIT_RESUME**: Receiver is ready again. Sender restores credit to the negotiated `window_size` and resumes.

5. **Timeout**: If the sender is blocked for more than `pause_timeout_ms` (default 5000 ms) with credit = 0 and no CREDIT_RESUME received, it emits a PING (see §7.2) and starts the dead-peer timer.

---

## 7. Control Messages

Control messages share the reverse channel used by ACKs. They are distinguished by `ack_type` values ≥ 0x0010.

```
0x0001  DATA_ACK        credit release (4 bytes, as above)
0x0002  CREDIT_PAUSE    stop sending
0x0003  CREDIT_RESUME   resume sending
0x0010  PING            liveness probe (sender → receiver or receiver → sender)
0x0011  PONG            liveness reply
0x0012  ERROR           fatal protocol error; payload is a UTF-8 error string (length-prefixed uint16)
0x0013  RESET           request orderly session teardown
```

### 7.1 PING / PONG

PING and PONG are 4-byte messages with `ack_type` set and `credit = 0`. The sender sets a 3-second PONG deadline; if no PONG arrives, the connection is declared dead and the session is aborted.

### 7.2 ERROR

```
Offset  Size  Field
0       2     ack_type = 0x0012
2       2     error_code (see §8)
4       2     message_length (bytes)
6       N     UTF-8 error string
```

After sending ERROR, the sender MUST close the write side of the connection. The receiver SHOULD drain remaining bytes then close.

### 7.3 RESET

Graceful teardown. The initiating side sends RESET, waits up to 1 second for the peer to echo RESET, then closes. If no echo arrives, closes unilaterally.

---

## 8. Error Codes

| Code | Hex | Meaning |
|---|---|---|
| `ERR_NONE` | 0x0000 | No error (informational) |
| `ERR_VERSION_MISMATCH` | 0x0001 | Incompatible protocol versions |
| `ERR_SESSION_MISMATCH` | 0x0002 | session_id in hello does not match |
| `ERR_DIM_MISMATCH` | 0x0003 | hidden_dim or num_layers disagreement |
| `ERR_GAP_DETECTED` | 0x0010 | token_start gap within a layer |
| `ERR_LAYER_OUT_OF_ORDER` | 0x0011 | layer_idx arrived before previous layer sentinel |
| `ERR_SENTINEL_DUPLICATE` | 0x0012 | Sentinel received for already-closed layer |
| `ERR_PAYLOAD_SIZE` | 0x0020 | Payload bytes ≠ token_count × hidden_dim × 2 |
| `ERR_RESERVED_FIELD` | 0x0021 | Reserved field value used in non-sentinel message |
| `ERR_CREDIT_OVERFLOW` | 0x0030 | Sender exceeded window; credit counter went negative |
| `ERR_TIMEOUT` | 0x0040 | Dead-peer timeout expired |
| `ERR_INTERNAL` | 0x00FF | Unclassified internal error |

---

## 9. Failure Modes and Recovery

### 9.1 Transport Disconnection

If the underlying connection drops mid-stream, both sides detect it via a read/write error or EOF. There is no in-protocol reconnection; the consumer layer must re-establish the session from scratch. Sessions are stateless once the handshake completes — the sender simply restarts from `layer_idx = 0, token_start = 0`.

### 9.2 Receiver OOM / Slow Consumer

The receiver issues CREDIT_PAUSE. The sender parks the next chunk in a one-message send buffer and blocks the inference pipeline at the upstream layer boundary. If the pause exceeds `max_pause_ms` (default 30 000 ms), the sender emits ERROR with `ERR_TIMEOUT` and terminates. The receiver is expected to recover within this window or close the connection itself.

### 9.3 Sender Crash Mid-Layer

The receiver detects EOF on the data channel before receiving the layer sentinel. It MUST discard all buffered activations for that layer and report an incomplete-layer error to its consumer. It SHOULD NOT attempt to use partial activations.

### 9.4 Corrupt Framing

If the receiver reads a message where `payload_bytes` would overflow a reasonable buffer (heuristic: > `token_count × hidden_dim × 2 + 64`), it treats this as framing corruption, sends ERROR `ERR_PAYLOAD_SIZE`, and closes. No attempt is made to resynchronize — the transport's reliability guarantee means corruption implies a framing bug.

### 9.5 Clock / Sequence Skew

Because sequence integrity is enforced via `(layer_idx, token_start)` monotonicity rather than wall-clock timestamps, there is no clock-skew failure mode. Timestamps are not used in this protocol version.

### 9.6 Partial Write on Sender Side

The sender MUST treat a short write as a fatal transport error. The protocol does not support partial message delivery; every message must be written atomically from the transport's perspective (i.e. using writev or an equivalent gather-write to send header + payload as a single syscall where possible).

---

## 10. Performance Considerations

### 10.1 Chunk Sizing

Optimal chunk size balances pipeline latency against per-message overhead. Recommended default: **128 tokens per chunk**. For `hidden_dim = 4096` (float16), this yields 1 MiB payloads — large enough for efficient DMA / socket send, small enough for sub-10 ms delivery on a 1 Gbps link.

Minimum meaningful chunk: 1 token. Maximum: `min(65534, sequence_length)`.

### 10.2 Zero-Copy Delivery

On macOS/Linux, senders SHOULD use `sendfile` or `MSG_ZEROCOPY` when the payload is backed by a file descriptor or a pinned MLX buffer. The protocol's flat, offset-free payload layout is intentionally compatible with zero-copy scatter-gather I/O.

### 10.3 Nagle Algorithm

Disable Nagle (`TCP_NODELAY`) on TCP sockets. The protocol already batches header + payload into a single write; Nagle's coalescing adds latency without throughput benefit.

### 10.4 Receiver Buffer Sizing

Set `SO_RCVBUF` to at least `window_size × max_message_bytes`. For `window_size = 4`, `hidden_dim = 4096`, `token_count = 128`:

```
4 × (8 + 1,048,576) = ~4 MiB
```

---

## 11. Versioning

The protocol version is negotiated during handshake. Minor version bumps (1.0 → 1.1) may add optional fields to the handshake JSON or new `ack_type` values but must not change the binary header layout. Major version bumps (1.x → 2.0) require a new handshake exchange and may change any aspect of the wire format.

A receiver that does not support the sender's major version MUST reject the session with `ERR_VERSION_MISMATCH`.

---

## 12. Wire Format Quick Reference

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA MESSAGE                                                        │
│  ┌──────────┬────────────────┬────────────┐                         │
│  │ layer_idx│  token_start   │token_count │  ← 8 bytes header       │
│  │  uint16  │    uint32      │  uint16    │                         │
│  └──────────┴────────────────┴────────────┘                         │
│  ┌──────────────────────────────────────────┐                       │
│  │  float16 payload: token_count × hidden_dim × 2 bytes            │
│  └──────────────────────────────────────────┘                       │
├──────────────────────────────────────────────────────────────────────┤
│  END-OF-LAYER SENTINEL                                               │
│  layer_idx=0xFFFF  token_start=<layer being closed>  token_count=0xFFFF  payload=empty │
├──────────────────────────────────────────────────────────────────────┤
│  END-OF-STREAM SENTINEL                                              │
│  layer_idx=0xFFFF  token_start=0xFFFFFFFF  token_count=0xFFFF  payload=empty           │
├──────────────────────────────────────────────────────────────────────┤
│  ACK / CONTROL (reverse channel, 4 bytes minimum)                   │
│  ┌────────────┬───────────┐                                         │
│  │  ack_type  │  credit   │                                         │
│  │  uint16    │  uint16   │                                         │
│  └────────────┴───────────┘                                         │
│  ERROR appends uint16 msg_len + UTF-8 string                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 13. Reference Implementation Notes

A compliant sender pseudocode sketch:

```python
credit = negotiated_window_size
for layer_idx in range(num_layers):
    for chunk in layer_chunks(layer_idx, token_chunk_size):
        while credit == 0:
            wait_for_ack()          # blocks; updates credit from DATA_ACK
        header = pack(">HLH", layer_idx, chunk.token_start, chunk.token_count)
        sock.sendmsg([header, chunk.data])   # gather-write: header + payload
        credit -= 1
    send_sentinel(layer_idx)        # layer_idx=0xFFFF, token_start=layer_idx, token_count=0xFFFF
send_eos_sentinel()                 # token_start=0xFFFFFFFF
```

ACK listener runs in a concurrent thread/task, updating `credit` and handling PAUSE/RESUME and PING/PONG.

---

*End of specification.*
