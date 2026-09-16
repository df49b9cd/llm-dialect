# Architecture

`llm-dialect` is the translation core of an LLM gateway: it converts between
the three major request/response wire dialects — Anthropic Messages
(`/v1/messages`), OpenAI Chat Completions (`/v1/chat/completions`), and OpenAI
Responses (`/v1/responses`) — through one canonical model, and frames streamed
turns back to the caller's dialect. Everything except a thin, feature-gated
SSE pump is pure data transformation: no async runtime, no HTTP types in the
API surface, no wall clock, no randomness.

## The three representations

Every turn passes through three shapes. Which one you hold depends on where
you sit:

| representation | type | who speaks it |
|---|---|---|
| **items canonical** | [`items::ItemRequest`](src/items.rs) — messages as typed, ordered content-item lists | the crate's own interchange model; what every `req.rs` parses into |
| **chat request** | [`canonical::ChatRequest`](src/canonical.rs) — legacy OpenAI chat shape (flat messages, tool calls on a side-array) | the engine/upstream side |
| **canonical chunks** | [`canonical::CanonChunk`](src/canonical.rs) — OpenAI-delta-shaped streaming events | the engine's stream output, before re-framing |

The items model is the hub: Anthropic Messages, OpenAI Responses, and Gemini
are natively item-shaped (typed content blocks per message); OpenAI Chat
Completions is the outlier that hoists tool calls into a side-array, and that
asymmetry is normalized at the dialect-adapter boundary — never inside the
model. [`dialect::deflate`](src/dialect/deflate.rs) is the one-way flattening
from items to the chat request (`ItemRequest → ChatRequest`); the response
direction renders from the chat-shaped canonical back to each dialect.

## Dataflow

Request path (any client dialect → engine):

```
wire JSON ──► dialect/<d>/req.rs ──► ItemRequest ──► deflate::items_to_chat_request ──► ChatRequest ──► engine
                (parse + validate)     (items hub)        (flatten, tool split)          (engine shape)
```

Response path, non-streaming:

```
engine ChatResponse ──► dialect/<d>/out.rs ──► wire JSON
                          (anthropic, openai_responses;
                           openai_chat needs none — canonical IS that shape)
```

Response path, streaming:

```
engine ──► Stream<CanonChunk> ──► dialect/<d>/stream.rs ──► SSE segments ──► dialect::sse::sse_response ──► client
             (framer = pure state machine)   (pump = the only runtime-coupled piece, feature "axum")
```

The framers are per-dialect state machines that turn one `CanonChunk` into
zero or more pre-formatted SSE segments. The pump ([`dialect/sse.rs`](src/dialect/sse.rs))
drives a chunk stream through the framer closure plus a finalizer and owns
only stream plumbing, headers, and the guaranteed terminal frame. Each
dialect's `*_stream_response` shell wires its framer to the pump.

## Module map

```
src/
├── items.rs                 the items canonical model (ItemRequest, ContentItem, Tool, …) + validate()
├── canonical.rs              ChatRequest / ChatResponse / CanonChunk the engine speaks
├── error.rs                  ProxyError + OpenAI-style error envelope (axum feature)
└── dialect/
    ├── mod.rs                re-exports; hosts the cross-surface stream test matrix
    ├── deflate.rs            ItemRequest → ChatRequest (tool split, prefill, thinking, extras)
    ├── sse.rs                the shared SSE pump (feature "axum") — the sanctioned driver boundary
    ├── anthropic/             req.rs (wire→items) · out.rs (canonical→wire + error JSON) · stream.rs (framer + StopWindow)
    ├── openai_chat/          req.rs (side-array fold→items) · stream.rs (OpenAiFramer)
    └── openai_responses/     req.rs (typed input items→items) · out.rs (canonical→Responses body) · stream.rs (ResponsesFramer)
```

## The purity boundary

The translation layer is sans-I/O by construction **and by enforcement**.
[`tests/purity_lint.rs`](tests/purity_lint.rs) scans the pure files and fails
`cargo test` on any occurrence of `tokio`, `reqwest`, `axum`, `sqlx`, `uuid`,
`SystemTime`, `Instant`, `Utc::now`, or `rand::` outside sanctioned regions.
The sanctioned regions are exactly the places where ids, timestamps, and the
runtime may appear:

- `ChatResponse::full` and `responses_body_from_chat` — id/`created` minting
- the framer constructors (`OpenAiFramer`, `ResponsesFramer`, item openers)
  — `sequence_number` stamping
- the three `*_stream_response` shells and `sse_response` — the pump

Test modules (`#[cfg(test)]` / `#[cfg(all(test, feature = "axum"))]`) are
excluded, so tests may use `tokio` freely. Adding a new runtime touchpoint
means adding its region to the lint's allowlist — deliberately a
review-visible change.

Why it matters: the crate must embed anywhere an LLM gateway lives — behind
axum, behind another HTTP framework, or in-process with no HTTP at all.
Purity is what makes the translation core testable without servers, ports,
or mocks, and what keeps id minting and timestamps at the caller's edge where
they can be made deterministic or account-bound.

## Cross-dialect invariants

These hold on every path and are guarded by tests; treat them as contracts
when touching translation code.

**Usage arithmetic is cache-inclusive everywhere.** Canonical
`Usage.prompt_tokens` = fresh + cached_read + cache_write (OpenAI's
convention), so `fresh = prompt_tokens − cached_read − cache_write` holds
uniformly and billing/rate-limit math stays simple. Anthropic's wire reports
`input_tokens` cache-**exclusive**; the Anthropic translators add the cache
classes in on parse and subtract them back out on render. The stream matrix
asserts the arithmetic survives to the wire on all three surfaces.

**Thinking has four flavors and none is fabricated.**

| flavor | inbound becomes | outbound |
|---|---|---|
| Anthropic-signed (`thinking` + `signature`) | `Thinking { signature: Some }` | `thinking_blocks` JSON on the chat message; re-emitted by Anthropic |
| unsigned (vLLM/DeepSeek `reasoning_content`) | `Thinking { signature: None }` | `reasoning_content` on chat; `reasoning` items on Responses |
| Responses-encrypted (`encrypted`) | `Thinking { encrypted }` | carried as-is where the dialect supports it |
| Anthropic-redacted (`redacted_data`) | `Thinking { redacted_data }` | only Anthropic re-emits; no Responses channel, skipped |

`signature` is opaque and never synthesized; `None` propagates as `None`.
Deflate keeps signed and unsigned separate — a signed block never degrades
into `reasoning_content` and vice versa.

**Tool-call linkage and ordering.** `ItemRequest::validate` rejects a
`ToolResult` whose `tool_call_id` doesn't resolve to a prior `ToolCall` in
the conversation, so every surface 400s consistently. Deflate splits tool
results into `role:"tool"` messages that carry their `tool_call_id` and are
emitted **before** the coalesced user text of the same message — a bare or
misordered tool message 400s on real upstreams. Non-text tool-result content
is a hard error rather than a silent stringify.

**Prefill is marked, never sent.** A trailing assistant text turn (with no
tool calls) is Anthropic-style prefill — the model must continue it. Deflate
sets `ChatRequest.extra["_prefill"]` ([`PREFILL_MARKER`](src/canonical.rs));
the engine-side sender consumes and strips it. It must never reach an
upstream wire.

**Dialect-unknown fields ride in `extra` and drop on the floor where
meaningless.** `ItemRequest.extra` carries what the typed model can't express
(`mcp_servers`, `service_tier`, …); `ChatRequest.extra` fills gaps without
clobbering keys the translator computed. Responses-only fields
(`background`, `previous_response_id`, `conversation`, `include`,
`truncation`) are dropped at deflate — they 400 on chat upstreams.

**Every stream ends with exactly one terminal frame.** OpenAI chat: a
`data: [DONE]`; Anthropic: `message_delta` + `message_stop`; Responses:
exactly one `response.completed` / `incomplete` / `failed`. The finalizer
emits it even when the upstream truncates, and no content frames follow the
terminal. Stream responses are always HTTP 200 — errors travel in-band as the
dialect's error frame (which carries its own terminator). Client-facing error
messages stay generic: provider bodies can echo request material and are
never forwarded.

**Anthropic block indices are protocol-correct.** Content blocks reopen at
fresh indices per Anthropic's stream protocol; thinking blocks occupy
`0..max_thinking_index` and everything else opens after. `StopWindow` is a
sliding match that withholds the last `max_len − 1` characters of the open
block so a stop sequence straddling a delta boundary is caught, and swallows
everything after it fires.

**Responses sequence numbers are monotonic** and stamped at the sanctioned
framer boundary; the terminal frame is deferred until the usage trailer (or
stream end) so it carries real token counts.

## Error model

One [`ProxyError`](src/error.rs) flows end-to-end through parsing,
translation, and the embedder's transport. `http_status()` maps variants to
the wire status (with `BudgetExceeded → 403`); `log_status()` intentionally
diverges for logging. The Anthropic dialect overrides the error envelope
shape (`anthropic/out.rs::error_json`) — e.g. budget maps to an
Anthropic-style 429 — while `error_response()` renders the OpenAI-style
`{error: {message, type, code}}` envelope for the non-streaming path.

## Testing, benchmarks, CI

- Unit tests live inline (`#[cfg(test)]`) next to the code they cover, with
  serde round-trips and validation cases in [items.rs](src/items.rs) and
  deflate cases in [deflate.rs](src/dialect/deflate.rs).
- [`stream_matrix_tests.rs`](src/dialect/stream_matrix_tests.rs) drives one
  table of canonical chunk scripts (plain text, tool calls, thinking,
  usage trailer, truncation) through **all three** framers, asserting the
  invariants above per dialect. No server, no ports, no mocks — framers take
  in-memory streams.
- [`tests/purity_lint.rs`](tests/purity_lint.rs) enforces the sans-I/O
  boundary (see above).
- [`benches/framer.rs`](benches/framer.rs) (criterion) covers the framer hot
  path — every streamed token flows through these cores — plus an empty-chunk
  early-out canary.
- [CI](.github/workflows/ci.yml) checks stable / nightly / MSRV 1.88 across
  default and `--all-features`, builds benches, runs fmt, clippy
  (`-D warnings`), tests, and doc tests.

Publishing is automated via trusted publishing on version tags — see
[.github/PUBLISHING.md](.github/PUBLISHING.md).
