# llm-dialect

[![ci](https://github.com/df49b9cd/llm-dialect/actions/workflows/ci.yml/badge.svg)](https://github.com/df49b9cd/llm-dialect/actions/workflows/ci.yml)
[![publish](https://github.com/df49b9cd/llm-dialect/actions/workflows/publish.yml/badge.svg)](https://github.com/df49b9cd/llm-dialect/actions/workflows/publish.yml)
[![crates.io](https://img.shields.io/crates/v/llm-dialect.svg)](https://crates.io/crates/llm-dialect)
[![docs.rs](https://docs.rs/llm-dialect/badge.svg)](https://docs.rs/llm-dialect)
[![rust version](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![license](https://img.shields.io/crates/l/llm-dialect.svg)](./LICENSE-MIT)
[![dependency status](https://deps.rs/crate/llm-dialect/0.1.1/status.svg)](https://deps.rs/crate/llm-dialect/0.1.1)
[![crates.io downloads](https://img.shields.io/crates/d/llm-dialect.svg)](https://crates.io/crates/llm-dialect)
[![GitHub stars](https://img.shields.io/github/stars/df49b9cd/llm-dialect.svg)](https://github.com/df49b9cd/llm-dialect/stargazers)

```toml
llm-dialect = "0.1.1" # MSRV 1.88
```

Pure sans-I/O LLM dialect translation: Anthropic Messages, OpenAI Chat
Completions, and OpenAI Responses wire formats ↔ a canonical request/response
model, plus SSE framer state machines for streamed turns. No async runtime, no
HTTP client/server types in the API surface, no wall clock, no randomness —
id minting and timestamps are supplied by the caller.

The mapping is the translation core of an LLM gateway (a LiteLLM-proxy-class
service): parse whatever dialect your client speaks into one `ItemRequest`,
translate to the `ChatRequest` your engine consumes, render the engine's
canonical chunks back to the caller's dialect.

## What it gives an embedder

Put an Anthropic `/v1/messages` (or OpenAI Responses) surface in front of any
chat/completions-class backend without rewriting the translation yourself:

```rust
use llm_dialect::{
    canonical::ChatRequest,
    dialect::{anthropic::{req, stream::{chunk_to_sse_events, finalize_stream, StreamState}}, deflate},
    items::ItemRequest,
};

// 1. parse an Anthropic Messages wire body into the canonical model
let items: ItemRequest = req::from_anthropic(&body)?;

// 2. flatten to the internal chat request your engine speaks
let req: ChatRequest = deflate::items_to_chat_request(&items)?;

// 3. per-chunk: canonical chunks back to Anthropic SSE frames
let mut st = StreamState::with_stop_sequences(stops);
for chunk in engine_chunks {
    for (event, data) in chunk_to_sse_events(&chunk, &model, &mut st, &msg_id) {
        write!(out, "event: {event}\ndata: {data}\n\n")?;
    }
}
// one guaranteed terminal pair (message_delta + message_stop)
for (event, data) in finalize_stream(&mut st) {
    write!(out, "event: {event}\ndata: {data}\n\n")?;
}
```

Enable the `axum` feature for the shared SSE pump and per-dialect
`*_stream_response` shells (`anthropic_stream_response(&stream, …) -> Response`),
if you're already an axum service and want drop-in handlers.

## Crate shape

| module | contents |
|---|---|
| `items` | the canonical request model (`ItemRequest`, `Item`, `ContentItem`, …) |
| `canonical` | the flattened `ChatRequest`/`ChatResponse`/`CanonChunk` the engine speaks |
| `dialect/{anthropic,openai_chat,openai_responses}` | wire parsers (`*/req.rs`), response renderers (`*/out.rs`), SSE framers (`*/stream.rs`) |
| `dialect::deflate` | `ItemRequest` → `ChatRequest` |
| `dialect::sse` *(feature `axum`)* | the pump that drives a framer over a stream |
| `error` | the shared `ProxyError` and its HTTP error envelope |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module dataflow, the purity
boundary and how it's enforced, and the cross-dialect invariants (usage
arithmetic, thinking, tool-call linkage, streaming terminal frames).

## Guarantees the crate holds

The dialect translation files are pure data transformation. A lint in the
integration tests enforces no `tokio`/`reqwest`/`axum`/`sqlx`/`rand`/
clock calls outside the sanctioned shell/factory regions, so the boundary
above is enforced rather than aspirational.

## License

MIT.
