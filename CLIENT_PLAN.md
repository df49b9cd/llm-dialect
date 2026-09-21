# Client/Server Feature Plan

How `llm-dialect` grows from a server-side translation core into a dual
client/server library, without breaking the sans-I/O core contract.

## Goal

Today the crate implements one direction of every conversion — the
**server/inbound** side of a gateway: dialect wire → canonical (requests),
canonical → dialect wire (responses), `CanonChunk` → SSE frames (streams).

The target state adds the **client/outbound** direction symmetrically: a
caller should be able to build a request in one dialect and send it, and
parse a response from any dialect — with the same pure-translation core and
optional, feature-gated I/O shells.

## Design principles (unchanged)

1. **The core stays pure.** No async runtime, no HTTP types, no wall clock,
   no randomness in translation code. `tests/purity_lint.rs` is amended, not
   weakened, when new sanctioned regions are added.
2. **`futures::Stream` is the only transport currency.** No channel library
   becomes a dependency of the core or of the primary client feature (see
   the crossfire appendix).
3. **Server users see zero change.** Everything new is additive — new
   modules, new feature flags; existing modules untouched except exports.
4. **Round-trip tests are the primary test strategy.** New code is mostly
   inverses of existing code; the existing chunk scripts in
   `stream_matrix_tests.rs` already provide the corpus.

## Feature layout

```
default = []                        pure translation, both directions
axum    = ["dep:axum"]              (existing) server SSE pump + Response shells
client  = ["dep:reqwest", ...]      async client transport (opt-in)
client-sync = ["client", "dep:crossfire"]   blocking iterator adapter (deferred)
```

`client` features never flip on transitively for server consumers; the
purity lint keeps scanning non-feature-gated files.

## Work packages

### WP1 — Pure client translation: requests (render outward)

Add, under each dialect module, a `req_out.rs`:

| module | function | notes |
|---|---|---|
| `anthropic/req_out.rs` | `ItemRequest → /v1/messages` JSON | system hoisting, tool split reversal, thinking config restore; a trailing assistant text turn renders as prefill naturally (never via the `_prefill` marker round-trip) |
| `openai_responses/req_out.rs` | `ItemRequest → /v1/responses` JSON | typed input items; `rs_`/`fc_`/`msg_` id minting at the sanctioned factory boundary |
| OpenAI chat | none needed | `deflate`'s `ChatRequest` already IS the wire shape; add only serde `Serialize` coverage if missing |

Acceptance: `parse(render(req))` round-trips for the existing req.rs test
corpus; `_prefill` marker behavior documented as server-only.

### WP2 — Pure client translation: responses (parse inward)

| module | function | notes |
|---|---|---|
| `anthropic/resp_in.rs` | `/v1/messages` body → typed reply | inverse of `out.rs`; reuses `ItemRequest`'s message items so an assistant turn can accumulate into a conversation directly |
| `openai_responses/resp_in.rs` | Responses body → typed reply | inverse of `out.rs::responses_body_from_chat` |
| OpenAI chat | parse into `ChatResponse` | canonical already is that shape |
| `error.rs` | `ProxyError::from_wire(status, body, dialect)` | parses Anthropic `{"type":"error"...}` and OpenAI `{error:{...}}` envelopes; inverse of `error_json`/`error_response` |

New type decision: introduce `canonical::Reply` (item-shaped assistant turn +
usage + stop reason) **or** reuse `items` message enum — resolve at WP2 kickoff;
reuse is preferred since conversation accumulation is the common client pattern.

### WP3 — Streaming deframers (SSE → canonical)

`dialect/<d>/deframe.rs` × 3: state machines mapping dialect SSE event JSON →
`CanonChunk`, each an inverse of the dialect's `stream.rs` framer:

- Anthropic: `content_block_start/delta/stop`, `message_delta`, `message_stop`
  → chunk deltas + terminal; validates index sequencing and the
  exactly-one-terminal invariant.
- OpenAI chat: `data:` lines → delta chunks; `data: [DONE]` terminal;
  usage trailer frames.
- Responses: `response.output_*` / `response.completed|incomplete|failed` →
  chunks; sequence-number monotonicity checked, not stamped.

Acceptance: extend `stream_matrix_tests.rs` — for every script and dialect,
`deframe(frame(chunks))` reconstructs the chunk sequence (canonical
equality, minus sanctioned id/seq fields) and rejects scripts with missing
or duplicated terminal frames.

### WP4 — Async client transport (feature `client`)

`src/client.rs` (feature-gated `dep:reqwest`, reusing existing
`eventsource-stream` + `futures`):

```rust
pub struct DialectClient { /* base_url, dialect, reqwest::Client, clock/id fns */ }
impl DialectClient {
    pub async fn send(&self, req: &ItemRequest) -> Result<Reply, ProxyError>;
    pub fn stream(&self, req: &ItemRequest)
        -> impl Stream<Item = Result<CanonChunk, ProxyError>>;
}
```

- Renders via WP1, POSTs, non-stream parses via WP2, stream runs bytes →
  `eventsource-stream` → WP3 deframer, all in combinators. **No channels.**
- Server-sent error envelopes route through WP2's `from_wire`.
- Ids/timestamps supplied by caller-provided fns at construction (defaults
  behind the feature; sanctioned lint regions documented).

### WP5 — Sync-channel bridging client (feature `client-sync`, deferred)

Optional blocking-caller story: tokio task feeds deframer output into a
bounded **crossfire** channel (async sender / blocking receiver halves — its
exact niche), exposed as `Iterator<Item = Result<CanonChunk, ProxyError>>`.

Deferred until a concrete sync consumer exists; acceptance criteria at that
point: bounded channel sized to deframer fan-out (backpressure, no unbounded
buffering), and crossfire's async-receiver `Stream` adapter verified so the
half can convert back to `futures::Stream` without glue.

## Sequencing and milestones

1. **M1: Anthropic client surface** — WP1 + WP2 for Anthropic (largest gap,
   most demanded).
2. **M2: Responses client surface** — WP1 + WP2 for OpenAI Responses +
   OpenAI chat parse coverage.
3. **M3: Streaming both directions** — WP3 deframers + round-trip matrix.
4. **M4: Transport** — WP4 `client` feature.
5. **M5: WP5** only on demonstrated need.

Each milestone merges independently and keeps `cargo test` +
`purity_lint` green on default and `--all-features`.

## Doc & lint updates (per milestone)

- `ARCHITECTURE.md`: new direction column in the dataflow diagram; module
  map extended; purity lint region list updated for id-minting factories.
- `README.md`: feature table (`default`, `axum`, `client`) + client example.
- `tests/purity_lint.rs`: new sanctioned regions listed explicitly —
  review-visible by construction.
- Crate docs in `lib.rs`: "client and server directions" headline update.

## Explicitly out of scope

- Channel/concurrency primitives in the core crate.
- Retries, rate-limit handling, auth header management beyond passing
  caller-provided headers (client transport stays a thin shell).
- Gemini dialect.
