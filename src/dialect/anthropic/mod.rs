//! Inbound Anthropic Messages adapter: translates Anthropic wire format
//! (`POST /v1/messages`, `/v1/messages/count_tokens`) to/from the canonical
//! items model. This is the only Anthropic ingress path; the legacy
//! `anthropic_in.rs` module was deleted (its conformance tests live in
//! `stream.rs` / `out.rs`).

pub mod out;
pub mod req;
pub mod stream;
