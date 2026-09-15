//! llm-dialect: pure sans-I/O translation between LLM wire formats.
//!
//! Parse an Anthropic (`/v1/messages`), OpenAI chat (`/v1/chat/completions`),
//! or OpenAI Responses (`/v1/responses`) wire request into the canonical
//! [`items`] model, flatten to the [`canonical`] `ChatRequest` an engine
//! consumes ([`dialect::deflate`]), and render the engine's responses/
//! streams back to the caller's dialect ([`dialect::*`]).
//!
//! The mapping is pure: no async runtime, no HTTP types in the public
//! signatures, no wall clock, no randomness — ids and timestamps are supplied
//! by the caller at the sanctioned factory boundaries. The optional `axum`
//! feature adds the SSE pump ([`dialect::sse`]) and per-dialect stream
//! `Response` shells for embedders that *are* an axum service.

pub mod canonical;
pub mod dialect;
pub mod error;
pub mod items;
