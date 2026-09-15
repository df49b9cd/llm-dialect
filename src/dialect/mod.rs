//! Wire-format dialect adapters.
//!
//! `*/req.rs` parses the client wire format into the [`crate::items`]
//! canonical model; `*/out.rs` renders canonical responses back to that
//! dialect and `*/stream.rs` holds the per-chunk SSE framer state machines.
//! All of it is pure data transformation; the SSE pump in [`sse`] (feature
//! `axum`) is the only runtime-coupled piece.

pub mod anthropic;
pub mod deflate;
pub mod openai_chat;
pub mod openai_responses;
pub mod sse;

#[cfg(all(test, feature = "axum"))]
#[path = "stream_matrix_tests.rs"]
mod stream_matrix_tests;
