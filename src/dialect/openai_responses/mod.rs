//! Inbound/outbound OpenAI Responses API adapter (`POST /v1/responses`).
//!
//! Responses is item-shaped natively: `input[]` is a list of typed items
//! (message | function_call | function_call_output | reasoning) rather than
//! a chat `messages` array. Most items map one-to-one onto the canonical
//! `ContentItem` shape; only message-role wrapping differs.

pub mod out;
pub mod req;
pub mod req_out;
pub mod resp_in;
pub mod stream;
