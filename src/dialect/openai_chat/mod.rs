//! Inbound/outbound OpenAI Chat-Completions adapter (`POST /v1/chat/completions`).
//!
//! OpenAI chat.completions is the outlier dialect: tool calls live on a
//! side-array (`choices[0].message.tool_calls`), not as content items. This
//! module folds the side-array back into `ContentItem::ToolCall` on the way
//! in, and un-folds it on the way out.

pub mod req;
pub mod resp_in;
pub mod stream;
