//! Cross-dialect canonical representation.
//!
//! The central model is *content items* — a typed, ordered list per message.
//! Anthropic Messages, OpenAI Responses, and Gemini all share this shape
//! natively; OpenAI Chat/Completions is the only dialect that hoists tool
//! calls into a side-array, and that asymmetry is normalized at the dialect
//! adapter boundary.
//!
//! Invariants enforced here:
//! * `ItemStreamMessage.items` preserves order verbatim from the wire.
//! * `ToolCall.id` is non-empty; `ToolResult.tool_call_id` references a prior
//!   `ToolCall` in the same conversation.
//! * `Thinking.signature` is opaque and never fabricated; `None` propagates
//!   as `None` through dialects that support unsigned thinking (all except
//!   Anthropic-signed).
//! * `Text("")` is legal in the model; dropping happens in dialect code
//!   (Gemini rejects empty parts upstream).
//!
//! Lossy notes are documented per-dialect; the canonical model itself stays
//! lossless.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentItem {
    /// Plain assistant/user text. Empty string is legal at the canonical
    /// layer but dialects (Gemini) may drop it.
    Text {
        text: String,
    },
    /// Chain-of-thought block, signed by Anthropic or plaintext elsewhere.
    Thinking {
        text: String,
        /// Opaque Anthropic signature. Never fabricated; preserved verbatim
        /// through Anthropic→Anthropic round trips. `None` when the upstream
        /// was unsigned (vLLM / reasoning_content).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// OpenAI Responses-style encrypted thinking payload.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted: Option<String>,
        /// Anthropic `redacted_thinking` payload (the wire field is `data`).
        /// Opaque; only the Anthropic dialects re-emit it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redacted_data: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        content: serde_json::Value,
        #[serde(default)]
        is_error: bool,
    },
    Refusal {
        text: String,
    },
}

impl ContentItem {
    /// Short diagnostic discriminant for error messages / logs.
    /// Only exercised by tests today; keep the discriminant table here so
    /// future error paths don't need to re-derive it.
    #[cfg(test)]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Thinking { .. } => "thinking",
            Self::ToolCall { .. } => "tool_call",
            Self::ToolResult { .. } => "tool_result",
            Self::Refusal { .. } => "refusal",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ItemMeta {
    /// Per-block cache-control markers (Anthropic `cache_control` ephemeral
    /// breakpoints). Parallel-indexed with the original content items.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_control: Vec<Option<serde_json::Value>>,
    /// OpenAI-only `name` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Anything the dialect layer couldn't express as a typed item.
    /// Dropped by dialects that don't understand them.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemStreamMessage {
    pub role: Role,
    pub items: Vec<ContentItem>,
    #[serde(default)]
    pub metadata: ItemMeta,
}

impl ItemStreamMessage {
    /// Concatenated visible text (Text items only; thinking is metadata). Test aid.
    #[cfg(test)]
    pub fn text(&self) -> String {
        self.items
            .iter()
            .filter_map(|i| match i {
                ContentItem::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    /// Alias for OpenAI `tool_choice: "none"`.
    None,
    /// Alias for OpenAI `tool_choice: "required"` (any tool) / Anthropic
    /// `tool_choice: {type: "any"}`.
    Required,
    Tool {
        name: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ThinkingCfg {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Cross-dialect effort hint: "low" | "medium" | "high" | "max"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default)]
        strict: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemRequest {
    pub model: String,
    pub messages: Vec<ItemStreamMessage>,
    #[serde(default)]
    pub stream: bool,
    /// `None` = client set no cap; each dialect then applies its own wire
    /// default (Anthropic requires a value and substitutes 4096) instead of
    /// silently clamping every request to an arbitrary ceiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingCfg>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Catch-all for dialect-specific fields that don't fit the canonical
    /// model (container / mcp_servers / service_tier etc.). Dropped by every
    /// dialect that doesn't understand them.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for ItemRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: Vec::new(),
            tool_choice: None,
            thinking: None,
            response_format: None,
            extra: serde_json::Map::new(),
        }
    }
}

impl ItemRequest {
    /// Validate the canonical request. Cheap, deterministic, no I/O.
    /// The dialect adapters call this on entry so malformed requests get a
    /// consistent 400 regardless of surface.
    pub fn validate(&self) -> Result<(), String> {
        if self.messages.is_empty() {
            return Err("messages must be non-empty".into());
        }
        if self.max_tokens == Some(0) {
            return Err("max_tokens must be >= 1".into());
        }
        for (i, m) in self.messages.iter().enumerate() {
            if m.items.is_empty() && m.role != Role::System {
                return Err(format!("messages[{i}]: items must be non-empty"));
            }
        }
        // tool_result references must resolve
        let mut known_ids: std::collections::HashSet<&str> = Default::default();
        for m in &self.messages {
            for item in &m.items {
                match item {
                    ContentItem::ToolCall { id, .. } => {
                        known_ids.insert(id.as_str());
                    }
                    ContentItem::ToolResult { tool_call_id, .. }
                        if !known_ids.contains(tool_call_id.as_str()) =>
                    {
                        return Err(format!(
                            "tool_result references unknown tool_call id {tool_call_id:?}"
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(items: Vec<ContentItem>) -> ItemStreamMessage {
        ItemStreamMessage {
            role: Role::Assistant,
            items,
            metadata: ItemMeta::default(),
        }
    }

    #[test]
    fn visible_text_concatenates_in_order() {
        let msg = assistant(vec![
            ContentItem::Text { text: "a".into() },
            ContentItem::Thinking {
                text: "t".into(),
                signature: None,
                encrypted: None,
                redacted_data: None,
            },
            ContentItem::Text { text: "b".into() },
        ]);
        assert_eq!(msg.text(), "ab");
    }

    #[test]
    fn validate_rejects_empty_messages() {
        let req = ItemRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: Some(10),
            ..Default::default()
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn validate_rejects_unknown_tool_call_reference() {
        let req = ItemRequest {
            model: "m".into(),
            max_tokens: Some(10),
            messages: vec![ItemStreamMessage {
                role: Role::User,
                items: vec![ContentItem::ToolResult {
                    tool_call_id: "missing".into(),
                    content: serde_json::json!("x"),
                    is_error: false,
                }],
                metadata: ItemMeta::default(),
            }],
            ..Default::default()
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn validate_accepts_tool_linkage() {
        let req = ItemRequest {
            model: "m".into(),
            max_tokens: Some(10),
            messages: vec![
                assistant(vec![ContentItem::ToolCall {
                    id: "tc_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"command":"ls"}),
                }]),
                ItemStreamMessage {
                    role: Role::User,
                    items: vec![ContentItem::ToolResult {
                        tool_call_id: "tc_1".into(),
                        content: serde_json::json!("file.txt"),
                        is_error: false,
                    }],
                    metadata: ItemMeta::default(),
                },
            ],
            ..Default::default()
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn item_request_round_trips_through_serde() {
        let original = ItemRequest {
            model: "m".into(),
            messages: vec![assistant(vec![
                ContentItem::Thinking {
                    text: "plan".into(),
                    signature: Some("sig".into()),
                    encrypted: None,
                    redacted_data: None,
                },
                ContentItem::Text {
                    text: "hello".into(),
                },
            ])],
            max_tokens: Some(100),
            thinking: Some(ThinkingCfg {
                budget_tokens: Some(2048),
                effort: Some("max".into()),
            }),
            response_format: Some(ResponseFormat::JsonSchema {
                name: "answer".into(),
                schema: serde_json::json!({"type":"object"}),
                strict: true,
            }),
            ..Default::default()
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: ItemRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn item_kind_labels_match_variants() {
        assert_eq!(
            ContentItem::Text {
                text: String::new()
            }
            .kind(),
            "text"
        );
        assert_eq!(
            ContentItem::Thinking {
                text: String::new(),
                signature: None,
                encrypted: None,
                redacted_data: None
            }
            .kind(),
            "thinking"
        );
        assert_eq!(
            ContentItem::ToolCall {
                id: "a".into(),
                name: "b".into(),
                arguments: serde_json::Value::Null
            }
            .kind(),
            "tool_call"
        );
        assert_eq!(
            ContentItem::ToolResult {
                tool_call_id: "a".into(),
                content: serde_json::Value::Null,
                is_error: false
            }
            .kind(),
            "tool_result"
        );
        assert_eq!(
            ContentItem::Refusal {
                text: String::new()
            }
            .kind(),
            "refusal"
        );
    }

    #[test]
    fn tool_choice_serde_tags() {
        let json = serde_json::to_value(ToolChoice::Tool {
            name: "bash".into(),
        })
        .unwrap();
        assert_eq!(json, serde_json::json!({"type":"tool","name":"bash"}));
        assert_eq!(
            serde_json::to_value(ToolChoice::Auto).unwrap(),
            serde_json::json!({"type":"auto"})
        );
    }

    #[test]
    fn tool_choice_and_response_format_round_trip() {
        let t = ToolChoice::Required;
        assert_eq!(
            serde_json::from_value::<ToolChoice>(serde_json::to_value(t.clone()).unwrap()).unwrap(),
            t
        );
        let rf = ResponseFormat::JsonSchema {
            name: "n".into(),
            schema: serde_json::json!({}),
            strict: false,
        };
        assert_eq!(
            serde_json::from_value::<ResponseFormat>(serde_json::to_value(rf.clone()).unwrap())
                .unwrap(),
            rf
        );
    }

    #[test]
    fn signature_is_preserved_verbatim_when_signed() {
        let original = ContentItem::Thinking {
            text: "x".into(),
            signature: Some("EqQBCgIYAhIM...".into()),
            encrypted: None,
            redacted_data: None,
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: ContentItem = serde_json::from_str(&s).unwrap();
        match back {
            ContentItem::Thinking { signature, .. } => {
                assert_eq!(signature, Some("EqQBCgIYAhIM...".into()));
            }
            _ => panic!(),
        }
    }
}
