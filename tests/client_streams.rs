//! Client feature end-to-end: axum server on one side (the dialect
//! framers), `DialectClient` on the other (the deframers). No external HTTP —
//! the crate's server side tests its own client side.

use futures::StreamExt;
use llm_dialect::canonical::CanonChunk;
use llm_dialect::client::{ClientReply, Dialect, DialectClient};
use llm_dialect::error::ProxyError;
use llm_dialect::items::{ContentItem, ItemMeta, ItemRequest, ItemStreamMessage, Role};

fn text(s: &str) -> CanonChunk {
    CanonChunk {
        delta_text: s.into(),
        ..Default::default()
    }
}

fn req_text(t: &str) -> ItemRequest {
    ItemRequest {
        model: "m".into(),
        max_tokens: Some(100),
        messages: vec![ItemStreamMessage {
            role: Role::User,
            items: vec![ContentItem::Text { text: t.into() }],
            metadata: ItemMeta::default(),
        }],
        ..Default::default()
    }
}

/// Bind to an OS-assigned port and serve the given router. Returns the
/// `http://127.0.0.1:PORT` base address; the server lives on the spawned
/// task and dies with the test process.
async fn spawn(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

mod anthropic {
    use super::*;
    use llm_dialect::dialect::anthropic::out::response_from_canonical;
    use llm_dialect::dialect::anthropic::stream::anthropic_stream_response;

    #[tokio::test]
    async fn send_non_streaming_round_trip() {
        let app = axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(|| async {
                axum::Json(response_from_canonical(
                    &serde_json::json!({
                        "id":"msg_x","choices":[{"message":{"role":"assistant","content":"pong"},
                            "finish_reason":"stop"}],
                        "usage":{"prompt_tokens":7,"completion_tokens":2}
                    }),
                    "m",
                ))
            }),
        );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::Anthropic);
        let reply = client.send(&req_text("ping")).await.unwrap();
        match reply {
            ClientReply::Anthropic(r) => {
                assert!(
                    matches!(&r.message.items[0], ContentItem::Text { text } if text == "pong")
                );
                assert_eq!(r.usage.prompt_tokens, 7);
                assert_eq!(r.finish_reason.as_deref(), Some("stop"));
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn streaming_round_trip_through_deframer() {
        let chunks = vec![
            text("hello "),
            text("world"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(llm_dialect::canonical::Usage {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..Default::default()
            },
        ];
        let app =
            axum::Router::new().route(
                "/v1/messages",
                axum::routing::post(move || {
                    let s = futures::stream::iter(
                        chunks.clone().into_iter().map(Ok::<CanonChunk, ProxyError>),
                    );
                    async move {
                        anthropic_stream_response(Box::pin(s), "m".into(), "msg_1".into(), vec![])
                    }
                }),
            );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::Anthropic);
        let mut req = req_text("ping");
        req.stream = true;
        let mut stream = client.stream(&req);
        let mut got = String::new();
        let mut saw_end = false;
        while let Some(item) = stream.next().await {
            let c = item.unwrap();
            got.push_str(&c.delta_text);
            if c.finish_reason.is_some() {
                saw_end = true;
                assert_eq!(c.usage.as_ref().unwrap().prompt_tokens, 5);
            }
        }
        assert_eq!(got, "hello world");
        assert!(saw_end);
    }

    #[tokio::test]
    async fn error_envelope_maps_to_typed_variant() {
        let app = axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(|| async {
                (
                    axum::http::StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({
                        "type":"error",
                        "error":{"type":"authentication_error","message":"bad key"}
                    })),
                )
            }),
        );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::Anthropic);
        let err = client.send(&req_text("x")).await.unwrap_err();
        assert!(matches!(err, ProxyError::Unauthorized), "{err:?}");
    }
}

mod openai_chat {
    use super::*;
    use llm_dialect::dialect::openai_chat::stream::openai_stream_response;

    #[tokio::test]
    async fn send_non_streaming_parses_chat_body() {
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(|| async {
                axum::Json(serde_json::json!({
                    "id":"c1","object":"chat.completion","created":0,"model":"m",
                    "choices":[{"index":0,"finish_reason":"stop",
                        "message":{"role":"assistant","content":"pong"}}],
                    "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,
                             "prompt_tokens_details":{"cached_tokens":2},
                             "completion_tokens_details":{"reasoning_tokens":0}}
                }))
            }),
        );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::OpenAiChat);
        let reply = client.send(&req_text("ping")).await.unwrap();
        match reply {
            ClientReply::OpenAiChat(chat) => {
                assert_eq!(chat.choices[0].message.text(), "pong");
                assert_eq!(chat.usage.cached_read_tokens, 2);
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn streaming_reassembles_text_and_usage() {
        let chunks = vec![
            text("a"),
            text("b"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(llm_dialect::canonical::Usage {
                    prompt_tokens: 7,
                    completion_tokens: 2,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..Default::default()
            },
        ];
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || {
                let s = futures::stream::iter(
                    chunks.clone().into_iter().map(Ok::<CanonChunk, ProxyError>),
                );
                async move { openai_stream_response(Box::pin(s), "id".into(), "m".into(), 0, true) }
            }),
        );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::OpenAiChat);
        let mut req = req_text("x");
        req.stream = true;
        let mut stream = client.stream(&req);
        let mut text = String::new();
        let mut usage_seen = false;
        while let Some(item) = stream.next().await {
            let c = item.unwrap();
            text.push_str(&c.delta_text);
            if let Some(u) = &c.usage {
                assert_eq!(u.prompt_tokens, 7);
                usage_seen = true;
            }
        }
        assert_eq!(text, "ab");
        assert!(usage_seen);
    }
}

mod openai_responses {
    use super::*;
    use llm_dialect::dialect::openai_responses::out::responses_body_from_chat;
    use llm_dialect::dialect::openai_responses::stream::responses_stream_response;

    #[tokio::test]
    async fn send_non_streaming_parses_responses_body() {
        let mut chat = llm_dialect::canonical::ChatResponse::new(
            "m",
            "pong".into(),
            Some("stop".into()),
            llm_dialect::canonical::Usage {
                prompt_tokens: 4,
                completion_tokens: 2,
                cached_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            },
        );
        chat.id = "resp_x".into();
        chat.choices[0].message.reasoning_content = Some("because".into());
        let wire = responses_body_from_chat(&chat, "m");
        let app = axum::Router::new().route(
            "/v1/responses",
            axum::routing::post(move || {
                let b = wire.clone();
                async move { axum::Json(b) }
            }),
        );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::OpenAiResponses);
        let reply = client.send(&req_text("ping")).await.unwrap();
        match reply {
            ClientReply::OpenAiResponses(r) => {
                assert!(
                    matches!(&r.message.items[0], ContentItem::Thinking { text, .. } if text == "because")
                );
                assert!(
                    matches!(&r.message.items[1], ContentItem::Text { text } if text == "pong")
                );
                assert_eq!(r.usage.prompt_tokens, 4);
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn streaming_round_trip_through_deframer() {
        let chunks = vec![
            text("hi"),
            CanonChunk {
                finish_reason: Some("stop".into()),
                usage: Some(llm_dialect::canonical::Usage {
                    prompt_tokens: 9,
                    completion_tokens: 4,
                    cached_read_tokens: 0,
                    cache_write_tokens: 0,
                    reasoning_tokens: None,
                }),
                ..Default::default()
            },
        ];
        let app = axum::Router::new().route(
            "/v1/responses",
            axum::routing::post(move || {
                let s = futures::stream::iter(
                    chunks.clone().into_iter().map(Ok::<CanonChunk, ProxyError>),
                );
                async move { responses_stream_response(Box::pin(s), "resp_1".into(), "m".into()) }
            }),
        );
        let base = spawn(app).await;
        let client = DialectClient::new(base, Dialect::OpenAiResponses);
        let mut req = req_text("ping");
        req.stream = true;
        let mut stream = client.stream(&req);
        let mut text = String::new();
        let mut saw_term = false;
        while let Some(item) = stream.next().await {
            let c = item.unwrap();
            text.push_str(&c.delta_text);
            if let Some(u) = &c.usage {
                assert_eq!(u.prompt_tokens, 9);
                assert_eq!(c.finish_reason.as_deref(), Some("stop"));
                saw_term = true;
            }
        }
        assert_eq!(text, "hi");
        assert!(saw_term);
    }
}
