//! HTTP client transport (feature `client`): reqwest + the pure deframers.
//!
//! [`DialectClient`] POSTs a canonical [`ItemRequest`] rendered to the
//! configured wire dialect, and returns either the parsed non-streaming
//! reply or a `Stream<Item = Result<CanonChunk, ProxyError>>` driven through
//! `eventsource-stream` → that dialect's deframer state machine.
//!
//! All translation work (render request, parse response, deframe chunks) is
//! pure and lives in [`crate::dialect`]; this module only owns the socket,
//! headers, and stream plumbing. Auth and tracing headers are the caller's —
//! this is a transport shell, not an SDK.

use futures::{Stream, StreamExt, TryStreamExt, stream};

use crate::canonical::{CanonChunk, ChatResponse};
use crate::error::{ErrorDialect, ProxyError, error_from_wire};
use crate::items::ItemRequest;

/// Wire dialect for the endpoint this client speaks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Anthropic,
    OpenAiChat,
    OpenAiResponses,
}

impl Dialect {
    fn path(self) -> &'static str {
        match self {
            Self::Anthropic => "/v1/messages",
            Self::OpenAiChat => "/v1/chat/completions",
            Self::OpenAiResponses => "/v1/responses",
        }
    }
    fn error_dialect(self) -> ErrorDialect {
        match self {
            Self::Anthropic => ErrorDialect::Anthropic,
            _ => ErrorDialect::OpenAi,
        }
    }
    /// Anthropic versions the wire schema; OpenAI dialects don't. Callers
    /// pin a specific version via `anthropic-version` in `with_headers`.
    fn default_dialect_header(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Anthropic => Some(("anthropic-version", "2023-06-01")),
            _ => None,
        }
    }
}

/// Canonical non-streaming reply across dialects. `Anthropic`/`Responses`
/// carry the item-shaped assistant turn so it appends to a conversation
/// directly; `OpenAiChat` is the raw OpenAI body (canonical IS chat).
#[derive(Debug, Clone)]
pub enum ClientReply {
    Anthropic(crate::dialect::anthropic::resp_in::AnthropicReply),
    OpenAiChat(ChatResponse),
    OpenAiResponses(crate::dialect::openai_responses::resp_in::ResponsesReply),
}

/// HTTP client for a single dialect endpoint. Cheap to clone (shares the
/// underlying connection pool). Ids and timestamps are the caller's — this
/// transport mints nothing.
#[derive(Clone)]
pub struct DialectClient {
    http: reqwest::Client,
    base_url: String,
    dialect: Dialect,
    /// Anthropic authenticates with x-api-key; the OpenAI dialects use
    /// Authorization Bearer. Bridging callers set both and pick per request.
    anthropic_api_key: Option<String>,
    openai_bearer: Option<String>,
    extra_headers: reqwest::header::HeaderMap,
}

impl DialectClient {
    pub fn new(base_url: impl Into<String>, dialect: Dialect) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            dialect,
            anthropic_api_key: None,
            openai_bearer: None,
            extra_headers: reqwest::header::HeaderMap::new(),
        }
    }

    pub fn with_http(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }
    pub fn with_anthropic_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }
    pub fn with_openai_bearer(mut self, token: impl Into<String>) -> Self {
        self.openai_bearer = Some(token.into());
        self
    }
    /// Extra headers applied after the dialect defaults (tracing ids,
    /// dialect extension headers).
    pub fn with_headers(mut self, headers: reqwest::header::HeaderMap) -> Self {
        self.extra_headers.extend(headers);
        self
    }

    /// Render the canonical request for this dialect.
    fn render(&self, req: &ItemRequest) -> Result<serde_json::Value, ProxyError> {
        match self.dialect {
            Dialect::Anthropic => crate::dialect::anthropic::req_out::to_anthropic(req),
            Dialect::OpenAiChat => {
                let chat = crate::dialect::deflate::items_to_chat_request(req)?;
                serde_json::to_value(&chat).map_err(|e| ProxyError::Internal(anyhow::anyhow!(e)))
            }
            Dialect::OpenAiResponses => {
                crate::dialect::openai_responses::req_out::to_openai_responses(req)
            }
        }
    }

    fn apply_auth(&self, r: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut r = r;
        if let Some((k, v)) = self.dialect.default_dialect_header() {
            r = r.header(k, v);
        }
        match self.dialect {
            Dialect::Anthropic => {
                if let Some(key) = &self.anthropic_api_key {
                    r = r.header("x-api-key", key);
                }
            }
            _ => {
                if let Some(t) = &self.openai_bearer {
                    r = r.bearer_auth(t);
                }
            }
        }
        r.headers(self.extra_headers.clone())
    }

    /// Non-streaming request → parsed typed reply. Errors route through
    /// [`error_from_wire`] so callers see typed variants (`Unauthorized`,
    /// `RateLimited`, …) rather than raw bodies. Streaming requests must use
    /// [`Self::stream`] instead.
    pub async fn send(&self, req: &ItemRequest) -> Result<ClientReply, ProxyError> {
        if req.stream {
            return Err(ProxyError::BadRequest(
                "stream:true must use DialectClient::stream".into(),
            ));
        }
        let body = self.render(req)?;
        let resp = self
            .apply_auth(
                self.http
                    .post(format!("{}{}", self.base_url, self.dialect.path())),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| ProxyError::Transport(format!("http send: {e}")))?;
        let status = resp.status().as_u16();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProxyError::Transport(format!("http body: {e}")))?;
        if !(200..300).contains(&status) {
            return Err(error_from_wire(
                status,
                &String::from_utf8_lossy(&bytes),
                self.dialect.error_dialect(),
            ));
        }
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| ProxyError::Transport(format!("malformed response JSON: {e}")))?;
        match self.dialect {
            Dialect::Anthropic => {
                crate::dialect::anthropic::resp_in::parse_response(&v).map(ClientReply::Anthropic)
            }
            Dialect::OpenAiChat => crate::dialect::openai_chat::resp_in::parse_response(&v)
                .map(ClientReply::OpenAiChat),
            Dialect::OpenAiResponses => {
                crate::dialect::openai_responses::resp_in::parse_response(&v)
                    .map(ClientReply::OpenAiResponses)
            }
        }
    }

    /// Streaming request → canonical chunk stream.
    ///
    /// The response body pipes through `eventsource-stream` into the
    /// dialect's deframer; the stream yields chunks as they arrive and ends
    /// after the terminal frame. Render failures, non-2xx statuses, and a
    /// truncated body (EOF without the terminal frame) surface as `Err`.
    /// The returned stream borrows `self` and `req`; callers wanting an
    /// owned stream clone both into a fresh client first.
    pub fn stream(
        &self,
        req: &ItemRequest,
    ) -> futures::stream::BoxStream<'_, Result<CanonChunk, ProxyError>> {
        let mut req = req.clone();
        req.stream = true;
        let init = self.render(&req).map(|body| {
            self.apply_auth(
                self.http
                    .post(format!("{}{}", self.base_url, self.dialect.path())),
            )
            .json(&body)
            .header(reqwest::header::ACCEPT, "text/event-stream")
        });
        let dialect = self.dialect;
        let err_dialect = dialect.error_dialect();

        stream::once(async move { init })
            .and_then(move |rb| async move {
                let resp = rb
                    .send()
                    .await
                    .map_err(|e| ProxyError::Transport(format!("http send: {e}")))?;
                let status = resp.status().as_u16();
                if !(200..300).contains(&status) {
                    let body = resp
                        .text()
                        .await
                        .map_err(|e| ProxyError::Transport(format!("http body: {e}")))?;
                    return Err(error_from_wire(status, &body, err_dialect));
                }
                Ok(resp)
            })
            // Flatten into the chunk stream on success or a one-shot Err.
            .map(move |r| {
                let resp_stream_out: stream::BoxStream<'static, Result<CanonChunk, ProxyError>> =
                    match r {
                        Err(e) => Box::pin(stream::once(async move { Err(e) })),
                        Ok(resp) => Box::pin(resp_stream(resp, dialect)),
                    };
                resp_stream_out
            })
            .flatten()
            .boxed()
    }
}

type BoxByteStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>;

/// Pipe one SSE response through the dialect's deframer, yielding canonical
/// chunks until the terminal frame or stream end.
fn resp_stream(
    resp: reqwest::Response,
    dialect: Dialect,
) -> impl Stream<Item = Result<CanonChunk, ProxyError>> + Send + 'static {
    let byte_stream: BoxByteStream = Box::pin(resp.bytes_stream());
    let source = eventsource_stream::Eventsource::eventsource(byte_stream);

    stream::unfold(
        Some((source, DeframerState::new(dialect))),
        |acc| async move {
            let (mut source, mut state) = acc?;
            loop {
                let ev = match source.next().await {
                    Some(Ok(ev)) => ev,
                    Some(Err(e)) => {
                        return Some((Err(ProxyError::Transport(format!("sse: {e}"))), None));
                    }
                    None => {
                        // SSE body ended without the terminal frame: finalize
                        // decides truncation vs clean close.
                        return match state.finalize() {
                            Ok(()) => None, // already saw terminal; body close is clean
                            Err(e) => Some((Err(e), None)),
                        };
                    }
                };
                match state.push(&ev) {
                    Ok(Some(chunk)) => return Some((Ok(chunk), Some((source, state)))),
                    Ok(None) if state.terminated() => return None,
                    Ok(None) => continue, // bookkeeping frame — keep polling
                    Err(e) => return Some((Err(e), None)),
                }
            }
        },
    )
}

/// Deframer dispatch, kept concrete per dialect — plain `match` at the
/// boundary rather than boxing a trait object.
enum DeframerState {
    Anthropic(crate::dialect::anthropic::deframe::AnthropicDeframer),
    OpenAiChat(crate::dialect::openai_chat::deframe::OpenAiDeframer),
    OpenAiResponses(crate::dialect::openai_responses::deframe::ResponsesDeframer),
}

impl DeframerState {
    fn new(dialect: Dialect) -> Self {
        use Dialect::*;
        match dialect {
            Anthropic => {
                Self::Anthropic(crate::dialect::anthropic::deframe::AnthropicDeframer::new())
            }
            OpenAiChat => {
                Self::OpenAiChat(crate::dialect::openai_chat::deframe::OpenAiDeframer::new())
            }
            OpenAiResponses => Self::OpenAiResponses(
                crate::dialect::openai_responses::deframe::ResponsesDeframer::new(),
            ),
        }
    }

    /// Push one SSE event; `Ok(Some(_))` surfaces a chunk, `Ok(None)` is a
    /// bookkeeping frame — after which `terminated()` reports the stream end.
    fn push(&mut self, ev: &eventsource_stream::Event) -> Result<Option<CanonChunk>, ProxyError> {
        match self {
            Self::Anthropic(d) => d.push(ev.event.as_str(), &ev.data).map(|o| o.chunk),
            Self::OpenAiChat(d) => d.push_data(&ev.data).map(|o| o.chunk),
            Self::OpenAiResponses(d) => d.push(ev.event.as_str(), &ev.data).map(|o| o.chunk),
        }
    }

    fn terminated(&self) -> bool {
        match self {
            Self::Anthropic(d) => d.finalize().is_ok(),
            Self::OpenAiChat(d) => d.finalize().is_ok(),
            Self::OpenAiResponses(d) => d.finalize().is_ok(),
        }
    }

    /// End-of-body invoice. On success the stream just ends (no chunk
    /// emitted); on truncation the error is the last item.
    fn finalize(self) -> Result<(), ProxyError> {
        match self {
            Self::Anthropic(d) => d.finalize(),
            Self::OpenAiChat(d) => d.finalize(),
            Self::OpenAiResponses(d) => d.finalize(),
        }
    }
}
