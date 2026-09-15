//! Error type shared by every translation entry point.

/// Errors from request parsing, translation, and (in an embedder's runtime)
/// upstream transport. `http_status`/`log_status` map each variant to the wire
/// and log mappings an ingress surface would use; the variants are shared here
/// so a mapped request carries one error type end to end.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("no healthy deployment for route {0}")]
    NoHealthyDeployment(String),
    #[error("upstream {status}: {body}")]
    Upstream {
        status: u16,
        body: String,
        /// Upstream-supplied backoff hint (Retry-After seconds), when present.
        /// None on errors constructed without a response (timeouts, bad JSON).
        retry_after_secs: Option<u64>,
    },
    #[error("rate limit exceeded")]
    RateLimited,
    #[error("budget exceeded")]
    BudgetExceeded,
    #[error("model not allowed for this key")]
    ModelNotAllowed,
    #[error("transport error: {0}")]
    Transport(String),
    #[error("invalid api key")]
    Unauthorized,
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ProxyError {
    pub fn upstream(status: u16, body: String) -> Self {
        ProxyError::Upstream {
            status,
            body,
            retry_after_secs: None,
        }
    }

    /// Status recorded in a spend/request log for this failure. Unlike
    /// http_status(), a client-side disconnect mid-stream is logged as 503
    /// (the client's problem), and unknown-internal as 500 — the log table
    /// intentionally diverges from the wire mapping.
    pub fn log_status(&self) -> i64 {
        match self {
            ProxyError::Upstream { status, .. } => *status as i64,
            ProxyError::RateLimited => 429,
            ProxyError::Unauthorized => 401,
            _ => 500,
        }
    }

    /// HTTP status for this error on the wire. Individual dialects may
    /// override (the Anthropic dialect maps BudgetExceeded to Anthropic-style
    /// 429 — see dialect/anthropic/out.rs).
    pub fn http_status(&self) -> u16 {
        match self {
            ProxyError::Unauthorized => 401,
            ProxyError::RateLimited => 429,
            ProxyError::BudgetExceeded => 403,
            ProxyError::ModelNotAllowed => 403,
            ProxyError::UnknownModel(_) => 404,
            ProxyError::NoHealthyDeployment(_) => 503,
            ProxyError::Transport(_) => 502,
            ProxyError::Upstream { status, .. } => *status,
            ProxyError::BadRequest(_) => 400,
            ProxyError::Internal(_) => 500,
        }
    }
}

/// Render the OpenAI-style error envelope (`{error: {message, type, code}}`)
/// for an out-of-band (pre-stream) failure. In-stream errors stay in-band as
/// dialect error frames; this is for the non-streaming path only.
///
/// Client-facing messages for upstream failures stay generic: provider error
/// bodies can echo request material and are not client-safe.
#[cfg(feature = "axum")]
pub fn error_response(err: &ProxyError) -> axum::response::Response {
    use axum::Json;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let status = StatusCode::from_u16(err.http_status()).unwrap_or(StatusCode::BAD_GATEWAY);
    let code = match err {
        ProxyError::Unauthorized => "invalid_api_key",
        ProxyError::RateLimited => "rate_limit_exceeded",
        ProxyError::BudgetExceeded => "budget_exceeded",
        ProxyError::ModelNotAllowed => "model_not_allowed",
        ProxyError::UnknownModel(_) => "model_not_found",
        ProxyError::NoHealthyDeployment(_) => "no_healthy_deployment",
        ProxyError::Transport(_) => "upstream_transport",
        ProxyError::Upstream { .. } => "upstream_error",
        ProxyError::BadRequest(_) => "bad_request",
        ProxyError::Internal(_) => "internal_error",
    };
    let message = match err {
        ProxyError::Upstream { status, .. } => {
            format!("upstream returned status {status}")
        }
        _ => err.to_string(),
    };
    let ty = match err {
        ProxyError::Upstream { .. } => "upstream_error",
        ProxyError::Transport(_) => "server_error",
        ProxyError::Internal(_) => "server_error",
        ProxyError::BadRequest(_) | ProxyError::UnknownModel(_) => "invalid_request_error",
        _ => "authentication_error",
    };
    (
        status,
        Json(serde_json::json!({
            "error": { "message": message, "type": ty, "code": code, "param": null }
        })),
    )
        .into_response()
}
