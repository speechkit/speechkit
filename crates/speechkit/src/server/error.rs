//! OpenAI-shaped error responses.

use crate::SpeechError;
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

/// `{"error": {"message", "type", "code"}}`, as OpenAI's API answers.
pub(crate) fn respond(status: StatusCode, message: impl std::fmt::Display) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let body = json!({
        "error": {
            "message": message.to_string(),
            "type": kind,
            "code": status.as_u16(),
        }
    });
    (status, Json(body)).into_response()
}

/// The 503 for a request that found every session slot busy.
pub(crate) fn slot_busy() -> Response {
    respond(
        StatusCode::SERVICE_UNAVAILABLE,
        "every session slot is busy; retry later",
    )
}

/// The status for a failed transcription or synthesis.
pub(crate) fn status_of(error: &SpeechError) -> StatusCode {
    match error {
        SpeechError::InvalidInput(_) | SpeechError::Unsupported(_) => StatusCode::BAD_REQUEST,
        SpeechError::Capacity => StatusCode::SERVICE_UNAVAILABLE,
        SpeechError::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        SpeechError::Backend {
            retryable: true, ..
        } => StatusCode::BAD_GATEWAY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// The error, with its source when there is one.
pub(crate) fn describe(error: &SpeechError) -> String {
    match std::error::Error::source(error) {
        Some(source) => format!("{error}: {source}"),
        None => error.to_string(),
    }
}

pub(crate) async fn require_auth(
    axum::extract::State(state): axum::extract::State<crate::server::state::Shared>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(token) = &state.server.auth else {
        return next.run(request).await;
    };
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    match presented {
        Some(presented) if constant_time_eq(presented.as_bytes(), token.expose().as_bytes()) => {
            next.run(request).await
        }
        _ => respond(StatusCode::UNAUTHORIZED, "missing or invalid bearer token"),
    }
}

/// Compares without an early exit, so timing does not reveal the prefix.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut difference = a.len() ^ b.len();
    for (i, x) in a.iter().enumerate() {
        difference |= usize::from(x ^ b.get(i).copied().unwrap_or(0));
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses() {
        let cases = [
            (SpeechError::InvalidInput("x".into()), 400),
            (SpeechError::Unsupported("x".into()), 400),
            (SpeechError::Capacity, 503),
            (SpeechError::DeadlineExceeded, 504),
            (SpeechError::backend("b", true, "x"), 502),
            (SpeechError::backend("b", false, "x"), 500),
            (SpeechError::Cancelled, 500),
        ];
        for (error, status) in cases {
            assert_eq!(status_of(&error).as_u16(), status, "{error}");
        }
        assert_eq!(
            describe(&SpeechError::backend("b", true, "why")),
            "backend `b` failed: why"
        );
    }

    #[test]
    fn token_comparison() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"", b"a"));
    }
}
