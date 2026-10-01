//! The WebSocket transport shared by Realtime and DashScope: TLS, the
//! handshake, bounded messages, and error mapping. Protocol state lives in
//! each backend.

use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use crate::{Secret, SpeechError, speech::deadline::instant_after};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream,
    tungstenite::{
        Error as WsError, Message,
        client::IntoClientRequest,
        error::ProtocolError,
        http::Uri,
        protocol::{CloseFrame, WebSocketConfig, frame::coding::CloseCode},
    },
};

use crate::cloud::CloudRuntime;

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// The largest message accepted: 1 MiB.
const MAX_MESSAGE: usize = 1024 * 1024;

fn tls() -> Result<Arc<rustls::ClientConfig>, SpeechError> {
    static CONFIG: OnceLock<Result<Arc<rustls::ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let roots =
                rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            rustls::ClientConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .map(|builder| Arc::new(builder.with_root_certificates(roots).with_no_client_auth()))
            .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(|e| SpeechError::backend("websocket", false, e))
}

/// Checks the settings every WebSocket backend shares: a `ws://` or
/// `wss://` endpoint, a model and key, and a positive timeout.
pub(crate) fn check_session(
    endpoint: &str,
    model: &str,
    api_key: &Secret,
    timeout: Duration,
) -> Result<(), SpeechError> {
    check_endpoint(endpoint)?;
    if model.trim().is_empty() || api_key.expose().trim().is_empty() {
        return Err(SpeechError::InvalidInput(
            "the model and API key must not be empty".into(),
        ));
    }
    if timeout.is_zero() {
        return Err(SpeechError::InvalidInput(
            "the timeout must be positive".into(),
        ));
    }
    Ok(())
}

/// Checks a `ws://` or `wss://` endpoint.
fn check_endpoint(endpoint: &str) -> Result<(), SpeechError> {
    let invalid = |why: &str| {
        Err(SpeechError::InvalidInput(format!(
            "invalid WebSocket endpoint: {why}"
        )))
    };
    if endpoint.contains('#') {
        return invalid("remove the fragment");
    }
    let Ok(uri) = endpoint.parse::<Uri>() else {
        return invalid("not a URL");
    };
    let websocket = uri
        .scheme_str()
        .is_some_and(|scheme| ["ws", "wss"].iter().any(|s| scheme.eq_ignore_ascii_case(s)));
    if !websocket || uri.host().is_none_or(str::is_empty) {
        return invalid("use a ws:// or wss:// URL with a host");
    }
    if uri
        .authority()
        .is_some_and(|authority| authority.as_str().contains('@'))
    {
        return invalid("credentials belong in the API key, not the URL");
    }
    Ok(())
}

/// Maps a WebSocket failure. Network trouble is retryable.
pub(crate) fn failure(backend: &str, error: &WsError) -> SpeechError {
    let (retryable, message) = match error {
        WsError::ConnectionClosed
        | WsError::AlreadyClosed
        | WsError::Protocol(ProtocolError::ResetWithoutClosingHandshake) => {
            (true, "the connection closed".to_owned())
        }
        WsError::Io(e) => (true, format!("I/O failed: {}", e.kind())),
        WsError::Tls(_) => (true, "the TLS handshake failed".to_owned()),
        WsError::Capacity(_) => (false, "a message exceeded the size limit".to_owned()),
        WsError::Http(response) => {
            let status = response.status().as_u16();
            (
                matches!(status, 408 | 429 | 500..=599),
                format!("the handshake returned HTTP {status}"),
            )
        }
        WsError::Url(_) | WsError::HttpFormat(_) => (false, "the request is invalid".to_owned()),
        _ => (false, "the WebSocket protocol failed".to_owned()),
    };
    SpeechError::backend(backend.to_owned(), retryable, message)
}

/// A message from the server.
#[derive(Debug)]
pub(crate) enum Incoming {
    /// A text message holding JSON.
    Json(Value),
    /// A binary message.
    Binary(
        #[cfg_attr(
            not(feature = "dashscope"),
            expect(dead_code, reason = "only DashScope synthesis reads binary audio")
        )]
        Vec<u8>,
    ),
}

/// A connected WebSocket used from a blocking thread. Dropping it without
/// [`close`](Self::close) ends the TCP connection without a close handshake.
pub(crate) struct Ws {
    runtime: CloudRuntime,
    socket: Option<Socket>,
    backend: &'static str,
}

#[cfg_attr(
    not(feature = "dashscope"),
    expect(
        dead_code,
        reason = "only DashScope synthesis reads the socket from its own thread"
    )
)]
impl Ws {
    /// Connects, sending `key` as a bearer token plus `headers`.
    pub(crate) fn connect(
        runtime: CloudRuntime,
        backend: &'static str,
        endpoint: &str,
        key: &Secret,
        headers: &[(&'static str, &'static str)],
        timeout: Duration,
    ) -> Result<Self, SpeechError> {
        let mut request = endpoint
            .into_client_request()
            .map_err(|_| SpeechError::InvalidInput("invalid WebSocket endpoint".into()))?;
        let value = format!("Bearer {}", key.expose()).parse().map_err(|_| {
            SpeechError::InvalidInput("the API key is not a valid header value".into())
        })?;
        request.headers_mut().insert("Authorization", value);
        for (name, value) in headers {
            let value = value
                .parse()
                .map_err(|_| SpeechError::InvalidInput(format!("invalid header {name}")))?;
            request.headers_mut().insert(*name, value);
        }
        let mut config = WebSocketConfig::default();
        config.max_message_size = Some(MAX_MESSAGE);
        config.max_frame_size = Some(MAX_MESSAGE);
        let connector = Connector::Rustls(tls()?);
        let socket = runtime.block_on(async {
            let connecting = tokio_tungstenite::connect_async_tls_with_config(
                request,
                Some(config),
                false,
                Some(connector),
            );
            match tokio::time::timeout(timeout, connecting).await {
                Ok(Ok((socket, _))) => Ok(socket),
                Ok(Err(error)) => Err(failure(backend, &error)),
                Err(_) => Err(SpeechError::backend(
                    backend,
                    true,
                    "the connection timed out",
                )),
            }
        })?;
        Ok(Self {
            runtime,
            socket: Some(socket),
            backend,
        })
    }

    fn socket(&mut self) -> Result<&mut Socket, SpeechError> {
        let backend = self.backend;
        self.socket
            .as_mut()
            .ok_or_else(|| SpeechError::backend(backend, true, "the connection is closed"))
    }

    /// Sends a message.
    pub(crate) fn send(&mut self, message: Message, timeout: Duration) -> Result<(), SpeechError> {
        let backend = self.backend;
        let runtime = self.runtime.clone();
        let socket = self.socket()?;
        runtime.block_on(async {
            match tokio::time::timeout(timeout, socket.send(message)).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(error)) => Err(failure(backend, &error)),
                Err(_) => Err(SpeechError::backend(backend, true, "sending timed out")),
            }
        })
    }

    /// Sends a JSON text message.
    pub(crate) fn send_json(
        &mut self,
        value: &Value,
        timeout: Duration,
    ) -> Result<(), SpeechError> {
        self.send(Message::Text(value.to_string().into()), timeout)
    }

    /// The time left before `deadline`, or a retryable error once it has
    /// passed.
    pub(crate) fn time_left(&self, deadline: Instant) -> Result<Duration, SpeechError> {
        deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| {
                SpeechError::backend(self.backend, true, "the server did not answer in time")
            })
    }

    /// Waits up to `timeout` for a JSON or binary message. `Ok(None)`
    /// means nothing arrived in time, or a control frame arrived.
    pub(crate) fn recv_frame(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Incoming>, SpeechError> {
        let backend = self.backend;
        let runtime = self.runtime.clone();
        let socket = self.socket()?;
        let waited = runtime.block_on(async { tokio::time::timeout(timeout, socket.next()).await });
        let Ok(message) = waited else {
            // Nothing arrived in time.
            return Ok(None);
        };
        match message {
            None => Err(SpeechError::backend(backend, true, "the connection closed")),
            Some(Err(error)) => Err(failure(backend, &error)),
            Some(Ok(Message::Text(text))) => serde_json::from_str(&text)
                .map(|value| Some(Incoming::Json(value)))
                .map_err(|_| SpeechError::backend(backend, false, "a message is not valid JSON")),
            Some(Ok(Message::Binary(bytes))) => Ok(Some(Incoming::Binary(bytes.to_vec()))),
            Some(Ok(Message::Close(_))) => {
                self.socket = None;
                Err(SpeechError::backend(
                    backend,
                    true,
                    "the server closed the connection",
                ))
            }
            // Control frames; tungstenite answers pings itself.
            Some(Ok(_)) => Ok(None),
        }
    }

    /// Closes the connection politely, ignoring failures.
    pub(crate) fn close(&mut self, timeout: Duration) {
        let Some(mut socket) = self.socket.take() else {
            return;
        };
        let _ = self.runtime.block_on(async {
            let frame = CloseFrame {
                code: CloseCode::Normal,
                reason: "finished".into(),
            };
            tokio::time::timeout(timeout, socket.send(Message::Close(Some(frame)))).await
        });
    }
}

/// What a [`Pushed`] connection's reader task does with each message.
pub(crate) trait Handler: Send + 'static {
    /// Handles one message as it arrives. An error ends the connection.
    fn on_message(&mut self, message: Incoming) -> Result<(), SpeechError>;

    /// Whether the connection may end now without it being an error.
    fn may_close(&self) -> bool;

    /// The connection ended with `error`, between calls or in one.
    fn failed(&mut self, error: &SpeechError);
}

/// The handler and how the reader task ended, shared with the stream.
struct Reading<H> {
    handler: H,
    error: Option<SpeechError>,
    ended: bool,
}

struct Reader<H> {
    state: std::sync::Mutex<Reading<H>>,
    changed: std::sync::Condvar,
}

impl<H> Reader<H> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Reading<H>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A connection whose messages a task on the runtime hands to a
/// [`Handler`] as they arrive, so events reach the session while no audio
/// is pushed (A-10). The stream sends from its own thread. Dropping it
/// stops the task and closes the socket before returning, since the
/// session's slot is held until then.
pub(crate) struct Pushed<H> {
    runtime: CloudRuntime,
    backend: &'static str,
    sink: Option<futures::stream::SplitSink<Socket, Message>>,
    reader: Arc<Reader<H>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

/// How long dropping a connection waits to close it politely.
const CLOSE_WAIT: Duration = Duration::from_secs(1);

impl Ws {
    /// Hands every message from now on to `handler`, on a task of the
    /// runtime.
    pub(crate) fn push<H: Handler>(mut self, handler: H) -> Result<Pushed<H>, SpeechError> {
        let backend = self.backend;
        let socket = self
            .socket
            .take()
            .ok_or_else(|| SpeechError::backend(backend, true, "the connection is closed"))?;
        let (sink, mut incoming) = socket.split();
        let reader = Arc::new(Reader {
            state: std::sync::Mutex::new(Reading {
                handler,
                error: None,
                ended: false,
            }),
            changed: std::sync::Condvar::new(),
        });
        let task = self.runtime.handle().spawn({
            let reader = reader.clone();
            async move {
                loop {
                    let message = match incoming.next().await {
                        None => Err(SpeechError::backend(backend, true, "the connection closed")),
                        Some(Err(error)) => Err(failure(backend, &error)),
                        Some(Ok(Message::Text(text))) => serde_json::from_str(&text)
                            .map(Incoming::Json)
                            .map_err(|_| {
                                SpeechError::backend(backend, false, "a message is not valid JSON")
                            }),
                        Some(Ok(Message::Binary(bytes))) => Ok(Incoming::Binary(bytes.to_vec())),
                        Some(Ok(Message::Close(_))) => Err(SpeechError::backend(
                            backend,
                            true,
                            "the server closed the connection",
                        )),
                        // Control frames; tungstenite answers pings itself.
                        Some(Ok(_)) => continue,
                    };
                    let mut state = reader.lock();
                    // The connection is gone after an error from it.
                    let gone = message.is_err();
                    let outcome = match message {
                        Ok(message) => state.handler.on_message(message),
                        Err(_) if state.handler.may_close() => Ok(()),
                        Err(error) => Err(error),
                    };
                    let failed = outcome.is_err();
                    if let Err(error) = outcome {
                        state.handler.failed(&error);
                        state.error = Some(error);
                    }
                    state.ended = gone || failed;
                    drop(state);
                    reader.changed.notify_all();
                    if gone || failed {
                        return;
                    }
                }
            }
        });
        Ok(Pushed {
            runtime: self.runtime.clone(),
            backend,
            sink: Some(sink),
            reader,
            task: Some(task),
        })
    }
}

impl<H: Handler> Pushed<H> {
    /// Sends a message.
    pub(crate) fn send(&mut self, message: Message, timeout: Duration) -> Result<(), SpeechError> {
        let backend = self.backend;
        let Some(sink) = self.sink.as_mut() else {
            return Err(SpeechError::backend(
                backend,
                true,
                "the connection is closed",
            ));
        };
        self.runtime.block_on(async {
            match tokio::time::timeout(timeout, sink.send(message)).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(error)) => Err(failure(backend, &error)),
                Err(_) => Err(SpeechError::backend(backend, true, "sending timed out")),
            }
        })
    }

    /// Sends a JSON text message.
    pub(crate) fn send_json(
        &mut self,
        value: &Value,
        timeout: Duration,
    ) -> Result<(), SpeechError> {
        self.send(Message::Text(value.to_string().into()), timeout)
    }

    /// Runs `f` on the handler, while no message is handled.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut H) -> R) -> R {
        f(&mut self.reader.lock().handler)
    }

    /// The error that ended the connection, if one did.
    pub(crate) fn check(&self) -> Result<(), SpeechError> {
        match &self.reader.lock().error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Waits up to `timeout` until `done` holds for the handler.
    pub(crate) fn wait(
        &self,
        timeout: Duration,
        done: impl Fn(&H) -> bool,
    ) -> Result<(), SpeechError> {
        let deadline = instant_after(Instant::now(), timeout);
        let mut state = self.reader.lock();
        loop {
            if done(&state.handler) {
                return Ok(());
            }
            if let Some(error) = &state.error {
                return Err(error.clone());
            }
            if state.ended {
                return Err(SpeechError::backend(
                    self.backend,
                    true,
                    "the connection closed",
                ));
            }
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return Err(SpeechError::backend(
                    self.backend,
                    true,
                    "the server did not answer in time",
                ));
            };
            state = self
                .reader
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    /// Closes the connection politely, ignoring failures.
    pub(crate) fn close(&mut self, timeout: Duration) {
        let Some(mut sink) = self.sink.take() else {
            return;
        };
        let _ = self.runtime.block_on(async {
            let frame = CloseFrame {
                code: CloseCode::Normal,
                reason: "finished".into(),
            };
            tokio::time::timeout(timeout, sink.send(Message::Close(Some(frame)))).await
        });
    }
}

impl<H> Drop for Pushed<H> {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = self.runtime.block_on(task);
        }
        if let Some(mut sink) = self.sink.take() {
            let _ = self.runtime.block_on(async {
                let frame = CloseFrame {
                    code: CloseCode::Normal,
                    reason: "finished".into(),
                };
                let _ =
                    tokio::time::timeout(CLOSE_WAIT, sink.send(Message::Close(Some(frame)))).await;
                sink.close().await
            });
        }
    }
}

/// The text of a server error field, trimmed and capped at 300 characters.
pub(crate) fn field(value: &Value) -> Option<String> {
    let text = value.as_str()?.trim();
    if text.is_empty() {
        return None;
    }
    let mut out: String = text.chars().take(300).collect();
    if out.len() < text.len() {
        out.push('…');
    }
    Some(out)
}

/// `"{prefix} ({code}: {message})"`, degrading gracefully.
pub(crate) fn server_error(
    backend: &str,
    prefix: &str,
    code: &Value,
    message: &Value,
    retryable: bool,
) -> SpeechError {
    let detail = match (field(code), field(message)) {
        (Some(code), Some(message)) => format!("{prefix} ({code}: {message})"),
        (None, Some(message)) => format!("{prefix} ({message})"),
        (Some(code), None) => format!("{prefix} ({code})"),
        (None, None) => prefix.to_owned(),
    };
    SpeechError::backend(backend.to_owned(), retryable, detail)
}

#[cfg(feature = "dashscope")]
pub(crate) use tokio_tungstenite::tungstenite::Message as WsMessage;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints() {
        for good in [
            "wss://api.openai.com/v1/realtime?intent=transcription",
            "ws://127.0.0.1:9000/x",
            "ws://[::1]:9000/x",
            "WSS://example.com/x",
        ] {
            assert!(check_endpoint(good).is_ok(), "{good}");
        }
        for bad in [
            "https://x",
            "wss://",
            "wss://user:pw@x/",
            "wss://x/#f",
            "wss://x@/",
            "wss://a b/",
            "nothing",
        ] {
            assert!(check_endpoint(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn server_errors() {
        let long = "m".repeat(400);
        let error = server_error(
            "b",
            "failed",
            &Value::from("quota"),
            &Value::from(long.as_str()),
            true,
        );
        let text = std::error::Error::source(&error).unwrap().to_string();
        assert!(text.starts_with("failed (quota: mmm"), "{text}");
        assert!(text.ends_with("…)"), "{text}");
        assert!(error.retryable());
        let bare = server_error("b", "failed", &Value::Null, &Value::from(" "), false);
        assert_eq!(
            std::error::Error::source(&bare).unwrap().to_string(),
            "failed"
        );
        let code_only = server_error("b", "failed", &Value::from("x"), &Value::Null, false);
        assert_eq!(
            std::error::Error::source(&code_only).unwrap().to_string(),
            "failed (x)"
        );
    }

    #[test]
    fn failures_map_retryability() {
        assert!(failure("b", &WsError::ConnectionClosed).retryable());
        let reset = WsError::Protocol(ProtocolError::ResetWithoutClosingHandshake);
        assert!(failure("b", &reset).retryable());
        assert!(!failure("b", &WsError::Utf8(String::new())).retryable());
    }
}
