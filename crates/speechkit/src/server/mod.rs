#![doc = include_str!("README.md")]

mod builder;
mod deadline;
mod delta;
mod error;
mod speech;
mod stall;
mod state;
mod transcriptions;

#[cfg(test)]
mod tests;

use std::{future::Future, net::SocketAddr};

use crate::SpeechError;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::json;
use tokio::net::TcpListener;

pub use builder::Server;

impl Server {
    /// The routes, to serve yourself or nest in your own app. Without an
    /// engine, `/health` answers 503, so a health check notices;
    /// [`start`](Self::start) refuses to serve such a server. With an empty
    /// [auth token](Self::with_auth), every `/v1/` request answers 401.
    pub fn router(self) -> Router {
        let max_body_bytes = self.max_body_bytes;
        let body_timeout = self.body_timeout;
        let shared = state::shared(self);
        let v1 = Router::new().route("/v1/models", get(models));
        let v1 = v1.route(
            "/v1/audio/transcriptions",
            axum::routing::post(transcriptions::handle),
        );
        let v1 = v1.route("/v1/audio/speech", axum::routing::post(speech::handle));
        let v1 = v1.layer(axum::middleware::from_fn_with_state(
            shared.clone(),
            error::require_auth,
        ));
        Router::new()
            .route("/health", get(health))
            .merge(v1)
            .layer(DefaultBodyLimit::max(max_body_bytes))
            .layer(axum::middleware::from_fn_with_state(
                body_timeout,
                deadline::limit,
            ))
            .with_state(shared)
    }

    /// Binds the address and starts serving in the background, until
    /// [`RunningServer::shutdown`] or `stop` completes.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad address, a server with no
    /// engine, or an empty [auth token](Self::with_auth); or a backend
    /// error if the socket cannot be bound.
    pub async fn start(
        self,
        stop: impl Future<Output = ()> + Send + 'static,
    ) -> Result<RunningServer, SpeechError> {
        if self.asr.is_none() && self.tts.is_none() {
            return Err(SpeechError::InvalidInput(
                "the server has no engine to serve; add one with with_asr or with_tts".into(),
            ));
        }
        if self
            .auth
            .as_ref()
            .is_some_and(|token| token.expose().is_empty())
        {
            return Err(SpeechError::InvalidInput(
                "the auth token is empty; an empty token would let anyone in".into(),
            ));
        }
        let listener = bind(&self.bind).await?;
        let local_addr = listener.local_addr().map_err(io_error)?;
        let timeouts = Timeouts {
            header: self.header_timeout,
            write: self.write_timeout,
            shutdown: self.shutdown_timeout,
        };
        let app = self.router();
        let (shutdown, stopped) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let signal = async move {
                tokio::select! {
                    _ = stopped => {},
                    () = stop => {},
                }
            };
            serve_connections(listener, app, timeouts, signal).await;
            Ok(())
        });
        tracing::info!(%local_addr, "listening");
        Ok(RunningServer {
            local_addr,
            shutdown: Some(shutdown),
            task,
        })
    }

    /// Serves until SIGINT or SIGTERM, then lets in-flight requests finish.
    ///
    /// # Errors
    ///
    /// As [`start`](Self::start).
    pub async fn run(self) -> Result<(), SpeechError> {
        self.start(shutdown_signal()).await?.wait().await
    }
}

async fn health(State(state): State<state::Shared>) -> Response {
    if state.server.asr.is_none() && state.server.tts.is_none() {
        let body = json!({ "status": "no engine" });
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    }
    let mut body = json!({ "status": "ok" });
    if let Some((asr, id)) = &state.server.asr {
        body["model"] = json!(id);
        body["active_sessions"] = json!(asr.active_sessions());
        body["max_sessions"] = json!(asr.max_sessions());
    }
    if let Some((tts, id)) = &state.server.tts {
        body["speech"] = json!({
            "model": id,
            "active_sessions": tts.active_sessions(),
            "max_sessions": tts.max_sessions(),
        });
    }
    Json(body).into_response()
}

async fn models(State(state): State<state::Shared>) -> Response {
    let server = &state.server;
    let asr = server.asr.as_ref().map(|(engine, id)| (id, engine.name()));
    let tts = server.tts.as_ref().map(|(engine, id)| (id, engine.name()));
    let data: Vec<_> = asr
        .into_iter()
        .chain(tts)
        .map(|(id, backend)| {
            json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "speechkit",
                "backend": backend,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data })).into_response()
}

/// A server listening on a socket.
///
/// Dropping it stops the server as [`shutdown`](Self::shutdown) does,
/// without waiting: it stops accepting connections and lets in-flight
/// requests finish in the background. Keep it for as long as the server
/// should run, or call [`wait`](Self::wait) to serve until the `stop`
/// future given to [`Server::start`] completes.
#[must_use = "dropping a RunningServer stops the server"]
pub struct RunningServer {
    local_addr: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<Result<(), SpeechError>>,
}

impl RunningServer {
    /// The address it listens on, useful after binding port 0.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Stops accepting connections, lets in-flight requests finish, and
    /// waits for the server to stop.
    ///
    /// # Errors
    ///
    /// Any error the server hit while running.
    pub async fn shutdown(mut self) -> Result<(), SpeechError> {
        if let Some(stop) = self.shutdown.take() {
            let _ = stop.send(());
        }
        self.wait().await
    }

    /// Waits until the server stops.
    ///
    /// # Errors
    ///
    /// Any error the server hit while running.
    pub async fn wait(self) -> Result<(), SpeechError> {
        (&mut { self.task })
            .await
            .map_err(|e| SpeechError::backend("speechkit-server", false, e))?
    }
}

/// The per-connection timeouts `serve_connections` enforces.
struct Timeouts {
    header: std::time::Duration,
    write: std::time::Duration,
    shutdown: std::time::Duration,
}

/// Accepts connections until `signal`, then lets open ones finish for up
/// to the shutdown timeout, and closes the rest.
///
/// `axum::serve` sets no timer, so hyper would never enforce a header
/// timeout; serving each connection here fixes that, which stops clients
/// that trickle their headers from holding connections open.
async fn serve_connections(
    listener: TcpListener,
    app: Router,
    timeouts: Timeouts,
    signal: impl Future<Output = ()>,
) {
    use hyper_util::{
        rt::{TokioIo, TokioTimer},
        server::graceful::GracefulShutdown,
        service::TowerToHyperService,
    };

    let graceful = GracefulShutdown::new();
    // Set once the shutdown timeout passes: every connection still open
    // is dropped, which closes its socket.
    let (close, closed) = tokio::sync::watch::channel(false);
    let mut signal = std::pin::pin!(signal);
    loop {
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(error) => {
                    // Usually too many open files; wait instead of spinning.
                    tracing::warn!(%error, "accepting a connection failed");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                }
            },
            () = &mut signal => break,
        };
        let mut builder = hyper::server::conn::http1::Builder::new();
        builder
            .timer(TokioTimer::new())
            .header_read_timeout(timeouts.header);
        let stream = stall::WriteTimeout::new(stream, timeouts.write);
        let connection =
            builder.serve_connection(TokioIo::new(stream), TowerToHyperService::new(app.clone()));
        let connection = graceful.watch(connection);
        let mut closed = closed.clone();
        tokio::spawn(async move {
            tokio::select! {
                result = connection => {
                    if let Err(error) = result {
                        tracing::debug!(%error, "connection ended with an error");
                    }
                }
                _ = closed.wait_for(|&closed| closed) => {
                    tracing::debug!("connection closed at shutdown");
                }
            }
        });
    }
    drop(listener);
    let shutdown_timeout = timeouts.shutdown;
    if tokio::time::timeout(shutdown_timeout, graceful.shutdown())
        .await
        .is_err()
    {
        tracing::warn!(
            ?shutdown_timeout,
            "connections were still open after the shutdown timeout; closing them"
        );
        let _ = close.send(true);
    }
}

/// Listens on the first address `bind` names that can be bound: an address
/// such as `0.0.0.0:8080`, or a name and port such as `localhost:8080`.
async fn bind(bind: &str) -> Result<TcpListener, SpeechError> {
    let invalid = |why: &dyn std::fmt::Display| {
        SpeechError::InvalidInput(format!("invalid bind address {bind:?}: {why}"))
    };
    let addresses: Vec<SocketAddr> = tokio::net::lookup_host(bind)
        .await
        .map_err(|e| invalid(&e))?
        .collect();
    let mut last = None;
    for address in addresses {
        match TcpListener::bind(address).await {
            Ok(listener) => return Ok(listener),
            Err(error) => last = Some(error),
        }
    }
    Err(last.map_or_else(|| invalid(&"it names no address"), io_error))
}

fn io_error(error: std::io::Error) -> SpeechError {
    SpeechError::backend("speechkit-server", true, error)
}

/// Completes on Ctrl+C or, on Unix, SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
