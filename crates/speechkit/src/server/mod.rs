#![doc = include_str!("README.md")]

mod builder;
mod delta;
mod error;
mod speech;
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
    /// [`start`](Self::start) refuses to serve such a server.
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
            .layer(tower_http::timeout::RequestBodyTimeoutLayer::new(
                body_timeout,
            ))
            .with_state(shared)
    }

    /// Binds the address and starts serving in the background, until
    /// [`RunningServer::shutdown`] or `stop` completes.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad address or a server with no
    /// engine, or a backend error if the socket cannot be bound.
    pub async fn start(
        self,
        stop: impl Future<Output = ()> + Send + 'static,
    ) -> Result<RunningServer, SpeechError> {
        if self.asr.is_none() && self.tts.is_none() {
            return Err(SpeechError::InvalidInput(
                "the server has no engine to serve; add one with with_asr or with_tts".into(),
            ));
        }
        let address: SocketAddr = self.bind.parse().map_err(|_| {
            SpeechError::InvalidInput(format!("invalid bind address {:?}", self.bind))
        })?;
        let listener = TcpListener::bind(address).await.map_err(io_error)?;
        let local_addr = listener.local_addr().map_err(io_error)?;
        let header_timeout = self.header_timeout;
        let app = self.router();
        let (shutdown, stopped) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let signal = async move {
                tokio::select! {
                    _ = stopped => {},
                    () = stop => {},
                }
            };
            serve_connections(listener, app, header_timeout, signal).await;
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

/// Accepts connections until `signal`, then lets open ones finish.
///
/// `axum::serve` sets no timer, so hyper would never enforce a header
/// timeout; serving each connection here fixes that, which stops clients
/// that trickle their headers from holding connections open.
async fn serve_connections(
    listener: TcpListener,
    app: Router,
    header_timeout: std::time::Duration,
    signal: impl Future<Output = ()>,
) {
    use hyper_util::{
        rt::{TokioIo, TokioTimer},
        server::graceful::GracefulShutdown,
        service::TowerToHyperService,
    };

    let graceful = GracefulShutdown::new();
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
            .header_read_timeout(header_timeout);
        let connection =
            builder.serve_connection(TokioIo::new(stream), TowerToHyperService::new(app.clone()));
        let connection = graceful.watch(connection);
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::debug!(%error, "connection ended with an error");
            }
        });
    }
    drop(listener);
    graceful.shutdown().await;
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
