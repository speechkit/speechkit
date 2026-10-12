//! `speechkit serve`.

use std::{
    io::Write,
    net::{SocketAddr, ToSocketAddrs},
    path::Path,
    time::Duration,
};

use speechkit::Secret;
use speechkit::server::{RunningServer, Server};

use crate::{BackendFactory, CliError, ServeArgs};

/// A server started by [`serve`], running on its own runtime.
pub struct ServeHandle {
    runtime: tokio::runtime::Runtime,
    server: RunningServer,
}

impl ServeHandle {
    /// The address the server listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.server.local_addr()
    }

    /// Stops the server after in-flight requests finish.
    ///
    /// # Errors
    ///
    /// Any error the server hit while running.
    pub fn shutdown(self) -> Result<(), CliError> {
        self.runtime
            .block_on(self.server.shutdown())
            .map_err(CliError::from)
    }

    fn wait_for_signal(self) -> Result<(), CliError> {
        self.runtime.block_on(async move {
            speechkit::server::shutdown_signal().await;
            self.server.shutdown().await
        })?;
        Ok(())
    }
}

/// The model id the server reports: `explicit`, else the model directory's
/// name, else the backend's name.
fn model_name(explicit: Option<&String>, model: Option<&Path>, backend: &str) -> String {
    explicit.cloned().unwrap_or_else(|| {
        model.and_then(Path::file_name).map_or_else(
            || backend.to_owned(),
            |name| name.to_string_lossy().into_owned(),
        )
    })
}

/// Whether every address `bind` names, such as `localhost:8080`, is a
/// loopback address. A name that does not resolve is not.
fn is_loopback(bind: &str) -> bool {
    bind.to_socket_addrs().is_ok_and(|addresses| {
        let mut any = false;
        let all = addresses.into_iter().all(|address| {
            any = true;
            address.ip().is_loopback()
        });
        any && all
    })
}

/// Builds the engine and starts serving it, without blocking.
///
/// # Errors
///
/// A usage error for a missing token variable, or any error building the
/// backend or binding the address.
pub fn serve(
    args: &ServeArgs,
    backends: &dyn BackendFactory,
    stderr: &mut dyn Write,
) -> Result<ServeHandle, CliError> {
    let auth = match &args.auth_token_env {
        Some(var) => Some(Secret::from_env(var).map_err(|e| CliError::usage(e.to_string()))?),
        None => None,
    };
    if auth.is_none() && !is_loopback(&args.bind) {
        let _ = writeln!(
            stderr,
            "warning: serving on {} without authentication; anyone who can reach it can use it \
             (set --auth-token-env)",
            args.bind
        );
    }
    let engine = backends.asr(&args.backend)?;
    let model_id = model_name(
        args.model_id.as_ref(),
        args.backend.model.as_deref(),
        engine.name(),
    );
    let mut server = Server::new()
        .with_asr(engine, model_id)
        .with_bind(args.bind.clone())
        .with_max_body_bytes(args.max_body_mib.saturating_mul(1024 * 1024));
    if let Some(tts_args) = args.tts() {
        let tts = backends.tts(&tts_args)?;
        let tts_id = model_name(
            args.tts_model_id.as_ref(),
            tts_args.model.as_deref(),
            tts.name(),
        );
        server = server.with_tts(tts, tts_id);
    }
    if let Some(seconds) = args.timeout {
        server = server.with_timeout(Duration::from_secs(seconds));
    }
    if let Some(language) = &args.backend.language {
        server = server.with_language(language.clone());
    }
    if let Some(auth) = auth {
        server = server.with_auth(auth);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("speechkit-serve")
        .build()?;
    let server = runtime.block_on(server.start(std::future::pending()))?;
    let _ = writeln!(stderr, "listening on http://{}", server.local_addr());
    Ok(ServeHandle { runtime, server })
}

pub(crate) fn serve_until_signal(
    args: &ServeArgs,
    backends: &dyn BackendFactory,
    stderr: &mut dyn Write,
) -> Result<(), CliError> {
    serve(args, backends, stderr)?.wait_for_signal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_detection() {
        assert!(is_loopback("127.0.0.1:8080"));
        assert!(is_loopback("[::1]:80"));
        assert!(is_loopback("localhost:8080"));
        assert!(!is_loopback("0.0.0.0:8080"));
        assert!(!is_loopback("not an address"));
    }
}
