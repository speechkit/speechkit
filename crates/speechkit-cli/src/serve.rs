//! `speechkit serve`.

use std::{io::Write, net::SocketAddr, time::Duration};

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

fn is_loopback(bind: &str) -> bool {
    bind.parse::<SocketAddr>()
        .is_ok_and(|address| address.ip().is_loopback())
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
    let model_id = args.model_id.clone().unwrap_or_else(|| {
        args.backend
            .model
            .as_ref()
            .and_then(|path| path.file_name())
            .map_or_else(
                || engine.name().to_owned(),
                |name| name.to_string_lossy().into_owned(),
            )
    });
    let mut server = Server::new()
        .with_asr(engine, model_id)
        .with_bind(args.bind.clone())
        .with_max_body_bytes(args.max_body_mib.saturating_mul(1024 * 1024));
    if let Some(tts_args) = args.tts() {
        let tts = backends.tts(&tts_args)?;
        let tts_id = args.tts_model_id.clone().unwrap_or_else(|| {
            tts_args
                .model
                .as_ref()
                .and_then(|path| path.file_name())
                .map_or_else(
                    || tts.name().to_owned(),
                    |name| name.to_string_lossy().into_owned(),
                )
        });
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
        .build()
        .map_err(|e| CliError::input(e.to_string()))?;
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
        assert!(!is_loopback("0.0.0.0:8080"));
        assert!(!is_loopback("not an address"));
    }
}
