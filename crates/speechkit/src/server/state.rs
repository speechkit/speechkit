//! The state a router's handlers share.

use std::sync::Arc;

use crate::server::Server;

pub(crate) struct Inner {
    /// Bounds transcription requests, uploads and decoding included, by
    /// the recognition engine's session limit.
    pub(crate) admission: Arc<tokio::sync::Semaphore>,
    pub(crate) server: Server,
}

pub(crate) type Shared = Arc<Inner>;

pub(crate) fn shared(server: Server) -> Shared {
    let slots = server
        .asr
        .as_ref()
        .map_or(0, |(engine, _)| engine.max_sessions());
    Arc::new(Inner {
        admission: Arc::new(tokio::sync::Semaphore::new(slots)),
        server,
    })
}
