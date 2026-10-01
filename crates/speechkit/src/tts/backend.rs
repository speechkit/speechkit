//! The traits a speech synthesis backend implements.

use std::sync::Arc;

use super::{TtsCapabilities, TtsOptions, Voice};
use crate::{Flow, SpeechError};

/// A loaded synthesis backend, shared by every session of an engine.
pub trait TtsBackend: Send + Sync + 'static {
    /// A short name for logs and errors.
    fn name(&self) -> &str;

    /// What the backend supports.
    fn capabilities(&self) -> &TtsCapabilities;

    /// The voices it offers.
    fn voices(&self) -> &[Voice];

    /// Opens a stream for one session. `opts` has already been checked
    /// against [`capabilities`](Self::capabilities) and
    /// [`voices`](Self::voices).
    ///
    /// # Errors
    ///
    /// Any error the backend hits while opening the stream.
    fn open(&self, opts: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError>;
}

/// A shared backend, so an `Arc<dyn TtsBackend>` can run an engine.
impl<B: TtsBackend + ?Sized> TtsBackend for Arc<B> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn capabilities(&self) -> &TtsCapabilities {
        (**self).capabilities()
    }

    fn voices(&self) -> &[Voice] {
        (**self).voices()
    }

    fn open(&self, opts: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
        (**self).open(opts)
    }
}

/// A boxed backend, so a `Box<dyn TtsBackend>` can run an engine.
impl<B: TtsBackend + ?Sized> TtsBackend for Box<B> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn capabilities(&self) -> &TtsCapabilities {
        (**self).capabilities()
    }

    fn voices(&self) -> &[Voice] {
        (**self).voices()
    }

    fn open(&self, opts: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
        (**self).open(opts)
    }
}

/// One session's synthesis stream.
///
/// The engine calls it from the session's worker thread, one chunk at a
/// time, in text order. Chunks are at most `max_chunk_chars` long.
pub trait TtsStream: Send {
    /// Synthesizes one chunk, calling `sink` with audio at the backend's
    /// native rate as it is produced. The sink may block while the
    /// session's output queue is full; that is how backpressure reaches
    /// the backend. If it returns [`Flow::Stop`], stop as soon as possible
    /// and return `Ok`.
    ///
    /// # Errors
    ///
    /// Any error the backend hits. The session fails with it.
    fn synthesize(
        &mut self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<(), SpeechError>;

    /// Asks the stream to stop early. Best effort; called at most once,
    /// between other calls.
    fn cancel(&mut self) {}
}
