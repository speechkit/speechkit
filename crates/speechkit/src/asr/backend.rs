//! The traits a speech recognition backend implements.

use std::{sync::Arc, time::Duration};

use super::{AsrCapabilities, AsrEvents, AsrOptions, Partial, Segment, UtteranceId};
use crate::SpeechError;

/// A loaded speech recognition backend: a model, or a client for a
/// service. Shared by every session of an engine.
pub trait AsrBackend: Send + Sync + 'static {
    /// A short name for logs and errors, such as `sherpa-streaming`.
    fn name(&self) -> &str;

    /// What the backend supports.
    fn capabilities(&self) -> &AsrCapabilities;

    /// Opens a stream for one session, which sends its results through
    /// `events`. `options` has already been checked against
    /// [`capabilities`](Self::capabilities).
    ///
    /// # Errors
    ///
    /// Any error the backend hits while opening the stream.
    fn open(
        &self,
        options: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError>;
}

/// A shared backend, so an `Arc<dyn AsrBackend>` can run an engine.
impl<B: AsrBackend + ?Sized> AsrBackend for Arc<B> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn capabilities(&self) -> &AsrCapabilities {
        (**self).capabilities()
    }

    fn open(
        &self,
        options: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        (**self).open(options, events)
    }
}

/// A boxed backend, so a `Box<dyn AsrBackend>` can run an engine.
impl<B: AsrBackend + ?Sized> AsrBackend for Box<B> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn capabilities(&self) -> &AsrCapabilities {
        (**self).capabilities()
    }

    fn open(
        &self,
        options: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        (**self).open(options, events)
    }
}

/// A result a backend stream sends through its [`AsrEvents`].
///
/// Times count from the first sample the stream was fed; the engine adds
/// the session's origin. Deliberately exhaustive: a new kind of event
/// changes what every backend must do, so adding one breaks them rather
/// than being silently ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrEvent {
    /// New pending text for an utterance.
    Partial(Partial),
    /// The final text of an utterance, even one with no words. Every
    /// utterance ends with exactly one.
    Segment(Segment),
    /// Speech started here.
    SpeechStarted {
        /// Where speech started.
        at: Duration,
    },
    /// Speech ended here. `utterance` is the last utterance of that speech,
    /// whose segment may come later.
    SpeechEnded {
        /// Where speech ended.
        at: Duration,
        /// The last utterance of the speech.
        utterance: UtteranceId,
    },
    /// Every start and end of speech before `through` has been sent.
    ActivityKnown {
        /// How far activity is known.
        through: Duration,
    },
}

/// One session's recognition stream.
///
/// The engine calls it from one thread, never calls `accept` after
/// `finish`, and passes mono samples that are finite, in [-1.0, 1.0], and
/// already at the backend's `sample_rate`.
///
/// The stream sends its results through the [`AsrEvents`] it was opened
/// with, from `accept` or from a thread of its own. Events arrive in order,
/// utterance IDs never decrease, and a `Segment` replaces the pending text
/// of its utterance, with no events for that utterance after it. Dropping
/// the stream must stop any work it runs in the background, since the
/// session's slot is freed only then.
pub trait AsrStream: Send {
    /// Feeds audio.
    ///
    /// # Errors
    ///
    /// Any error the backend hits. The session fails with it.
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError>;

    /// Signals the end of input, and returns once every final event has
    /// been sent.
    ///
    /// # Errors
    ///
    /// Any error the backend hits. The session fails with it.
    fn finish(&mut self) -> Result<(), SpeechError>;

    /// Asks the stream to stop early. Best effort: native calls may not be
    /// interruptible. Called at most once, between other calls.
    fn cancel(&mut self) {}
}
