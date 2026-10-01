//! The ASR engine.

use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use super::{
    AsrBackend, AsrCapabilities, AsrFailure, AsrLimits, AsrOptions, AsrResult, PostProcessor,
    post::PostProcessing,
    session::{AsrSession, Endpoints, Settings, Shared, Worker},
    validate,
};
use crate::{
    AudioBuffer, Deadline, SampleRate, SpeechError,
    speech::{
        audio::find_invalid,
        opening::{Opening, wait_open},
        slots::{DEFAULT_MAX_SESSIONS, SlotLimiter},
    },
};

/// Process-wide session counter, for session IDs and thread names.
static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

struct Inner {
    backend: Arc<dyn AsrBackend>,
    slots: SlotLimiter,
    max_sessions: NonZeroUsize,
    limits: AsrLimits,
    post: Option<Arc<dyn PostProcessor>>,
}

/// Runs recognition sessions on one backend.
///
/// Clones share the backend and the session limit (A-16). Each session
/// runs on its own thread and holds one slot until its backend stream has
/// been dropped, so work that is still running always counts against the
/// limit (A-09).
#[derive(Clone)]
pub struct AsrEngine {
    inner: Arc<Inner>,
}

impl AsrEngine {
    /// An engine running `backend`: at most 8 sessions at once, each with
    /// the default [`AsrLimits`]. `backend` may also be shared or
    /// boxed, such as an `Arc<dyn AsrBackend>`.
    pub fn new(backend: impl AsrBackend) -> Self {
        Self::build(
            Arc::new(backend),
            SlotLimiter::new(DEFAULT_MAX_SESSIONS),
            DEFAULT_MAX_SESSIONS,
            AsrLimits::default(),
            None,
        )
    }

    fn build(
        backend: Arc<dyn AsrBackend>,
        slots: SlotLimiter,
        max_sessions: NonZeroUsize,
        limits: AsrLimits,
        post: Option<Arc<dyn PostProcessor>>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                backend,
                slots,
                max_sessions,
                limits,
                post,
            }),
        }
    }

    /// Lets `max` sessions run at once, across every clone: at least 1, so
    /// 0 is raised to 1. Default: 8.
    ///
    /// The returned engine has its own session slots, so call this before
    /// cloning the engine. The other `with_*` methods keep sharing the
    /// slots of the engine they are called on (A-16).
    #[must_use]
    pub fn with_max_sessions(self, max: usize) -> Self {
        let max = NonZeroUsize::new(max).unwrap_or(NonZeroUsize::MIN);
        let inner = &self.inner;
        Self::build(
            inner.backend.clone(),
            SlotLimiter::new(max),
            max,
            inner.limits,
            inner.post.clone(),
        )
    }

    /// Sets the limits of each session.
    #[must_use]
    pub fn with_limits(self, limits: AsrLimits) -> Self {
        let inner = &self.inner;
        Self::build(
            inner.backend.clone(),
            inner.slots.clone(),
            inner.max_sessions,
            limits,
            inner.post.clone(),
        )
    }

    /// Applies `processor` to every committed segment, for example to add
    /// punctuation.
    #[must_use]
    pub fn with_post_processor(self, processor: impl PostProcessor) -> Self {
        let inner = &self.inner;
        Self::build(
            inner.backend.clone(),
            inner.slots.clone(),
            inner.max_sessions,
            inner.limits,
            Some(Arc::new(processor)),
        )
    }

    /// The backend's name.
    pub fn name(&self) -> &str {
        self.inner.backend.name()
    }

    /// What the backend supports.
    pub fn capabilities(&self) -> &AsrCapabilities {
        self.inner.backend.capabilities()
    }

    /// The most sessions that run at once, across every clone.
    pub fn max_sessions(&self) -> usize {
        self.inner.max_sessions.get()
    }

    /// The limits of each session.
    pub fn limits(&self) -> &AsrLimits {
        &self.inner.limits
    }

    /// Sessions currently holding a slot, across every clone. A session
    /// holds its slot until its worker thread exits.
    pub fn active_sessions(&self) -> usize {
        self.inner.slots.in_use()
    }

    /// Starts a session for audio at `sample_rate`: waits for a free slot,
    /// then opens the backend's stream (a WebSocket handshake, say), all
    /// within `deadline`. The session resamples the audio to the rate the
    /// backend wants.
    ///
    /// # Errors
    ///
    /// - [`SpeechError::Unsupported`] at once if the options ask for
    ///   something the backend's [`AsrCapabilities`] rule out: hints without
    ///   `accepts_hints`, a language without `accepts_language`, or a turn
    ///   end, an end at a pause, or a no-speech timeout without
    ///   `reports_activity`;
    /// - [`SpeechError::InvalidInput`] at once if the limits are invalid or
    ///   a duration in the options is zero;
    /// - [`SpeechError::Capacity`] if the deadline passes while every slot
    ///   is busy (A-14);
    /// - [`SpeechError::DeadlineExceeded`] if it passes while the backend
    ///   opens, whose late stream is dropped when the open returns, holding
    ///   the slot until then. A deadline that has already passed takes no
    ///   slot (A-19);
    /// - any error the backend returns when opening its stream.
    pub fn start(
        &self,
        sample_rate: SampleRate,
        options: AsrOptions,
        deadline: impl Into<Deadline>,
    ) -> Result<AsrSession, SpeechError> {
        self.start_with(
            sample_rate,
            options,
            deadline.into(),
            Opening::Wait,
            Duration::ZERO,
        )
    }

    /// [`start`](Self::start), with the slot taken as `opening` says, and
    /// times counting from `origin`.
    pub(crate) fn start_with(
        &self,
        sample_rate: SampleRate,
        options: AsrOptions,
        deadline: Deadline,
        opening: Opening,
        origin: Duration,
    ) -> Result<AsrSession, SpeechError> {
        let inner = &*self.inner;
        let caps = inner.backend.capabilities();
        validate(&options, caps)?;
        let geometry = inner.limits.geometry(sample_rate, caps.sample_rate)?;
        // Nothing costly happens before the deadline and the slot are dealt
        // with: the session's thread builds the resampler once it has a slot.
        let slot = opening.take(&inner.slots, deadline)?;
        let id = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let shared = Arc::new(Shared::new(Settings {
            id,
            sample_rate,
            origin,
            capacity: geometry.capacity,
            max_history_bytes: inner.limits.max_history_bytes,
            post: inner.post.clone().map(PostProcessing::new),
            slots: inner.slots.clone(),
            endpoints: Endpoints::new(&options, origin),
        }));
        let max_length = options.max_length;
        let worker = Worker {
            shared: shared.clone(),
            backend: inner.backend.clone(),
            options,
            from: sample_rate,
            to: caps.sample_rate,
            block: geometry.block,
            max_length,
            slot,
        };
        std::thread::Builder::new()
            .name(format!("speechkit-asr-{id}"))
            .spawn(move || worker.run())
            .map_err(|error| SpeechError::backend("speechkit", true, error))?;
        let session = AsrSession {
            shared: shared.clone(),
        };
        if opening == Opening::Background {
            return Ok(session);
        }
        wait_open(
            &shared.changed,
            shared.lock(),
            deadline,
            |core| core.stage,
            || {
                shared.terminal.get().map(|result| match result.as_ref() {
                    Err(failure) => failure.error.clone(),
                    Ok(_) => SpeechError::Closed,
                })
            },
            |mut core, error| shared.settle_failure(&mut core, error),
        )?;
        Ok(session)
    }

    /// Transcribes a whole recording: starts a session at the recording's
    /// rate, pushes the audio, and finishes the session, all within
    /// `deadline`. Waits for a free slot within the deadline.
    ///
    /// # Errors
    ///
    /// An [`AsrFailure`] holding the error and what was confirmed. A
    /// deadline that has already passed fails without starting a session
    /// (A-19).
    pub fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: AsrOptions,
        deadline: impl Into<Deadline>,
    ) -> AsrResult {
        let deadline = deadline.into();
        let fail = |error| Err(AsrFailure::new(error));
        if deadline.remaining().is_none() {
            return fail(SpeechError::DeadlineExceeded);
        }
        if let Some(bad) = find_invalid(&audio.samples) {
            return fail(bad.into());
        }
        let session = match self.start(audio.sample_rate, options, deadline) {
            Ok(session) => session,
            Err(error) => return fail(error),
        };
        let chunk = usize::try_from(audio.sample_rate.frames_in(self.inner.limits.input_queue))
            .unwrap_or(usize::MAX)
            .max(1);
        for piece in audio.samples.chunks(chunk) {
            if session.push(piece, deadline).is_err() {
                break;
            }
        }
        session.finish(deadline)
    }
}

impl std::fmt::Debug for AsrEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsrEngine")
            .field("backend", &self.name())
            .field("max_sessions", &self.max_sessions())
            .field("limits", &self.inner.limits)
            .field("active_sessions", &self.active_sessions())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use super::*;
    use crate::{
        asr::{AsrEvents, AsrStream},
        speech::resample::filters_built_here,
    };

    /// A recognizer that hears nothing, at 24 kHz, so 16 kHz audio needs a
    /// resampler.
    struct Silent(AsrCapabilities);

    impl AsrBackend for Silent {
        fn name(&self) -> &'static str {
            "silent"
        }

        fn capabilities(&self) -> &AsrCapabilities {
            &self.0
        }

        fn open(&self, _: &AsrOptions, _: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
            Ok(Box::new(Nothing))
        }
    }

    struct Nothing;

    impl AsrStream for Nothing {
        fn accept(&mut self, _: &[f32]) -> Result<(), SpeechError> {
            Ok(())
        }

        fn finish(&mut self) -> Result<(), SpeechError> {
            Ok(())
        }
    }

    /// A limit of zero would leave every `start` waiting for nothing.
    #[test]
    fn a_limit_of_zero_is_raised_to_one() {
        let backend = Silent(AsrCapabilities::new(SampleRate::HZ_16000));
        let engine = AsrEngine::new(backend).with_max_sessions(0);
        assert_eq!(engine.max_sessions(), 1);
        let session = engine
            .start(
                SampleRate::HZ_16000,
                AsrOptions::default(),
                Duration::from_secs(10),
            )
            .expect("the one slot");
        assert_eq!(engine.active_sessions(), 1);
        drop(session);
    }

    /// `start` answers a busy engine, and takes a free slot, without building
    /// the resampler on the caller's thread: a short deadline is spent on
    /// waiting for the slot, not on a sinc table (A-14).
    #[test]
    fn start_builds_no_resampler_on_the_callers_thread() {
        let backend = Silent(AsrCapabilities::new(SampleRate::HZ_24000));
        let engine = AsrEngine::new(backend).with_max_sessions(1);
        let rate = SampleRate::HZ_16000;
        let before = filters_built_here();
        let busy = engine
            .start(rate, AsrOptions::default(), Duration::from_secs(10))
            .expect("a free slot");
        let rejected = engine.start_with(
            rate,
            AsrOptions::default(),
            Duration::from_secs(10).into(),
            Opening::NoWait,
            Duration::ZERO,
        );
        assert!(matches!(rejected, Err(SpeechError::Capacity)));
        let timed_out = engine.start(rate, AsrOptions::default(), Duration::from_millis(100));
        assert!(matches!(timed_out, Err(SpeechError::Capacity)));
        assert_eq!(filters_built_here(), before);
        drop(busy);
    }
}
