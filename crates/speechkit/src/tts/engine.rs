//! The TTS engine.

use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{
    Chunker, TtsBackend, TtsCapabilities, TtsFailure, TtsLimits, TtsOptions, TtsOutput, TtsSession,
    TtsUpdate, Voice,
    session::{Shared, Worker, capacity, halves},
    validate,
};
use crate::{
    AudioBuffer, Deadline, RecvError, SpeechError,
    speech::{
        opening::{Opening, wait_open},
        slots::{DEFAULT_MAX_SESSIONS, SlotLimiter},
    },
};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

struct Inner {
    backend: Arc<dyn TtsBackend>,
    slots: SlotLimiter,
    max_sessions: NonZeroUsize,
    limits: TtsLimits,
}

/// Runs synthesis sessions on one backend. Clones share the backend and
/// the session limit.
#[derive(Clone)]
pub struct TtsEngine {
    inner: Arc<Inner>,
}

impl TtsEngine {
    /// An engine running `backend`: at most 8 sessions at once, each with
    /// the default [`TtsLimits`]. `backend` may also be shared or
    /// boxed, such as an `Arc<dyn TtsBackend>`.
    pub fn new(backend: impl TtsBackend) -> Self {
        Self::build(
            Arc::new(backend),
            SlotLimiter::new(DEFAULT_MAX_SESSIONS),
            DEFAULT_MAX_SESSIONS,
            TtsLimits::default(),
        )
    }

    fn build(
        backend: Arc<dyn TtsBackend>,
        slots: SlotLimiter,
        max_sessions: NonZeroUsize,
        limits: TtsLimits,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                backend,
                slots,
                max_sessions,
                limits,
            }),
        }
    }

    /// Lets `max` sessions run at once, across every clone: at least 1, so
    /// 0 is raised to 1. Default: 8.
    ///
    /// The returned engine has its own session slots, so call this before
    /// cloning the engine. [`with_limits`](Self::with_limits) keeps sharing
    /// the slots of the engine it is called on.
    #[must_use]
    pub fn with_max_sessions(self, max: usize) -> Self {
        let max = NonZeroUsize::new(max).unwrap_or(NonZeroUsize::MIN);
        let inner = &self.inner;
        Self::build(
            inner.backend.clone(),
            SlotLimiter::new(max),
            max,
            inner.limits,
        )
    }

    /// Sets the limits of each session.
    #[must_use]
    pub fn with_limits(self, limits: TtsLimits) -> Self {
        let inner = &self.inner;
        Self::build(
            inner.backend.clone(),
            inner.slots.clone(),
            inner.max_sessions,
            limits,
        )
    }

    /// The backend's name.
    pub fn name(&self) -> &str {
        self.inner.backend.name()
    }

    /// What the backend supports.
    pub fn capabilities(&self) -> &TtsCapabilities {
        self.inner.backend.capabilities()
    }

    /// The voices the backend offers.
    pub fn voices(&self) -> &[Voice] {
        self.inner.backend.voices()
    }

    /// The most sessions that run at once, across every clone.
    pub fn max_sessions(&self) -> usize {
        self.inner.max_sessions.get()
    }

    /// The limits of each session.
    pub fn limits(&self) -> &TtsLimits {
        &self.inner.limits
    }

    /// Sessions holding a slot, across every clone.
    pub fn active_sessions(&self) -> usize {
        self.inner.slots.in_use()
    }

    /// Starts a synthesis: waits for a free slot, then opens the backend,
    /// all within `deadline`. Returns the text side and the audio side
    /// together, so no audio is made before it has a reader (T-02).
    ///
    /// # Errors
    ///
    /// - [`SpeechError::InvalidInput`] for an unknown voice, or a speed
    ///   that is not positive or outside the backend's range;
    /// - [`SpeechError::Unsupported`] for a speed other than 1.0 on a
    ///   backend without speed control, or a language no voice speaks;
    /// - [`SpeechError::Capacity`] if the deadline passes while every slot
    ///   is busy;
    /// - [`SpeechError::DeadlineExceeded`] if it passes while the backend
    ///   opens, whose late stream is dropped when the open returns, holding
    ///   the slot until then (T-07). A deadline that has already passed
    ///   takes no slot;
    /// - the backend's error opening a stream.
    pub fn start(
        &self,
        opts: TtsOptions,
        deadline: impl Into<Deadline>,
    ) -> Result<(TtsSession, TtsOutput), SpeechError> {
        self.start_with(opts, deadline.into(), Opening::Wait)
    }

    /// [`start`](Self::start), with the slot taken as `opening` says.
    pub(crate) fn start_with(
        &self,
        opts: TtsOptions,
        deadline: Deadline,
        opening: Opening,
    ) -> Result<(TtsSession, TtsOutput), SpeechError> {
        let inner = &*self.inner;
        let backend = &inner.backend;
        let caps = backend.capabilities();
        validate(&opts, caps, backend.voices())?;
        let rate = opts.sample_rate.unwrap_or(caps.sample_rate);
        // Nothing costly happens before the deadline and the slot are dealt
        // with: the session's thread builds the resampler once it has a slot.
        let slot = opening.take(&inner.slots, deadline)?;
        let id = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let shared = Arc::new(Shared::new(
            id,
            rate,
            capacity(rate, inner.limits.output_queue),
            inner.limits.max_text_chars,
            inner.slots.clone(),
        ));
        let chunk_chars = inner.limits.chunk_chars.min(caps.max_chunk_chars).max(1);
        let worker = Worker {
            shared: shared.clone(),
            backend: backend.clone(),
            options: opts,
            from: caps.sample_rate,
            to: rate,
            chunker: Chunker::new(chunk_chars),
            slot,
        };
        std::thread::Builder::new()
            .name(format!("speechkit-tts-{id}"))
            .spawn(move || worker.run())
            .map_err(|e| SpeechError::backend("speechkit", true, e))?;
        let (session, output) = halves(&shared);
        if opening == Opening::Background {
            return Ok((session, output));
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
            |core, error| shared.fail(&core, error),
        )?;
        Ok((session, output))
    }

    /// Synthesizes `text` in one call: starts a session, pushes the text,
    /// and reads the output, all within `deadline`.
    ///
    /// # Errors
    ///
    /// A [`TtsFailure`] with the progress made. Empty text fails without
    /// starting a session (T-04).
    pub fn synthesize(
        &self,
        text: &str,
        opts: TtsOptions,
        deadline: impl Into<Deadline>,
    ) -> Result<AudioBuffer, TtsFailure> {
        let deadline = deadline.into();
        if text.trim().is_empty() {
            return Err(TtsFailure::new(SpeechError::InvalidInput(
                "there is no text to synthesize".into(),
            )));
        }
        let (session, mut output) = self.start(opts, deadline).map_err(TtsFailure::new)?;
        session.push_text(text).map_err(TtsFailure::new)?;
        session.close_text();
        let mut samples = Vec::new();
        loop {
            match output.recv(deadline) {
                Ok(TtsUpdate::Audio(piece)) => samples.extend_from_slice(&piece),
                Ok(TtsUpdate::Closed(result)) => {
                    return result.map(|_| AudioBuffer::new(output.sample_rate(), samples));
                }
                Ok(_) => {}
                // The deadline passed: `finish` fails the synthesis.
                Err(RecvError::Timeout | RecvError::Empty | RecvError::Closed) => {
                    return session
                        .finish(deadline)
                        .map(|_| AudioBuffer::new(output.sample_rate(), samples));
                }
            }
        }
    }
}

impl std::fmt::Debug for TtsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsEngine")
            .field("backend", &self.name())
            .field("active_sessions", &self.active_sessions())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{Flow, SampleRate, speech::resample::filters_built_here, tts::TtsStream};

    /// A synthesizer that makes nothing, at 24 kHz, so output at 16 kHz
    /// needs a resampler.
    struct Silent(TtsCapabilities);

    impl TtsBackend for Silent {
        fn name(&self) -> &'static str {
            "silent"
        }

        fn capabilities(&self) -> &TtsCapabilities {
            &self.0
        }

        fn voices(&self) -> &[Voice] {
            &[]
        }

        fn open(&self, _: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
            Ok(Box::new(Nothing))
        }
    }

    struct Nothing;

    impl TtsStream for Nothing {
        fn synthesize(
            &mut self,
            _: &str,
            _: &mut dyn FnMut(&[f32]) -> Flow,
        ) -> Result<(), SpeechError> {
            Ok(())
        }
    }

    /// A limit of zero would leave every `start` waiting for nothing.
    #[test]
    fn a_limit_of_zero_is_raised_to_one() {
        let backend = Silent(TtsCapabilities::new(SampleRate::HZ_16000, 1_000));
        let engine = TtsEngine::new(backend).with_max_sessions(0);
        assert_eq!(engine.max_sessions(), 1);
        let started = engine
            .start(TtsOptions::default(), Duration::from_secs(10))
            .expect("the one slot");
        assert_eq!(engine.active_sessions(), 1);
        drop(started);
    }

    /// `start` answers a busy engine, and takes a free slot, without
    /// building the resampler on the caller's thread: the server asks for
    /// its own rate on every request, and rejects with 503 when busy.
    #[test]
    fn start_builds_no_resampler_on_the_callers_thread() {
        let backend = Silent(TtsCapabilities::new(SampleRate::HZ_24000, 1_000));
        let engine = TtsEngine::new(backend).with_max_sessions(1);
        let options = || TtsOptions::default().with_sample_rate(SampleRate::HZ_16000);
        let before = filters_built_here();
        let busy = engine
            .start(options(), Duration::from_secs(10))
            .expect("a free slot");
        let rejected =
            engine.start_with(options(), Duration::from_secs(10).into(), Opening::NoWait);
        assert!(matches!(rejected, Err(SpeechError::Capacity)));
        let timed_out = engine.start(options(), Duration::from_millis(100));
        assert!(matches!(timed_out, Err(SpeechError::Capacity)));
        assert_eq!(filters_built_here(), before);
        drop(busy);
    }
}
