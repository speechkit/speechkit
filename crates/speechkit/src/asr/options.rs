//! Session options, backend capabilities, and session limits.

use std::time::Duration;

use crate::{SampleRate, SpeechError};

/// What a backend supports.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AsrCapabilities {
    /// It reports partial results before an utterance is final.
    pub reports_partials: bool,
    /// It reports where speech starts and ends, and how far that is known.
    pub reports_activity: bool,
    /// It accepts [`AsrOptions::hints`].
    pub accepts_hints: bool,
    /// It accepts [`AsrOptions::language`].
    pub accepts_language: bool,
    /// Its text already contains punctuation.
    pub punctuated: bool,
    /// The rate the backend wants. Sessions resample to it.
    pub sample_rate: SampleRate,
}

impl AsrCapabilities {
    /// Capabilities with every flag off.
    pub const fn new(sample_rate: SampleRate) -> Self {
        Self {
            reports_partials: false,
            reports_activity: false,
            accepts_hints: false,
            accepts_language: false,
            punctuated: false,
            sample_rate,
        }
    }
}

/// Options for one session.
///
/// The last four are measured in audio time. `turn_end`,
/// `end_after_silence`, and `no_speech_timeout` need a backend that
/// [`reports_activity`](AsrCapabilities::reports_activity): each waits
/// until the backend has confirmed the silence, so a start of speech
/// reported late can't be missed (A-07, A-08).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct AsrOptions {
    /// The spoken language, for backends that accept an override.
    pub language: Option<String>,
    /// Phrases to favor, such as names and jargon, for backends that accept
    /// session hints. Empty means no hints (A-18).
    pub hints: Vec<String>,
    /// Report [`TurnEnded`](super::AsrUpdate::TurnEnded) after this much
    /// silence.
    pub turn_end: Option<Duration>,
    /// End the session at the first pause this long.
    pub end_after_silence: Option<Duration>,
    /// End the session here if no speech has started.
    pub no_speech_timeout: Option<Duration>,
    /// End the session after this much audio.
    pub max_length: Option<Duration>,
}

impl AsrOptions {
    /// Sets the language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the hints.
    #[must_use]
    pub fn with_hints(mut self, phrases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.hints = phrases.into_iter().map(Into::into).collect();
        self
    }

    /// Reports where each turn ends: once speech has ended and the backend
    /// has confirmed `silence` after it with no speech starting again. A
    /// shorter pause, or an utterance cut at its maximum, doesn't end a
    /// turn.
    #[must_use]
    pub fn with_turn_end(mut self, silence: Duration) -> Self {
        self.turn_end = Some(silence);
        self
    }

    /// Ends the session at the first pause of `silence` after speech. The
    /// session transcribes the audio up to where the backend confirmed the
    /// pause, and its result's `duration` says where that was.
    #[must_use]
    pub fn with_end_after_silence(mut self, silence: Duration) -> Self {
        self.end_after_silence = Some(silence);
        self
    }

    /// Ends the session, with an empty transcript, once `timeout` of audio
    /// has passed with no speech.
    #[must_use]
    pub fn with_no_speech_timeout(mut self, timeout: Duration) -> Self {
        self.no_speech_timeout = Some(timeout);
        self
    }

    /// Ends the session after exactly `length` of audio.
    #[must_use]
    pub fn with_max_length(mut self, length: Duration) -> Self {
        self.max_length = Some(length);
        self
    }
}

/// Checks `opts` against what a backend supports.
///
/// This is the only place a capability mismatch becomes
/// [`SpeechError::Unsupported`]. No hints count as no hints.
///
/// # Errors
///
/// [`SpeechError::Unsupported`] if the options ask for something the
/// backend cannot do, or [`SpeechError::InvalidInput`] for a zero
/// duration.
pub(crate) fn validate(opts: &AsrOptions, caps: &AsrCapabilities) -> Result<(), SpeechError> {
    let durations = [
        opts.turn_end,
        opts.end_after_silence,
        opts.no_speech_timeout,
        opts.max_length,
    ];
    if durations.iter().flatten().any(Duration::is_zero) {
        return Err(SpeechError::InvalidInput(
            "turn_end, end_after_silence, no_speech_timeout, and max_length must be positive"
                .into(),
        ));
    }
    let needs_activity = opts.turn_end.is_some()
        || opts.end_after_silence.is_some()
        || opts.no_speech_timeout.is_some();
    if needs_activity && !caps.reports_activity {
        return Err(SpeechError::Unsupported(
            "this backend does not report speech activity, which turn ends, ending at a pause, \
             and the no-speech timeout need"
                .into(),
        ));
    }
    if !opts.hints.is_empty() && !caps.accepts_hints {
        return Err(SpeechError::Unsupported(
            "this backend does not accept per-session speech hints".into(),
        ));
    }
    if opts.language.is_some() && !caps.accepts_language {
        return Err(SpeechError::Unsupported(
            "this backend does not accept a language override".into(),
        ));
    }
    Ok(())
}

/// Per-session limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct AsrLimits {
    /// How much audio the input queue holds, which is also the longest
    /// chunk one push takes. Default: 2 s.
    pub input_queue: Duration,
    /// The most history a session keeps for its readers: segment text plus
    /// a small fixed cost per speech event. Default: 8 MiB, about 100 hours
    /// of speech.
    pub max_history_bytes: usize,
}

impl Default for AsrLimits {
    fn default() -> Self {
        Self {
            input_queue: Duration::from_secs(2),
            max_history_bytes: 8 * 1024 * 1024,
        }
    }
}

impl AsrLimits {
    /// Sets [`input_queue`](Self::input_queue).
    #[must_use]
    pub const fn with_input_queue(mut self, value: Duration) -> Self {
        self.input_queue = value;
        self
    }

    /// Sets [`max_history_bytes`](Self::max_history_bytes).
    #[must_use]
    pub const fn with_max_history_bytes(mut self, value: usize) -> Self {
        self.max_history_bytes = value;
        self
    }
}

/// How much audio the backend is fed at a time, or the whole input queue if
/// that is shorter. Ending at a pause is checked after each block, so a
/// local model stops at the same point however fast the audio was pushed.
/// It is not a limit callers set: any value gives that guarantee, and a
/// setting would only let the pace of the audio move where a session ends.
const FEED_BLOCK: Duration = Duration::from_millis(100);

/// The limits in frames, for one session.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    /// Input queue capacity, at the session's input rate.
    pub(crate) capacity: usize,
    /// Feed block, at the backend's rate.
    pub(crate) block: usize,
}

/// The largest input queue allowed: 16 Mi frames, 64 MiB of f32.
const MAX_QUEUE_FRAMES: u64 = 16 * 1024 * 1024;

impl AsrLimits {
    /// Checks the limits and converts them to frames, the queue at the
    /// input rate and the feed block at the backend's.
    pub(crate) fn geometry(
        &self,
        input: SampleRate,
        backend: SampleRate,
    ) -> Result<Geometry, SpeechError> {
        let invalid = |message: &str| Err(SpeechError::InvalidInput(message.into()));
        let capacity = input.frames_in(self.input_queue);
        let block = backend.frames_in(FEED_BLOCK.min(self.input_queue));
        if capacity == 0 || block == 0 {
            return invalid("input_queue must cover at least one frame");
        }
        if capacity > MAX_QUEUE_FRAMES {
            return invalid("input_queue is too large");
        }
        if self.max_history_bytes == 0 {
            return invalid("max_history_bytes must be positive");
        }
        Ok(Geometry {
            capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
            block: usize::try_from(block).unwrap_or(usize::MAX),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_table() {
        let hint_cases: [(&[&str], bool); 2] = [(&[], false), (&["speechkit"], true)];
        for (hints, asks_hints) in hint_cases {
            for language in [None, Some("zh".to_owned())] {
                for accepts_hints in [false, true] {
                    for accepts_language in [false, true] {
                        for other in [false, true] {
                            let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
                            caps.accepts_hints = accepts_hints;
                            caps.accepts_language = accepts_language;
                            caps.reports_partials = other;
                            caps.punctuated = other;
                            let mut opts = AsrOptions::default().with_hints(hints.iter().copied());
                            opts.language.clone_from(&language);
                            let ok = (!asks_hints || accepts_hints)
                                && (language.is_none() || accepts_language);
                            let result = validate(&opts, &caps);
                            assert_eq!(result.is_ok(), ok, "{opts:?} {caps:?}");
                            if let Err(error) = result {
                                assert!(matches!(error, SpeechError::Unsupported(_)));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn endpoints_need_activity_and_positive_durations() {
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        let second = Duration::from_secs(1);
        let needing = [
            AsrOptions::default().with_turn_end(second),
            AsrOptions::default().with_end_after_silence(second),
            AsrOptions::default().with_no_speech_timeout(second),
        ];
        for opts in &needing {
            assert!(matches!(
                validate(opts, &caps),
                Err(SpeechError::Unsupported(_))
            ));
        }
        let length = AsrOptions::default().with_max_length(second);
        assert!(validate(&length, &caps).is_ok());
        caps.reports_activity = true;
        for opts in &needing {
            assert!(validate(opts, &caps).is_ok());
        }
        let zero = AsrOptions::default().with_max_length(Duration::ZERO);
        assert!(matches!(
            validate(&zero, &caps),
            Err(SpeechError::InvalidInput(_))
        ));
    }

    #[test]
    fn default_limits() {
        let limits = AsrLimits::default();
        assert_eq!(limits.input_queue, Duration::from_secs(2));
        assert_eq!(limits.max_history_bytes, 8 * 1024 * 1024);
        let geometry = limits
            .geometry(SampleRate::HZ_48000, SampleRate::HZ_16000)
            .unwrap();
        assert_eq!(geometry.capacity, 96_000);
        assert_eq!(geometry.block, 1_600);
    }

    #[test]
    fn a_short_input_queue_shortens_the_feed_block() {
        let rate = SampleRate::HZ_16000;
        let limits = AsrLimits::default().with_input_queue(Duration::from_millis(50));
        let geometry = limits.geometry(rate, rate).unwrap();
        assert_eq!(geometry.capacity, 800);
        assert_eq!(geometry.block, 800, "never more than the queue holds");
    }

    #[test]
    fn invalid_limits_are_rejected() {
        let rate = SampleRate::HZ_16000;
        let cases = [
            AsrLimits::default().with_input_queue(Duration::ZERO),
            AsrLimits::default().with_max_history_bytes(0),
            AsrLimits::default().with_input_queue(Duration::from_secs(10_000)),
        ];
        for limits in cases {
            assert!(limits.geometry(rate, rate).is_err(), "{limits:?}");
        }
    }
}
