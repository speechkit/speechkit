//! Voices, updates, summaries, and failures.

use std::{fmt, ops::Range, time::Duration};

use crate::SpeechError;

/// A voice a backend offers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Voice {
    /// The ID passed in [`TtsOptions::voice`](super::TtsOptions::voice).
    pub id: String,
    /// A readable name.
    pub name: String,
    /// Language codes it speaks, such as `en` or `zh`. Empty means any.
    pub languages: Vec<String>,
}

impl Voice {
    /// A voice named after its ID, for any language.
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            languages: Vec::new(),
        }
    }

    /// Sets the languages.
    #[must_use]
    pub fn with_languages(
        mut self,
        languages: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.languages = languages.into_iter().map(Into::into).collect();
        self
    }

    /// Whether the voice speaks `language`: one with no languages listed
    /// speaks any.
    pub(crate) fn speaks(&self, language: &str) -> bool {
        self.languages.is_empty() || self.languages.iter().any(|l| l == language)
    }
}

/// A piece of the pushed text, and the audio that speaks it.
///
/// The output carries a mark right after the last audio of its text
/// (T-03), so a player that has played up to `audio.end` has played the
/// text in `text`.
///
/// Times are counted in samples of the output, at its rate
/// ([`TtsOutput::sample_rate`](super::TtsOutput::sample_rate)), so they
/// hold when the output is resampled: the mark is sent only after the
/// resampler has delivered all of its audio.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Mark {
    /// The byte range in the text pushed so far.
    pub text: Range<usize>,
    /// Where its audio lies in the output, from the first sample.
    pub audio: Range<Duration>,
}

/// One item from a [`TtsOutput`](super::TtsOutput).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TtsUpdate {
    /// Mono audio at the output's sample rate, in text order.
    Audio(Vec<f32>),
    /// A piece of text is fully synthesized: its audio came before this.
    Mark(Mark),
    /// The synthesis ended; always the last item.
    Closed(TtsResult),
}

/// A finished synthesis.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TtsSummary {
    /// The audio made, at the output rate.
    pub duration: Duration,
    /// Every mark, in text order.
    pub marks: Vec<Mark>,
}

/// A failed synthesis, with the progress made before it failed (T-09).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TtsFailure {
    /// What went wrong.
    pub error: SpeechError,
    /// Bytes of the pushed text fully synthesized before the failure.
    pub text_done: usize,
    /// The audio made before the failure, at the output rate.
    pub duration: Duration,
}

impl TtsFailure {
    /// A failure with no progress.
    pub(crate) fn new(error: SpeechError) -> Self {
        Self {
            error,
            text_done: 0,
            duration: Duration::ZERO,
        }
    }
}

/// For `?`: the error, without the progress.
impl From<TtsFailure> for SpeechError {
    fn from(failure: TtsFailure) -> Self {
        failure.error
    }
}

impl fmt::Display for TtsFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "synthesis failed: {}", self.error)
    }
}

impl std::error::Error for TtsFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// How a synthesis ended.
pub type TtsResult = Result<TtsSummary, TtsFailure>;
