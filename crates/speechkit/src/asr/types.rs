//! Transcript and session result types.

use std::{collections::BTreeMap, fmt, time::Duration};

use crate::SpeechError;

/// Identifies one utterance within a session. IDs increase monotonically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UtteranceId(pub u64);

impl fmt::Display for UtteranceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Text recognized so far for an utterance that is not finished yet.
/// A later partial or segment for the same utterance replaces it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Partial {
    /// The utterance this text belongs to.
    pub utterance: UtteranceId,
    /// The text so far.
    pub text: String,
}

/// The final text of one utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Segment {
    /// The utterance this text belongs to.
    pub utterance: UtteranceId,
    /// The final text.
    pub text: String,
    /// Where the utterance starts, measured from the start of the session.
    pub start: Duration,
    /// Where the utterance ends.
    pub end: Duration,
}

/// The committed segments of a session, in input order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Transcript {
    /// The segments, in input order.
    pub segments: Vec<Segment>,
    /// The length of audio transcribed.
    pub duration: Duration,
}

impl Transcript {
    /// A transcript of `segments`, which should be in input order, over
    /// `duration` of audio.
    pub fn new(segments: Vec<Segment>, duration: Duration) -> Self {
        Self { segments, duration }
    }

    /// The full text. Segments are trimmed, and empty ones are skipped.
    /// A space goes only between Latin text: when the previous segment ends
    /// in an ASCII letter, digit, or one of `.!?;:,`, and the next one
    /// starts with an ASCII letter or digit. Chinese, Japanese, and Korean
    /// text is joined without spaces.
    pub fn text(&self) -> String {
        join_texts(self.segments.iter().map(|s| s.text.as_str()))
    }
}

pub(crate) fn join_texts<'a>(texts: impl Iterator<Item = &'a str>) -> String {
    let mut out = String::new();
    for text in texts.map(str::trim).filter(|t| !t.is_empty()) {
        let space = !out.is_empty()
            && out
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || ".!?;:,".contains(c))
            && text.starts_with(|c: char| c.is_ascii_alphanumeric());
        if space {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

/// One speaker's turn: every segment of it, and no segment of another turn.
/// See [`AsrOptions::with_turn_end`](super::AsrOptions::with_turn_end).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Turn {
    /// The turn's segments, in order. Segments with no words are left out,
    /// so a turn that was only a cough has none.
    pub segments: Vec<Segment>,
    /// Where the turn's first speech started.
    pub start: Duration,
    /// Where its last speech ended.
    pub end: Duration,
}

impl Turn {
    /// The turn's text, joined as [`Transcript::text`] joins segments.
    pub fn text(&self) -> String {
        join_texts(self.segments.iter().map(|s| s.text.as_str()))
    }
}

/// Committed segments plus the pending text of unfinished utterances.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct View {
    /// Committed segments, in input order.
    committed: Vec<Segment>,
    /// Pending text by utterance.
    partials: BTreeMap<UtteranceId, String>,
}

/// The text of a session as it is recognized: its committed segments
/// followed by the pending text of unfinished utterances, for showing words
/// as they arrive.
///
/// [`apply`](Self::apply) every [`AsrUpdate`] a reader receives, and show
/// [`text`](Self::text) after each one. Every segment arrives once and in
/// order, even to a reader that fell behind, so no other bookkeeping is
/// needed. Speech and turn events change nothing. `Closed` drops the
/// pending text, which the session abandons, so the text then matches the
/// session's transcript.
///
/// ```
/// # use std::time::Duration;
/// use speechkit::asr::{AsrUpdate, LiveTranscript, Partial, Segment, UtteranceId};
///
/// let mut live = LiveTranscript::new();
/// live.apply(&AsrUpdate::Segment(Segment {
///     utterance: UtteranceId(1),
///     text: "Hello.".into(),
///     start: Duration::ZERO,
///     end: Duration::from_secs(1),
/// }));
/// live.apply(&AsrUpdate::Partial(Partial {
///     utterance: UtteranceId(2),
///     text: "how are".into(),
/// }));
/// assert_eq!(live.text(), "Hello. how are");
/// ```
#[derive(Debug, Clone, Default)]
pub struct LiveTranscript {
    view: View,
}

impl LiveTranscript {
    /// An empty transcript, before any update.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies an update: a `Partial` sets its utterance's pending text, a
    /// `Segment` commits it, and `Closed` drops what is still pending.
    pub fn apply(&mut self, update: &AsrUpdate) {
        let view = &mut self.view;
        match update {
            AsrUpdate::Partial(partial) => {
                view.partials
                    .insert(partial.utterance, partial.text.clone());
            }
            AsrUpdate::Segment(segment) => {
                view.partials.remove(&segment.utterance);
                view.committed.push(segment.clone());
            }
            AsrUpdate::Closed(_) => view.partials.clear(),
            AsrUpdate::SpeechStarted { .. }
            | AsrUpdate::SpeechEnded { .. }
            | AsrUpdate::TurnEnded(_) => {}
        }
    }

    /// The committed text followed by the pending text, joined as
    /// [`Transcript::text`] joins segments.
    pub fn text(&self) -> String {
        let committed = self.view.committed.iter().map(|s| s.text.as_str());
        let pending = self.view.partials.values().map(String::as_str);
        join_texts(committed.chain(pending))
    }
}

/// One item from an [`AsrUpdates`](super::AsrUpdates) reader.
///
/// Every reader receives every segment since the session started, and
/// every speech and turn event since it started reading, exactly once and
/// in order.
/// A `Partial` may be replaced by a later one before it is read. The text
/// still pending when the session ends is abandoned.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AsrUpdate {
    /// New pending text for an utterance, replacing its earlier pending
    /// text.
    Partial(Partial),
    /// The final text of one utterance. It replaces the utterance's
    /// pending text.
    Segment(Segment),
    /// Speech started here.
    SpeechStarted {
        /// Where, from the session's first sample.
        at: Duration,
    },
    /// Speech ended here, sent once the backend is sure. `utterance` is the
    /// last utterance of that speech, whose segment may still be to come.
    SpeechEnded {
        /// Where, from the session's first sample.
        at: Duration,
        /// The last utterance of the speech.
        utterance: UtteranceId,
    },
    /// A turn ended ([`AsrOptions::with_turn_end`](super::AsrOptions::with_turn_end)).
    /// It carries every segment of the turn, including text the backend
    /// finished after the turn's speech ended, so it can arrive after newer
    /// speech started: compare the latest `SpeechStarted` with
    /// [`Turn::end`] to tell.
    TurnEnded(Turn),
    /// The session ended with this result. Always the last update.
    Closed(AsrResult),
}

/// A failed session, with what it confirmed before it failed (A-04).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AsrFailure {
    /// What went wrong.
    pub error: SpeechError,
    /// The segments committed before the failure, and the audio
    /// transcribed.
    pub confirmed: Transcript,
}

impl AsrFailure {
    /// A failure with no progress, for errors raised before a session runs.
    pub(crate) fn new(error: SpeechError) -> Self {
        Self {
            error,
            confirmed: Transcript::default(),
        }
    }
}

/// For `?`: the error, without the progress.
impl From<AsrFailure> for SpeechError {
    fn from(failure: AsrFailure) -> Self {
        failure.error
    }
}

impl fmt::Display for AsrFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "session failed: {}", self.error)
    }
}

impl std::error::Error for AsrFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// How a session ended.
pub type AsrResult = Result<Transcript, AsrFailure>;

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(texts: &[&str]) -> Transcript {
        Transcript {
            duration: Duration::ZERO,
            segments: texts
                .iter()
                .enumerate()
                .map(|(i, text)| Segment {
                    utterance: UtteranceId(i as u64),
                    text: (*text).to_owned(),
                    start: Duration::ZERO,
                    end: Duration::ZERO,
                })
                .collect(),
        }
    }

    #[test]
    fn chinese_joins_without_spaces() {
        assert_eq!(transcript(&["你好", "世界"]).text(), "你好世界");
        assert_eq!(transcript(&["你好。", "再见"]).text(), "你好。再见");
    }

    #[test]
    fn english_joins_with_spaces() {
        assert_eq!(transcript(&["hello", "world"]).text(), "hello world");
        assert_eq!(
            transcript(&["Hi.", "How are you?"]).text(),
            "Hi. How are you?"
        );
        assert_eq!(transcript(&[" padded ", "", "  "]).text(), "padded");
    }

    #[test]
    fn mixed_text_spaces_only_between_latin() {
        assert_eq!(
            transcript(&["我用", "Rust", "写代码"]).text(),
            "我用Rust写代码"
        );
        assert_eq!(transcript(&["version", "2"]).text(), "version 2");
        assert_eq!(transcript(&["OK", "好的"]).text(), "OK好的");
        assert_eq!(transcript(&["好的，", "OK"]).text(), "好的，OK");
    }

    #[test]
    fn cjk_joins_without_spaces() {
        assert_eq!(transcript(&["你好", "world"]).text(), "你好world");
        assert_eq!(transcript(&["Hi.", "你好"]).text(), "Hi.你好");
    }

    #[test]
    fn live_transcript_applies_updates() {
        let mut live = LiveTranscript::new();
        let partial = |utterance, text: &str| {
            AsrUpdate::Partial(Partial {
                utterance: UtteranceId(utterance),
                text: text.into(),
            })
        };
        live.apply(&partial(1, "hel"));
        live.apply(&partial(1, "hello"));
        assert_eq!(live.text(), "hello");
        live.apply(&AsrUpdate::Segment(Segment {
            utterance: UtteranceId(1),
            text: "Hello.".into(),
            start: Duration::ZERO,
            end: Duration::from_secs(1),
        }));
        assert_eq!(live.text(), "Hello.");
        live.apply(&partial(2, "bye"));
        assert_eq!(live.text(), "Hello. bye");
        live.apply(&AsrUpdate::Closed(Ok(Transcript::default())));
        assert_eq!(live.text(), "Hello.", "pending text is abandoned");
    }

    #[test]
    fn failure_clones_and_displays() {
        let failure = AsrFailure::new(SpeechError::backend("x", true, "boom"));
        let copy = failure.clone();
        assert!(copy.error.retryable());
        assert_eq!(copy.to_string(), "session failed: backend `x` failed");
    }
}
