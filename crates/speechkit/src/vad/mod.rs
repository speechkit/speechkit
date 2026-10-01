//! Voice activity detection, and the adapter that turns an offline
//! recognizer plus a VAD into a streaming [`AsrBackend`](crate::asr::AsrBackend).
//!
//! The adapter is written once here and reused by every "detect speech,
//! then recognize it" backend: SenseVoice, Paraformer, per-utterance HTTP
//! upload, and so on.

mod backend;
mod energy;
pub(crate) mod tracker;

use std::time::Duration;

pub use backend::VadBackend;
pub use energy::{EnergyVad, EnergyVadConfig};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrCapabilities, AsrOptions},
};

/// A stretch of detected speech.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    /// Where the speech starts, measured from the start of the stream.
    pub start: Duration,
    /// The speech, at the VAD's sample rate.
    pub samples: Vec<f32>,
}

/// A voice activity detector for one stream.
pub trait Vad: Send {
    /// Feeds audio and returns the speech segments that ended, in order.
    ///
    /// A segment should be no longer than the `max_speech` the detector
    /// was created with. Speech that runs longer is cut: the segment ends
    /// at the cut, and the rest of the speech starts there, so
    /// [`speaking_since`](Self::speaking_since) then returns the cut.
    fn accept(&mut self, samples: &[f32]) -> Vec<SpeechSegment>;

    /// Ends the stream and returns the speech still in progress.
    fn flush(&mut self) -> Vec<SpeechSegment>;

    /// Where the speech in progress started, measured from the start of
    /// the stream, or `None` between speech. Speech the detector later
    /// drops as too short counts too.
    fn speaking_since(&self) -> Option<Duration>;
}

/// A voice activity model, which creates one [`Vad`] per session.
pub trait VadModel: Send + Sync + 'static {
    /// The rate the VAD expects. It must equal the recognizer's
    /// `sample_rate`.
    fn sample_rate(&self) -> SampleRate;

    /// How late the detector reports the start of speech. Once it has
    /// been fed audio up to a point P, every speech that started before
    /// P minus this delay is reported, by
    /// [`Vad::speaking_since`] or by a segment.
    fn start_delay(&self) -> Duration;

    /// A fresh detector that cuts speech longer than `max_speech`.
    ///
    /// # Errors
    ///
    /// Any error creating the detector, such as a model that fails to load
    /// or a `max_speech` it cannot honor.
    fn create(&self, max_speech: Duration) -> Result<Box<dyn Vad>, SpeechError>;
}

/// A recognizer that transcribes one complete utterance at a time.
pub trait OfflineRecognizer: Send + Sync + 'static {
    /// A short name for logs and errors.
    fn name(&self) -> &str;

    /// What the recognizer supports. `reports_partials` and
    /// `reports_activity` are ignored: [`VadBackend`] never reports partial
    /// results, and its VAD reports activity.
    fn capabilities(&self) -> &AsrCapabilities;

    /// Transcribes one utterance, given at `sample_rate`.
    ///
    /// # Errors
    ///
    /// Any error the recognizer hits. The session fails with it.
    fn recognize(&self, samples: &[f32], opts: &AsrOptions) -> Result<String, SpeechError>;
}
