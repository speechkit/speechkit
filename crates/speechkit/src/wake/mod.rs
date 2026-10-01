//! Wake-word detection: spotting fixed phrases, such as "hey kit", in a
//! stream of audio.
//!
//! A [`WakeWordModel`] makes one [`WakeWordDetector`] per stream, as a
//! [`VadModel`](crate::vad::VadModel) does for voice activity. The
//! detector reports a [`WakeEvent`] each time it hears a keyword; what to
//! do next, such as starting an ASR session, is up to the caller.

use std::time::Duration;

use crate::{SampleRate, SpeechError};

/// A keyword heard in the stream.
///
/// Times count the audio fed to this detector, not wall-clock time: a
/// caller that stops feeding the detector, for example while an ASR
/// session runs, must add the audio it skipped.
#[derive(Debug, Clone, PartialEq)]
pub struct WakeEvent {
    /// The keyword, as the detector names it.
    pub keyword: String,
    /// Where the keyword starts, measured from the start of the stream.
    pub start: Duration,
    /// Where the keyword ends, as closely as the detector can tell. A
    /// detector reports the event some time after this, once it has heard
    /// enough to decide.
    pub end: Duration,
}

/// A wake-word detector for one stream.
pub trait WakeWordDetector: Send {
    /// Feeds audio, at the model's sample rate, and returns the keywords
    /// detected so far. A detector reports each utterance of a keyword
    /// once.
    fn accept(&mut self, samples: &[f32]) -> Vec<WakeEvent>;

    /// Ends the stream and returns keywords still being decided.
    fn flush(&mut self) -> Vec<WakeEvent>;
}

/// Creates one [`WakeWordDetector`] per stream.
pub trait WakeWordModel: Send + Sync + 'static {
    /// The rate the detector expects.
    fn sample_rate(&self) -> SampleRate;

    /// A fresh detector.
    ///
    /// # Errors
    ///
    /// Any error creating the detector.
    fn create(&self) -> Result<Box<dyn WakeWordDetector>, SpeechError>;
}
