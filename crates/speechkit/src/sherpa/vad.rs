//! Silero voice activity detection.

use std::{path::PathBuf, time::Duration};

use crate::{
    SampleRate, SpeechError,
    vad::{SpeechSegment, Vad, VadModel},
};
use sherpa_onnx::{VadModelConfig, VoiceActivityDetector};

use crate::sherpa::config::require_file;

/// Samples per silero window at 16 kHz.
const WINDOW: usize = 512;

/// The longest speech a detector can be asked to cut at.
pub(crate) const MAX_SPEECH: Duration = Duration::from_secs(300);

/// A window's length.
const WINDOW_TIME: Duration = Duration::from_millis(32);

/// Settings for silero VAD.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SileroVadConfig {
    /// The `silero_vad.onnx` file.
    pub model: PathBuf,
    /// Speech probability above which a window is speech. Default: 0.5.
    pub threshold: f32,
    /// Silence that ends a segment. Default: 0.5 s.
    pub min_silence: Duration,
    /// Speech is reported once it has lasted this long, and shorter
    /// speech is ignored. Default: 0.25 s.
    pub min_speech: Duration,
}

impl SileroVadConfig {
    /// Default settings for the model file `model`.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            threshold: 0.5,
            min_silence: Duration::from_millis(500),
            min_speech: Duration::from_millis(250),
        }
    }

    /// Sets the threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the silence that ends a segment.
    #[must_use]
    pub fn with_min_silence(mut self, min_silence: Duration) -> Self {
        self.min_silence = min_silence;
        self
    }

    /// Sets the shortest segment kept.
    #[must_use]
    pub fn with_min_speech(mut self, min_speech: Duration) -> Self {
        self.min_speech = min_speech;
        self
    }

    /// Checks the settings and the model file for [`load`](Self::load),
    /// without loading anything native.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] for a missing or empty model, or
    /// [`SpeechError::InvalidInput`] for settings out of range: the
    /// threshold must be in [0, 1), and durations positive and at most
    /// 60 s.
    pub fn validate(&self) -> Result<(), SpeechError> {
        require_file(&self.model, "VAD model")?;
        let durations = [self.min_silence, self.min_speech];
        let valid = (0.0..1.0).contains(&self.threshold)
            && durations
                .iter()
                .all(|d| !d.is_zero() && *d <= Duration::from_secs(60));
        if valid {
            Ok(())
        } else {
            Err(SpeechError::InvalidInput(
                "invalid VAD settings: the threshold must be in [0, 1), and durations positive \
                 and at most 60 s"
                    .into(),
            ))
        }
    }

    /// Checks that speech can be cut at `max_speech`: longer than
    /// `min_speech`, and at most [`MAX_SPEECH`].
    pub(crate) fn check_max_speech(&self, max_speech: Duration) -> Result<(), SpeechError> {
        if max_speech > self.min_speech && max_speech <= MAX_SPEECH {
            Ok(())
        } else {
            Err(SpeechError::InvalidInput(format!(
                "the longest utterance must be longer than the VAD's min_speech ({:?}) and at \
                 most {} s",
                self.min_speech,
                MAX_SPEECH.as_secs()
            )))
        }
    }

    fn native(&self, max_speech: Duration) -> VadModelConfig {
        let mut config = VadModelConfig::default();
        config.silero_vad.model = Some(self.model.to_string_lossy().into_owned());
        config.silero_vad.threshold = self.threshold;
        config.silero_vad.min_silence_duration = self.min_silence.as_secs_f32();
        config.silero_vad.min_speech_duration = self.min_speech.as_secs_f32();
        config.silero_vad.max_speech_duration = max_speech.as_secs_f32();
        config.silero_vad.window_size = 512;
        config.sample_rate = 16_000;
        config.num_threads = 1;
        config.provider = Some("cpu".into());
        config
    }

    fn buffer_seconds(&self, max_speech: Duration) -> f32 {
        (max_speech + self.min_silence).as_secs_f32() + 5.0
    }
}

/// Creates a silero detector per session. VAD always runs on the CPU.
/// [`SileroVadConfig::load`] makes one.
#[derive(Debug, Clone)]
pub struct SileroVad {
    config: SileroVadConfig,
}

impl SileroVadConfig {
    /// Checks the settings and loads the model once, so a broken model
    /// fails here rather than in every session.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] for a missing, empty, or unloadable
    /// model, or [`SpeechError::InvalidInput`] for settings out of range:
    /// the threshold must be in [0, 1), durations positive and at most
    /// 60 s.
    pub fn load(&self) -> Result<SileroVad, SpeechError> {
        self.validate()?;
        let vad = SileroVad {
            config: self.clone(),
        };
        drop(vad.detector(DEFAULT_MAX_SPEECH)?);
        Ok(vad)
    }
}

/// The cut `load` tries the model with.
const DEFAULT_MAX_SPEECH: Duration = Duration::from_secs(20);

impl SileroVad {
    fn detector(&self, max_speech: Duration) -> Result<VoiceActivityDetector, SpeechError> {
        let config = &self.config;
        VoiceActivityDetector::create(
            &config.native(max_speech),
            config.buffer_seconds(max_speech),
        )
        .ok_or_else(|| {
            SpeechError::InvalidModel(format!(
                "sherpa-onnx could not load the VAD model {}",
                self.config.model.display()
            ))
        })
    }
}

/// Speech is detected once it has lasted `min_speech`, and the detector
/// is fed whole 32 ms windows, so it reports a start up to `min_speech`
/// plus two windows late. Speech longer than `max_speech` ends at its next
/// short pause, which is how sherpa-onnx bounds it, so a segment can run
/// somewhat past it.
impl VadModel for SileroVad {
    fn sample_rate(&self) -> SampleRate {
        SampleRate::HZ_16000
    }

    fn start_delay(&self) -> Duration {
        self.config.min_speech + 2 * WINDOW_TIME
    }

    fn create(&self, max_speech: Duration) -> Result<Box<dyn Vad>, SpeechError> {
        self.config.check_max_speech(max_speech)?;
        Ok(Box::new(Silero {
            detector: self.detector(max_speech)?,
            pending: Vec::with_capacity(WINDOW),
            fed: 0,
            since: None,
            last_end: None,
            min_speech: SampleRate::HZ_16000.frames_in(self.config.min_speech),
        }))
    }
}

struct Silero {
    detector: VoiceActivityDetector,
    pending: Vec<f32>,
    /// Samples fed to the detector.
    fed: u64,
    /// Where the speech in progress started, estimated when it was
    /// detected.
    since: Option<u64>,
    /// Where the last segment ended.
    last_end: Option<u64>,
    /// `min_speech` in samples.
    min_speech: u64,
}

impl Silero {
    fn drain(&mut self, out: &mut Vec<SpeechSegment>) {
        while let Some(segment) = self.detector.front() {
            let start = u64::try_from(segment.start()).unwrap_or(0);
            self.last_end = Some(start + segment.samples().len() as u64);
            out.push(SpeechSegment {
                start: SampleRate::HZ_16000.duration_of(start),
                samples: segment.samples().to_vec(),
            });
            self.detector.pop();
        }
    }

    /// Notes whether speech is detected after a window. It is detected
    /// once it has lasted `min_speech`, which dates its start. sherpa-onnx
    /// never cuts speech that goes on, so new speech starts after the last
    /// segment.
    fn track(&mut self) {
        if !self.detector.detected() {
            self.since = None;
        } else if self.since.is_none() {
            let window = WINDOW as u64;
            let estimate = self.fed.saturating_sub(self.min_speech + window);
            self.since = Some(estimate.max(self.last_end.map_or(0, |end| end + 1)));
        }
    }
}

impl Vad for Silero {
    fn accept(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        self.pending.extend_from_slice(samples);
        let whole = self.pending.len() / WINDOW * WINDOW;
        let mut out = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        for window in pending[..whole].chunks(WINDOW) {
            self.detector.accept_waveform(window);
            self.fed += WINDOW as u64;
            self.drain(&mut out);
            self.track();
        }
        self.pending = pending;
        self.pending.drain(..whole);
        out
    }

    fn flush(&mut self) -> Vec<SpeechSegment> {
        if !self.pending.is_empty() {
            let rest = std::mem::take(&mut self.pending);
            self.detector.accept_waveform(&rest);
            self.fed += rest.len() as u64;
        }
        self.detector.flush();
        let mut out = Vec::new();
        self.drain(&mut out);
        self.since = None;
        out
    }

    fn speaking_since(&self) -> Option<Duration> {
        self.since
            .map(|since| SampleRate::HZ_16000.duration_of(since))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_checked_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("silero_vad.onnx");
        std::fs::write(&model, "not a model").unwrap();
        assert!(SileroVadConfig::new(&model).validate().is_ok());
        let bad = [
            SileroVadConfig::new(&model).with_threshold(1.0),
            SileroVadConfig::new(&model).with_min_silence(Duration::ZERO),
            SileroVadConfig::new(&model).with_min_speech(Duration::from_secs(61)),
        ];
        for config in bad {
            assert!(
                matches!(config.validate(), Err(SpeechError::InvalidInput(_))),
                "{config:?}"
            );
        }
        let config = SileroVadConfig::new(&model);
        assert!(config.check_max_speech(Duration::from_secs(20)).is_ok());
        for max in [Duration::from_millis(250), Duration::from_secs(301)] {
            assert!(config.check_max_speech(max).is_err(), "{max:?}");
        }
        let missing = SileroVadConfig::new(dir.path().join("missing.onnx"));
        assert!(matches!(
            missing.validate(),
            Err(SpeechError::InvalidModel(_))
        ));
    }
}
