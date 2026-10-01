//! A simple energy-threshold VAD.

use std::time::Duration;

use super::{SpeechSegment, Vad, VadModel};
use crate::{SampleRate, SpeechError};

/// Root-mean-square level of `samples`. Empty input is 0.0.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an RMS of f32 samples fits in f32"
    )]
    let level = (sum / samples.len() as f64).sqrt() as f32;
    level
}

/// Settings for [`EnergyVad`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EnergyVadConfig {
    /// The rate of the audio. Default: 16 kHz.
    pub sample_rate: SampleRate,
    /// RMS level above which a frame counts as speech. Default: 0.02.
    pub threshold: f32,
    /// Analysis frame length. Default: 30 ms.
    pub frame: Duration,
    /// Silence that ends a segment. Default: 300 ms.
    pub min_silence: Duration,
    /// Shorter segments are dropped. Default: 100 ms.
    pub min_speech: Duration,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            sample_rate: SampleRate::HZ_16000,
            threshold: 0.02,
            frame: Duration::from_millis(30),
            min_silence: Duration::from_millis(300),
            min_speech: Duration::from_millis(100),
        }
    }
}

impl EnergyVadConfig {
    /// Sets the sample rate.
    #[must_use]
    pub const fn with_sample_rate(mut self, rate: SampleRate) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Sets the threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the frame length.
    #[must_use]
    pub const fn with_frame(mut self, frame: Duration) -> Self {
        self.frame = frame;
        self
    }

    /// Sets the silence that ends a segment.
    #[must_use]
    pub const fn with_min_silence(mut self, min_silence: Duration) -> Self {
        self.min_silence = min_silence;
        self
    }

    /// Sets the shortest segment kept.
    #[must_use]
    pub const fn with_min_speech(mut self, min_speech: Duration) -> Self {
        self.min_speech = min_speech;
        self
    }
}

/// A VAD model that marks frames louder than a threshold as speech.
///
/// It needs no model file and no native code, which makes it useful for
/// tests and for trying speechkit out. It cannot tell speech from other
/// noise, so it is **not suitable for production**; use silero VAD from
/// `speechkit::sherpa` instead.
///
/// Speech starts at the first loud frame, and a detector reports it one
/// frame later at most, so [`start_delay`](VadModel::start_delay) is one
/// frame. A loud run shorter than `min_speech` is dropped after it was
/// reported as started.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    config: EnergyVadConfig,
}

fn frames(rate: SampleRate, duration: Duration) -> usize {
    usize::try_from(rate.frames_in(duration)).unwrap_or(usize::MAX)
}

impl EnergyVad {
    /// A model with `config`.
    pub fn new(config: EnergyVadConfig) -> Self {
        Self { config }
    }
}

impl VadModel for EnergyVad {
    fn sample_rate(&self) -> SampleRate {
        self.config.sample_rate
    }

    fn start_delay(&self) -> Duration {
        self.config.frame
    }

    fn create(&self, max_speech: Duration) -> Result<Box<dyn Vad>, SpeechError> {
        let config = self.config;
        let rate = config.sample_rate;
        Ok(Box::new(Detector {
            config,
            frame: frames(rate, config.frame).max(1),
            min_silence: frames(rate, config.min_silence).max(1),
            min_speech: frames(rate, config.min_speech).max(1),
            max_speech: frames(rate, max_speech).max(1),
            pending: Vec::new(),
            position: 0,
            speech: None,
            quiet: 0,
        }))
    }
}

/// Speech in progress.
#[derive(Debug)]
struct Speech {
    /// Its first frame.
    start: u64,
    samples: Vec<f32>,
    /// It continues speech that was cut, so it is kept however short.
    continued: bool,
}

/// One stream's detector.
#[derive(Debug)]
struct Detector {
    config: EnergyVadConfig,
    frame: usize,
    min_silence: usize,
    min_speech: usize,
    max_speech: usize,
    /// Samples not yet forming a whole frame.
    pending: Vec<f32>,
    /// Frames seen so far.
    position: u64,
    speech: Option<Speech>,
    /// Trailing quiet samples in the current speech.
    quiet: usize,
}

impl Detector {
    fn frame(&mut self, frame: &[f32], out: &mut Vec<SpeechSegment>) {
        let loud = rms(frame) >= self.config.threshold;
        let start = self.position;
        self.position += frame.len() as u64;
        match &mut self.speech {
            None if loud => {
                self.speech = Some(Speech {
                    start,
                    samples: frame.to_vec(),
                    continued: false,
                });
                self.quiet = 0;
            }
            None => {}
            Some(speech) => {
                speech.samples.extend_from_slice(frame);
                self.quiet = if loud { 0 } else { self.quiet + frame.len() };
                if self.quiet >= self.min_silence {
                    self.end(out);
                } else if speech.samples.len() >= self.max_speech {
                    self.cut(out);
                }
            }
        }
    }

    /// Ends the speech in progress at a pause, dropping it if too short.
    fn end(&mut self, out: &mut Vec<SpeechSegment>) {
        if let Some(Speech {
            start,
            mut samples,
            continued,
        }) = self.speech.take()
        {
            samples.truncate(samples.len() - self.quiet.min(samples.len()));
            let long_enough = if continued {
                !samples.is_empty()
            } else {
                samples.len() >= self.min_speech
            };
            if long_enough {
                out.push(SpeechSegment {
                    start: self.config.sample_rate.duration_of(start),
                    samples,
                });
            }
        }
        self.quiet = 0;
    }

    /// Cuts speech that reached `max_speech`; the rest of it starts here.
    fn cut(&mut self, out: &mut Vec<SpeechSegment>) {
        if let Some(Speech { start, samples, .. }) = self.speech.take() {
            out.push(SpeechSegment {
                start: self.config.sample_rate.duration_of(start),
                samples,
            });
        }
        self.speech = Some(Speech {
            start: self.position,
            samples: Vec::new(),
            continued: true,
        });
    }
}

impl Vad for Detector {
    fn accept(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        let mut out = Vec::new();
        self.pending.extend_from_slice(samples);
        let whole = self.pending.len() / self.frame * self.frame;
        let ready: Vec<f32> = self.pending.drain(..whole).collect();
        for frame in ready.chunks(self.frame) {
            self.frame(frame, &mut out);
        }
        out
    }

    fn flush(&mut self) -> Vec<SpeechSegment> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        if !rest.is_empty() {
            self.frame(&rest, &mut out);
        }
        self.end(&mut out);
        out
    }

    fn speaking_since(&self) -> Option<Duration> {
        self.speech
            .as_ref()
            .map(|speech| self.config.sample_rate.duration_of(speech.start))
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::*;

    const MAX: Duration = Duration::from_secs(15);

    fn tone(frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| 0.5 * (TAU * 440.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    fn detector(config: EnergyVadConfig, max_speech: Duration) -> Box<dyn Vad> {
        EnergyVad::new(config).create(max_speech).unwrap()
    }

    #[test]
    fn tone_silence_tone_gives_two_segments() {
        let mut signal = tone(16_000);
        signal.extend(vec![0.0; 16_000]);
        signal.extend(tone(16_000));
        let mut vad = detector(EnergyVadConfig::default(), MAX);
        let mut segments = Vec::new();
        for chunk in signal.chunks(1_000) {
            segments.extend(vad.accept(chunk));
        }
        segments.extend(vad.flush());
        assert_eq!(segments.len(), 2, "{segments:?}");
        let frame = Duration::from_millis(30);
        assert!(segments[0].start < frame);
        assert!(segments[1].start.abs_diff(Duration::from_secs(2)) <= frame);
        for segment in &segments {
            assert!(
                segment.samples.len().abs_diff(16_000) <= 480,
                "{}",
                segment.samples.len()
            );
        }
    }

    #[test]
    fn silence_and_blips_give_nothing() {
        let mut vad = detector(EnergyVadConfig::default(), MAX);
        // 33 whole frames of 30 ms.
        assert!(vad.accept(&vec![0.0; 15_840]).is_empty());
        assert_eq!(vad.speaking_since(), None);
        let mut blip = vec![0.0; 16_000];
        blip[100..300].copy_from_slice(&tone(200));
        assert!(vad.accept(&blip[..960]).is_empty());
        // The blip started speech, which is dropped once it ends.
        assert_eq!(vad.speaking_since(), Some(Duration::from_millis(990)));
        assert!(vad.accept(&blip[960..]).is_empty());
        assert_eq!(vad.speaking_since(), None);
        assert!(vad.flush().is_empty());
    }

    #[test]
    fn long_speech_is_cut_and_goes_on() {
        let mut vad = detector(EnergyVadConfig::default(), Duration::from_secs(1));
        let mut segments = vad.accept(&tone(40_000));
        let since = vad.speaking_since().unwrap();
        let last = segments.last().unwrap();
        // The rest of the speech starts at the cut.
        assert_eq!(
            since,
            last.start + SampleRate::HZ_16000.duration_of(last.samples.len() as u64)
        );
        segments.extend(vad.flush());
        assert_eq!(segments.len(), 3, "{segments:?}");
        assert!(segments.iter().all(|s| s.samples.len() <= 16_000 + 480));
        let total: usize = segments.iter().map(|s| s.samples.len()).sum();
        assert!(total.abs_diff(40_000) < 480, "{total}");
    }

    #[test]
    fn speech_running_into_the_end_is_flushed() {
        let model = EnergyVad::new(EnergyVadConfig::default().with_threshold(0.1));
        assert_eq!(model.sample_rate(), SampleRate::HZ_16000);
        assert_eq!(model.start_delay(), Duration::from_millis(30));
        let mut vad = model.create(MAX).unwrap();
        let mut signal = vec![0.0; 8_000];
        signal.extend(tone(8_010));
        assert!(vad.accept(&signal).is_empty());
        let since = vad.speaking_since().unwrap();
        assert!(since.abs_diff(Duration::from_millis(500)) <= Duration::from_millis(30));
        let segments = vad.flush();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, since);
        assert_eq!(vad.speaking_since(), None);
    }
}
