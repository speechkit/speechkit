//! Audio value types.
//!
//! Audio is always mono inside speechkit. Downmixing happens at the edges:
//! file decoding and microphone capture.

use std::{fmt, num::NonZeroU32, time::Duration};

use super::SpeechError;

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// A sample rate in hertz, between 8 kHz and 192 kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "u32", into = "u32"))]
pub struct SampleRate(NonZeroU32);

impl SampleRate {
    /// The lowest supported rate.
    const MIN_HZ: u32 = 8_000;
    /// The highest supported rate.
    const MAX_HZ: u32 = 192_000;

    /// 8 kHz, used by telephony.
    pub const HZ_8000: Self = Self::from_const(8_000);
    /// 16 kHz, the rate most ASR models expect.
    pub const HZ_16000: Self = Self::from_const(16_000);
    /// 22.05 kHz, common for TTS models.
    pub const HZ_22050: Self = Self::from_const(22_050);
    /// 24 kHz, used by OpenAI Realtime and many TTS models.
    pub const HZ_24000: Self = Self::from_const(24_000);
    /// 44.1 kHz, CD audio.
    pub const HZ_44100: Self = Self::from_const(44_100);
    /// 48 kHz, the usual device rate.
    pub const HZ_48000: Self = Self::from_const(48_000);

    const fn from_const(hz: u32) -> Self {
        match NonZeroU32::new(hz) {
            Some(hz) => Self(hz),
            None => Self(NonZeroU32::MIN),
        }
    }

    /// Creates a rate, rejecting values outside 8 kHz to 192 kHz.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] if `hz` is out of range.
    pub fn new(hz: u32) -> Result<Self, SpeechError> {
        match NonZeroU32::new(hz) {
            Some(nz) if (Self::MIN_HZ..=Self::MAX_HZ).contains(&hz) => Ok(Self(nz)),
            _ => Err(SpeechError::InvalidInput(format!(
                "sample rate {hz} Hz is outside {}..={} Hz",
                Self::MIN_HZ,
                Self::MAX_HZ
            ))),
        }
    }

    /// The rate in hertz.
    pub const fn hz(self) -> u32 {
        self.0.get()
    }

    /// The number of whole frames in `duration` at this rate, rounded down.
    pub fn frames_in(self, duration: Duration) -> u64 {
        let frames = duration.as_nanos() * u128::from(self.hz()) / NANOS_PER_SECOND;
        u64::try_from(frames).unwrap_or(u64::MAX)
    }

    /// The duration of `frames` frames at this rate.
    pub fn duration_of(self, frames: u64) -> Duration {
        let nanos = u128::from(frames) * NANOS_PER_SECOND / u128::from(self.hz());
        Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
    }
}

impl TryFrom<u32> for SampleRate {
    type Error = SpeechError;

    fn try_from(hz: u32) -> Result<Self, Self::Error> {
        Self::new(hz)
    }
}

impl From<SampleRate> for u32 {
    fn from(rate: SampleRate) -> Self {
        rate.hz()
    }
}

impl fmt::Display for SampleRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz", self.hz())
    }
}

/// Why a sample was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SampleErrorKind {
    /// The sample is NaN.
    NaN,
    /// The sample is positive or negative infinity.
    Infinite,
    /// The sample is finite but outside [-1.0, 1.0].
    OutOfRange,
}

impl fmt::Display for SampleErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NaN => "NaN",
            Self::Infinite => "infinite",
            Self::OutOfRange => "outside [-1.0, 1.0]",
        })
    }
}

/// The first invalid sample found in a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("sample {index} is {kind}")]
pub(crate) struct InvalidSample {
    /// Position of the sample in the slice.
    pub(crate) index: usize,
    /// What is wrong with it.
    pub(crate) kind: SampleErrorKind,
}

impl From<InvalidSample> for SpeechError {
    fn from(error: InvalidSample) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

/// Returns the first sample that is not finite or lies outside [-1.0, 1.0].
pub(crate) fn find_invalid(samples: &[f32]) -> Option<InvalidSample> {
    samples.iter().enumerate().find_map(|(index, &sample)| {
        let kind = if sample.is_nan() {
            SampleErrorKind::NaN
        } else if sample.is_infinite() {
            SampleErrorKind::Infinite
        } else if !(-1.0..=1.0).contains(&sample) {
            SampleErrorKind::OutOfRange
        } else {
            return None;
        };
        Some(InvalidSample { index, kind })
    })
}

/// A complete mono recording held in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    /// The rate of `samples`.
    pub sample_rate: SampleRate,
    /// Mono samples, normally in [-1.0, 1.0].
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    /// A buffer holding `samples` at `sample_rate`.
    pub const fn new(sample_rate: SampleRate, samples: Vec<f32>) -> Self {
        Self {
            sample_rate,
            samples,
        }
    }

    /// How long the recording lasts.
    pub fn duration(&self) -> Duration {
        self.sample_rate
            .duration_of(u64::try_from(self.samples.len()).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn rates_are_range_checked() {
        assert_eq!(SampleRate::new(16_000).unwrap(), SampleRate::HZ_16000);
        assert!(SampleRate::new(0).is_err());
        assert!(SampleRate::new(7_999).is_err());
        assert!(SampleRate::new(192_001).is_err());
        assert_eq!(SampleRate::HZ_22050.hz(), 22_050);
    }

    #[test]
    fn frames_and_durations_convert() {
        let rate = SampleRate::HZ_16000;
        assert_eq!(rate.frames_in(Duration::from_secs(2)), 32_000);
        assert_eq!(rate.frames_in(Duration::from_micros(62)), 0);
        assert_eq!(rate.duration_of(8_000), Duration::from_millis(500));
        let buffer = AudioBuffer::new(rate, vec![0.0; 24_000]);
        assert_eq!(buffer.duration(), Duration::from_millis(1_500));
    }

    #[test]
    fn invalid_sample_messages() {
        let error = find_invalid(&[0.0, f32::NAN]).unwrap();
        assert_eq!(error.to_string(), "sample 1 is NaN");
        let error: SpeechError = find_invalid(&[f32::NEG_INFINITY]).unwrap().into();
        assert_eq!(error.to_string(), "invalid input: sample 0 is infinite");
        let error = find_invalid(&[0.5, 0.25, 1.5]).unwrap();
        assert_eq!(error.to_string(), "sample 2 is outside [-1.0, 1.0]");
    }

    #[test]
    fn empty_slice_is_valid() {
        assert_eq!(find_invalid(&[]), None);
    }

    fn bad_value() -> impl Strategy<Value = (f32, SampleErrorKind)> {
        prop_oneof![
            Just((f32::NAN, SampleErrorKind::NaN)),
            Just((f32::INFINITY, SampleErrorKind::Infinite)),
            Just((f32::NEG_INFINITY, SampleErrorKind::Infinite)),
            (1.000_001_f32..1e30).prop_map(|v| (v, SampleErrorKind::OutOfRange)),
            (-1e30_f32..-1.000_001).prop_map(|v| (v, SampleErrorKind::OutOfRange)),
        ]
    }

    proptest! {
        #[test]
        fn valid_slices_are_accepted(samples in prop::collection::vec(-1.0_f32..=1.0, 0..512)) {
            prop_assert_eq!(find_invalid(&samples), None);
        }

        #[test]
        fn bad_sample_is_found_at_its_index(
            mut samples in prop::collection::vec(-1.0_f32..=1.0, 1..512),
            position in any::<prop::sample::Index>(),
            (value, kind) in bad_value(),
        ) {
            let index = position.index(samples.len());
            samples[index] = value;
            prop_assert_eq!(find_invalid(&samples), Some(InvalidSample { index, kind }));
        }
    }
}
