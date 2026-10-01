//! WAV reading and writing with hound.

use std::io::{Cursor, Read};

use super::{DecodeLimits, invalid, to_session_range};
use crate::{AudioBuffer, SampleRate, SpeechError};

/// Why hound couldn't read a WAV.
pub(super) enum Error {
    /// The file is broken, too long, or unreadable.
    Invalid(SpeechError),
    /// A valid WAV in an encoding hound doesn't read, such as A-law.
    Unsupported(
        #[cfg_attr(
            feature = "decode",
            expect(dead_code, reason = "with `decode`, symphonia reports its own error")
        )]
        String,
    ),
}

impl From<SpeechError> for Error {
    fn from(error: SpeechError) -> Self {
        Self::Invalid(error)
    }
}

fn wav_error(error: hound::Error) -> Error {
    match error {
        hound::Error::Unsupported => Error::Unsupported("unsupported WAV encoding".into()),
        other => Error::Invalid(invalid(format_args!("WAV: {other}"))),
    }
}

/// Reads a WAV: integer PCM of any depth up to 32 bits, or 32-bit float,
/// with any number of channels, averaged to mono. A file that ends early
/// keeps the whole frames before the cut.
pub(super) fn read(source: impl Read, limits: DecodeLimits) -> Result<AudioBuffer, Error> {
    let mut reader = hound::WavReader::new(source).map_err(wav_error)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        return Err(invalid("WAV: the file has no channels").into());
    }
    let rate = SampleRate::new(spec.sample_rate)?;
    let max_frames = rate.frames_in(limits.max_duration);
    if u64::from(reader.duration()) > max_frames {
        return Err(invalid(format_args!(
            "the audio is longer than the {} s limit",
            limits.max_duration.as_secs()
        ))
        .into());
    }
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => mix(reader.samples::<f32>(), channels, 1.0),
        (hound::SampleFormat::Int, bits @ 1..=32) => {
            let scale = 2.0_f32.powi(1 - i32::from(bits));
            #[expect(
                clippy::cast_precision_loss,
                reason = "32-bit samples keep 24 bits of precision, more than audio needs"
            )]
            let samples = reader.samples::<i32>().map(|s| s.map(|v| v as f32));
            mix(samples, channels, scale)
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "unsupported WAV sample format: {}-bit {:?}",
                spec.bits_per_sample, spec.sample_format
            )));
        }
    }?;
    Ok(AudioBuffer::new(rate, samples))
}

/// Averages interleaved frames to mono, scaling each sample.
fn mix(
    samples: impl Iterator<Item = hound::Result<f32>>,
    channels: usize,
    scale: f32,
) -> Result<Vec<f32>, Error> {
    #[expect(clippy::cast_precision_loss, reason = "channel counts are tiny")]
    let scale = scale / channels as f32;
    let mut mono = Vec::new();
    let mut sum = 0.0_f32;
    let mut in_frame = 0;
    for sample in samples {
        match sample {
            Ok(sample) => sum += sample,
            // The header is read, so this is data that ends early: keep the
            // whole frames before the cut, as the other decoders do.
            Err(error) if !mono.is_empty() => {
                tracing::warn!(%error, "the WAV data ends early; keeping the audio before it");
                break;
            }
            Err(error) => return Err(wav_error(error)),
        }
        in_frame += 1;
        if in_frame == channels {
            mono.push(to_session_range(sum * scale));
            sum = 0.0;
            in_frame = 0;
        }
    }
    Ok(mono)
}

/// Encodes a buffer as a mono 16-bit PCM WAV file. Samples are clamped to
/// [-1.0, 1.0].
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] if a sample is not finite.
pub fn encode_wav(buffer: &AudioBuffer) -> Result<Vec<u8>, SpeechError> {
    encode_samples(buffer.sample_rate, &buffer.samples)
}

/// [`encode_wav`] for samples that are not in an [`AudioBuffer`], so a
/// caller holding a slice does not copy it.
///
/// # Errors
///
/// As [`encode_wav`].
pub(crate) fn encode_samples(
    sample_rate: SampleRate,
    samples: &[f32],
) -> Result<Vec<u8>, SpeechError> {
    let invalid = |message: &dyn std::fmt::Display| {
        SpeechError::InvalidInput(format!("cannot encode WAV: {message}"))
    };
    if let Some(bad) = samples.iter().position(|s| !s.is_finite()) {
        return Err(invalid(&format_args!("sample {bad} is not finite")));
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate.hz(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut bytes = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    let mut writer = hound::WavWriter::new(&mut bytes, spec).map_err(|e| invalid(&e))?;
    let frames = u32::try_from(samples.len()).map_err(|e| invalid(&e))?;
    let mut out = writer.get_i16_writer(frames);
    for &sample in samples {
        out.write_sample(f32_to_pcm16(sample));
    }
    out.flush().map_err(|e| invalid(&e))?;
    writer.finalize().map_err(|e| invalid(&e))?;
    Ok(bytes.into_inner())
}

/// Converts a sample to 16-bit PCM, clamping to [-1.0, 1.0]. NaN becomes 0.
fn f32_to_pcm16(sample: f32) -> i16 {
    let scaled = (sample.clamp(-1.0, 1.0) * 32_768.0).round();
    if scaled.is_nan() {
        0
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "clamped to the i16 range just above"
        )]
        let value = scaled.clamp(-32_768.0, 32_767.0) as i16;
        value
    }
}

/// Encodes samples as little-endian 16-bit PCM bytes.
#[cfg_attr(
    not(any(test, feature = "server", feature = "openai", feature = "dashscope")),
    expect(
        dead_code,
        reason = "only the server and the cloud backends send raw PCM"
    )
)]
pub(crate) fn pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| f32_to_pcm16(s).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::audio::decode;

    fn buffer(samples: Vec<f32>) -> AudioBuffer {
        AudioBuffer::new(SampleRate::HZ_16000, samples)
    }

    fn write<S: hound::Sample + Copy>(spec: hound::WavSpec, samples: &[S]) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut bytes, spec).unwrap();
        for &sample in samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        bytes.into_inner()
    }

    fn spec(channels: u16, bits: u16, format: hound::SampleFormat) -> hound::WavSpec {
        hound::WavSpec {
            channels,
            sample_rate: 16_000,
            bits_per_sample: bits,
            sample_format: format,
        }
    }

    #[test]
    fn pcm16_conversions() {
        assert_eq!(f32_to_pcm16(0.0), 0);
        assert_eq!(f32_to_pcm16(1.0), 32_767);
        assert_eq!(f32_to_pcm16(-1.0), -32_768);
        assert_eq!(f32_to_pcm16(2.0), 32_767);
        assert_eq!(f32_to_pcm16(f32::NAN), 0);
        assert_eq!(pcm16_bytes(&[0.5]), 16_384_i16.to_le_bytes());
    }

    #[test]
    fn rejects_non_finite_samples() {
        assert!(encode_wav(&buffer(vec![0.0, f32::INFINITY])).is_err());
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let bytes = write(
            spec(2, 16, hound::SampleFormat::Int),
            &[16_384_i16, 0, -16_384, -16_384],
        );
        let audio = decode(&bytes, DecodeLimits::default()).unwrap();
        assert_eq!(audio.samples, [0.25, -0.5]);
    }

    #[test]
    fn every_depth_scales_to_full_range() {
        let full = |bits: u16| {
            let max = i32::MAX >> (32 - bits);
            let bytes = write(spec(1, bits, hound::SampleFormat::Int), &[max, -max - 1]);
            decode(&bytes, DecodeLimits::default()).unwrap().samples
        };
        for bits in [8, 16, 24, 32] {
            let samples = full(bits);
            assert!((samples[0] - 1.0).abs() < 0.01, "{bits}: {samples:?}");
            assert!(
                (samples[1] + 1.0).abs() < f32::EPSILON,
                "{bits}: {samples:?}"
            );
        }
        let float = write(spec(1, 32, hound::SampleFormat::Float), &[0.5_f32, 2.0]);
        assert_eq!(
            decode(&float, DecodeLimits::default()).unwrap().samples,
            [0.5, 1.0]
        );
    }

    #[test]
    fn a_truncated_file_keeps_whole_frames() {
        let mut bytes = write(spec(2, 16, hound::SampleFormat::Int), &[0_i16; 8]);
        // Four stereo frames of 16 bits; the cut leaves three and a byte.
        bytes.truncate(bytes.len() - 3);
        let audio = decode(&bytes, DecodeLimits::default()).unwrap();
        assert_eq!(audio.samples.len(), 3);
    }

    #[test]
    fn the_duration_limit_uses_the_header() {
        let limits = DecodeLimits::new(std::time::Duration::from_secs(1));
        let long = write(spec(1, 16, hound::SampleFormat::Int), &vec![0_i16; 16_001]);
        assert!(decode(&long, limits).is_err());
        // The header's claim counts, even when the data ends early.
        assert!(decode(&long[..long.len() - 100], limits).is_err());
        let exact = write(spec(1, 16, hound::SampleFormat::Int), &vec![0_i16; 16_000]);
        assert_eq!(decode(&exact, limits).unwrap().samples.len(), 16_000);
    }

    #[test]
    fn reads_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        std::fs::write(&path, encode_wav(&buffer(vec![0.25; 10])).unwrap()).unwrap();
        let read = crate::audio::read(&path, DecodeLimits::default()).unwrap();
        assert_eq!(read.samples, vec![0.25; 10]);
        let error = crate::audio::read(dir.path().join("missing.wav"), DecodeLimits::default())
            .unwrap_err();
        assert!(error.to_string().contains("missing.wav"), "{error}");
    }

    proptest! {
        #[test]
        fn round_trip_within_one_step(samples in prop::collection::vec(-1.0_f32..=1.0, 0..2_000)) {
            let original = buffer(samples);
            let decoded = decode(&encode_wav(&original).unwrap(), DecodeLimits::default()).unwrap();
            prop_assert_eq!(decoded.sample_rate, original.sample_rate);
            prop_assert_eq!(decoded.samples.len(), original.samples.len());
            for (a, b) in original.samples.iter().zip(&decoded.samples) {
                prop_assert!((a - b).abs() <= 1.0 / 32_768.0 + f32::EPSILON);
            }
        }
    }
}
