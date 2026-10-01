//! Synthesis options, capabilities, and limits.

use std::{ops::RangeInclusive, time::Duration};

use super::Voice;
use crate::{SampleRate, SpeechError};

/// What a synthesis backend supports.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TtsCapabilities {
    /// Audio for a chunk arrives before its synthesis finishes.
    pub streams_audio: bool,
    /// The supported speed range, or `None` if only 1.0 is supported.
    pub speed: Option<RangeInclusive<f32>>,
    /// The rate the backend produces.
    pub sample_rate: SampleRate,
    /// The longest text one backend call accepts, in characters.
    pub max_chunk_chars: usize,
}

impl TtsCapabilities {
    /// Capabilities with every flag off, at `sample_rate`, accepting
    /// `max_chunk_chars` per call.
    pub const fn new(sample_rate: SampleRate, max_chunk_chars: usize) -> Self {
        Self {
            streams_audio: false,
            speed: None,
            sample_rate,
            max_chunk_chars,
        }
    }
}

/// Options for one synthesis session, checked before synthesis starts
/// (T-04).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TtsOptions {
    /// The voice ID, or `None` for the backend's default.
    pub voice: Option<String>,
    /// The language, for voices that speak several.
    pub language: Option<String>,
    /// The speaking speed. Default: 1.0.
    pub speed: f32,
    /// Resample the output to this rate, or `None` for the backend's rate.
    pub sample_rate: Option<SampleRate>,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            voice: None,
            language: None,
            speed: 1.0,
            sample_rate: None,
        }
    }
}

impl TtsOptions {
    /// Sets the voice, one of the engine's `voices()`.
    #[must_use]
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Sets the language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the speed, within the backend's `capabilities().speed`.
    #[must_use]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Resamples the output to `rate`.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: SampleRate) -> Self {
        self.sample_rate = Some(rate);
        self
    }
}

/// Checks `opts` against a backend's capabilities and voices.
///
/// # Errors
///
/// - [`SpeechError::InvalidInput`] for an unknown voice or a speed that is
///   not finite and positive, or outside the supported range;
/// - [`SpeechError::Unsupported`] for a speed other than 1.0 on a backend
///   without speed control, or a language no candidate voice speaks.
pub(crate) fn validate(
    opts: &TtsOptions,
    caps: &TtsCapabilities,
    voices: &[Voice],
) -> Result<(), SpeechError> {
    let voice = match &opts.voice {
        Some(id) => Some(
            voices
                .iter()
                .find(|voice| &voice.id == id)
                .ok_or_else(|| SpeechError::InvalidInput(format!("unknown voice {id:?}")))?,
        ),
        None => None,
    };
    if !opts.speed.is_finite() || opts.speed <= 0.0 {
        return Err(SpeechError::InvalidInput(format!(
            "speed {} must be positive",
            opts.speed
        )));
    }
    match &caps.speed {
        None if (opts.speed - 1.0).abs() > f32::EPSILON => {
            return Err(SpeechError::Unsupported(
                "this backend does not change the speed".into(),
            ));
        }
        Some(range) if !range.contains(&opts.speed) => {
            return Err(SpeechError::InvalidInput(format!(
                "speed {} is outside {}..={}",
                opts.speed,
                range.start(),
                range.end()
            )));
        }
        _ => {}
    }
    if let Some(language) = &opts.language {
        let spoken = match voice {
            Some(voice) => voice.speaks(language),
            None => voices.is_empty() || voices.iter().any(|v| v.speaks(language)),
        };
        if !spoken {
            return Err(SpeechError::Unsupported(format!(
                "no voice speaks {language:?}"
            )));
        }
    }
    Ok(())
}

/// Per-session limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TtsLimits {
    /// How much audio the output queue holds. Default: 2 s.
    pub output_queue: Duration,
    /// The most text one session accepts, in characters. Default: 10 000.
    pub max_text_chars: usize,
    /// The longest chunk sent to the backend, in characters, and at most
    /// the backend's `max_chunk_chars`. Default: 300.
    pub chunk_chars: usize,
}

impl Default for TtsLimits {
    fn default() -> Self {
        Self {
            output_queue: Duration::from_secs(2),
            max_text_chars: 10_000,
            chunk_chars: 300,
        }
    }
}

impl TtsLimits {
    /// Sets the output queue.
    #[must_use]
    pub const fn with_output_queue(mut self, value: Duration) -> Self {
        self.output_queue = value;
        self
    }

    /// Sets the text limit.
    #[must_use]
    pub const fn with_max_text_chars(mut self, value: usize) -> Self {
        self.max_text_chars = value;
        self
    }

    /// Sets the chunk length.
    #[must_use]
    pub const fn with_chunk_chars(mut self, value: usize) -> Self {
        self.chunk_chars = value;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voices() -> Vec<Voice> {
        vec![
            Voice::new("alloy").with_languages(["en"]),
            Voice::new("xiaoyun").with_languages(["zh"]),
        ]
    }

    #[test]
    fn validate_table() {
        let mut caps = TtsCapabilities::new(SampleRate::HZ_24000, 4_096);
        caps.speed = Some(0.5..=2.0);
        let fixed = TtsCapabilities::new(SampleRate::HZ_24000, 4_096);
        let rows: Vec<(TtsOptions, &TtsCapabilities, Result<(), &str>)> = vec![
            (TtsOptions::default(), &caps, Ok(())),
            (TtsOptions::default().with_voice("alloy"), &caps, Ok(())),
            (
                TtsOptions::default().with_voice("nobody"),
                &caps,
                Err("invalid"),
            ),
            (TtsOptions::default().with_speed(1.5), &caps, Ok(())),
            (TtsOptions::default().with_speed(2.5), &caps, Err("invalid")),
            (TtsOptions::default().with_speed(0.0), &caps, Err("invalid")),
            (
                TtsOptions::default().with_speed(f32::NAN),
                &caps,
                Err("invalid"),
            ),
            (
                TtsOptions::default().with_speed(1.5),
                &fixed,
                Err("unsupported"),
            ),
            (TtsOptions::default().with_speed(1.0), &fixed, Ok(())),
            (
                TtsOptions::default()
                    .with_voice("alloy")
                    .with_language("en"),
                &caps,
                Ok(()),
            ),
            (
                TtsOptions::default()
                    .with_voice("alloy")
                    .with_language("zh"),
                &caps,
                Err("unsupported"),
            ),
            (TtsOptions::default().with_language("zh"), &caps, Ok(())),
            (
                TtsOptions::default().with_language("fr"),
                &caps,
                Err("unsupported"),
            ),
        ];
        for (opts, caps, expected) in rows {
            let result = validate(&opts, caps, &voices());
            match (result, expected) {
                (Ok(()), Ok(()))
                | (Err(SpeechError::InvalidInput(_)), Err("invalid"))
                | (Err(SpeechError::Unsupported(_)), Err("unsupported")) => {}
                (other, expected) => panic!("{opts:?}: got {other:?}, expected {expected:?}"),
            }
        }
        assert!(validate(&TtsOptions::default().with_language("fr"), &caps, &[]).is_ok());
    }

    #[test]
    fn default_limits() {
        let limits = TtsLimits::default();
        assert_eq!(limits.output_queue, Duration::from_secs(2));
        assert_eq!(limits.max_text_chars, 10_000);
        assert_eq!(limits.chunk_chars, 300);
        assert_eq!(limits.with_chunk_chars(5).chunk_chars, 5);
    }
}
