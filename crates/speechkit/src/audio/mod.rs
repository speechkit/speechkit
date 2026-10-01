//! Audio files: reading them into mono [`AudioBuffer`]s, and writing WAV.
//!
//! [`read`] and [`decode`] return mono audio at the file's own rate:
//! channels are averaged, and samples are clamped to [-1.0, 1.0]. Sessions
//! resample to their backend's rate themselves. The format comes from the
//! content, not the file name. WAV always works; FLAC, MP3, AAC, Ogg, and
//! Matroska need the `decode` feature.

#[cfg(feature = "decode")]
mod decode;
mod wav;

use std::{
    fs::File,
    io::{BufReader, Cursor, Read, SeekFrom},
    path::Path,
    time::Duration,
};

#[cfg(feature = "openai")]
pub(crate) use wav::encode_samples;
pub use wav::encode_wav;
#[cfg(any(feature = "server", feature = "openai", feature = "dashscope"))]
pub(crate) use wav::pcm16_bytes;

use crate::{AudioBuffer, SpeechError};

/// The file extensions, in lowercase and without the dot, of the formats
/// [`read`] and [`decode`] handle in this build, for filtering a file
/// dialog.
///
/// Without the `decode` feature this is only `wav`. With it, it also
/// covers FLAC, MP3, AAC (ADTS, or MP4 as `.m4a` and `.mp4`), Ogg (Vorbis
/// or FLAC), and Matroska. `.webm` and `.opus` are left out: they usually
/// hold Opus, which is not decoded. The format is found from the content,
/// so a file named otherwise still decodes, and a listed container holding
/// another codec still fails.
#[cfg(feature = "decode")]
pub const EXTENSIONS: &[&str] = &[
    "aac", "flac", "m4a", "mka", "mkv", "mp3", "mp4", "oga", "ogg", "wav",
];

/// The file extensions, in lowercase and without the dot, of the formats
/// [`read`] and [`decode`] handle in this build, for filtering a file
/// dialog.
///
/// Without the `decode` feature this is only `wav`. With it, it also
/// covers FLAC, MP3, AAC, Ogg, and Matroska.
#[cfg(not(feature = "decode"))]
pub const EXTENSIONS: &[&str] = &["wav"];

/// Limits on reading audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeLimits {
    /// The longest recording accepted. Default: 20 min. This bounds memory:
    /// mono f32 at 48 kHz takes about 11 MiB per minute.
    pub max_duration: Duration,
}

impl DecodeLimits {
    /// Limits accepting at most `max_duration` of audio.
    pub const fn new(max_duration: Duration) -> Self {
        Self { max_duration }
    }
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::new(Duration::from_secs(20 * 60))
    }
}

/// The formats this build reads, for error messages.
#[cfg(feature = "decode")]
const SUPPORTED_FORMATS: &str = "WAV (PCM or float), FLAC, MP3, M4A/MP4/AAC, Ogg Vorbis or FLAC, \
                                 and MKV/WebM with one of those codecs";

/// The formats this build reads, for error messages.
#[cfg(not(feature = "decode"))]
const SUPPORTED_FORMATS: &str =
    "WAV (PCM or float); FLAC, MP3, AAC, Ogg, and Matroska need the `decode` feature";

fn invalid(message: impl std::fmt::Display) -> SpeechError {
    SpeechError::InvalidInput(format!("cannot decode audio: {message}"))
}

/// Reads an audio file.
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] if the file cannot be read or decoded, is
/// in a format this build doesn't read, or is longer than
/// `limits.max_duration`.
pub fn read(path: impl AsRef<Path>, limits: DecodeLimits) -> Result<AudioBuffer, SpeechError> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| invalid(format_args!("{}: {e}", path.display())))?;
    from_source(file, limits)
}

/// Decodes audio held in memory.
///
/// # Errors
///
/// As [`read`].
pub fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<AudioBuffer, SpeechError> {
    from_source(Cursor::new(bytes.to_vec()), limits)
}

/// Where audio is read from: a file or bytes in memory.
#[cfg(feature = "decode")]
trait Source: symphonia::core::io::MediaSource + 'static {}
#[cfg(feature = "decode")]
impl<T: symphonia::core::io::MediaSource + 'static> Source for T {}

/// Where audio is read from: a file or bytes in memory.
#[cfg(not(feature = "decode"))]
trait Source: Read + std::io::Seek {}
#[cfg(not(feature = "decode"))]
impl<T: Read + std::io::Seek> Source for T {}

/// Reads WAV with hound, and everything else, including WAV encodings
/// hound doesn't read, with symphonia when the `decode` feature is on.
fn from_source(mut source: impl Source, limits: DecodeLimits) -> Result<AudioBuffer, SpeechError> {
    let mut header = [0_u8; 12];
    let filled = read_up_to(&mut source, &mut header).map_err(invalid)?;
    source.seek(SeekFrom::Start(0)).map_err(invalid)?;
    if filled == header.len() && &header[..4] == b"RIFF" && &header[8..] == b"WAVE" {
        match wav::read(BufReader::new(&mut source), limits) {
            Ok(audio) => return Ok(audio),
            Err(wav::Error::Invalid(error)) => return Err(error),
            // symphonia reads some encodings hound doesn't, such as A-law.
            #[cfg(feature = "decode")]
            Err(wav::Error::Unsupported(_)) => {
                source.seek(SeekFrom::Start(0)).map_err(invalid)?;
            }
            #[cfg(not(feature = "decode"))]
            Err(wav::Error::Unsupported(message)) => return Err(invalid(message)),
        }
    }
    #[cfg(feature = "decode")]
    {
        decode::decode(source, limits)
    }
    #[cfg(not(feature = "decode"))]
    {
        Err(invalid(format_args!(
            "not a WAV file; supported formats: {SUPPORTED_FORMATS}"
        )))
    }
}

/// Fills as much of `buffer` as the source holds.
fn read_up_to(source: &mut impl Read, buffer: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match source.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

/// Clamps a decoded sample into [-1.0, 1.0], the range sessions accept.
/// Lossy decoders overshoot full scale and float WAVs may peak above it;
/// a non-finite sample becomes silence.
fn to_session_range(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_content_names_the_supported_formats() {
        let error = decode(b"definitely not audio", DecodeLimits::default()).unwrap_err();
        assert!(
            error.to_string().contains("supported formats: WAV"),
            "{error}"
        );
        assert!(decode(b"", DecodeLimits::default()).is_err());
    }

    #[test]
    fn wav_is_always_listed() {
        assert!(EXTENSIONS.contains(&"wav"));
        assert!(EXTENSIONS.is_sorted());
    }
}
