//! The worker protocol: messages and framing.

use std::{
    io::{self, BufRead, Read, Write},
    time::Duration,
};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrCapabilities, AsrEvent, AsrOptions, Partial, Segment, UtteranceId},
};
use serde::{Deserialize, Serialize};

/// The protocol version, the first byte of every frame payload. It covers
/// the `--config` JSON too.
pub(super) const VERSION: u8 = 4;

/// The line a worker prints before its first frame.
pub(super) const HANDSHAKE: &[u8] = b"SPEECHKIT-WORKER-4\n";

/// How every version's handshake starts.
const HANDSHAKE_PREFIX: &[u8] = b"SPEECHKIT-WORKER-";

/// The largest frame accepted: 64 MiB.
const MAX_FRAME: u32 = 64 * 1024 * 1024;

/// A request from the parent. A worker serves one stream at a time, and
/// answers each request with `Done` or `Error`, after every event the
/// request produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum Request {
    /// Open a stream.
    Open {
        /// The session's language.
        language: Option<String>,
        /// The session's hints.
        hints: Vec<String>,
    },
    /// Feed samples to the stream.
    Accept {
        /// Samples at the backend's rate.
        samples: Vec<f32>,
    },
    /// Finish the stream.
    Finish,
    /// Cancel the stream.
    Cancel,
    /// Drop the stream. Once `Done` is sent, no event of it follows.
    Close,
}

/// A message from the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum Response {
    /// The first message: what the backend is.
    Hello {
        /// The backend's name.
        name: String,
        /// Its capabilities.
        caps: WireCaps,
    },
    /// The request succeeded.
    Done,
    /// The request failed.
    Error(WireError),
    /// An event the stream sent, at any time.
    Event(WireEvent),
    /// The stream reported a failure between calls.
    Failed(WireError),
}

/// [`AsrCapabilities`] on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[expect(clippy::struct_excessive_bools, reason = "mirrors AsrCapabilities")]
pub struct WireCaps {
    reports_partials: bool,
    reports_activity: bool,
    accepts_hints: bool,
    accepts_language: bool,
    punctuated: bool,
    rate: u32,
}

impl From<&AsrCapabilities> for WireCaps {
    fn from(caps: &AsrCapabilities) -> Self {
        Self {
            reports_partials: caps.reports_partials,
            reports_activity: caps.reports_activity,
            accepts_hints: caps.accepts_hints,
            accepts_language: caps.accepts_language,
            punctuated: caps.punctuated,
            rate: caps.sample_rate.hz(),
        }
    }
}

impl TryFrom<WireCaps> for AsrCapabilities {
    type Error = SpeechError;

    fn try_from(wire: WireCaps) -> Result<Self, SpeechError> {
        let mut caps = Self::new(SampleRate::new(wire.rate)?);
        caps.reports_partials = wire.reports_partials;
        caps.reports_activity = wire.reports_activity;
        caps.accepts_hints = wire.accepts_hints;
        caps.accepts_language = wire.accepts_language;
        caps.punctuated = wire.punctuated;
        Ok(caps)
    }
}

/// An [`AsrEvent`] on the wire, times in nanoseconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireEvent {
    /// [`AsrEvent::Partial`].
    Partial {
        /// The utterance.
        utterance: u64,
        /// The pending text.
        text: String,
    },
    /// [`AsrEvent::Segment`].
    Segment {
        /// The utterance.
        utterance: u64,
        /// The final text.
        text: String,
        /// Where it starts.
        start_ns: u64,
        /// Where it ends.
        end_ns: u64,
    },
    /// [`AsrEvent::SpeechStarted`].
    SpeechStarted {
        /// Where.
        at_ns: u64,
    },
    /// [`AsrEvent::SpeechEnded`].
    SpeechEnded {
        /// Where.
        at_ns: u64,
        /// The last utterance of the speech.
        utterance: u64,
    },
    /// [`AsrEvent::ActivityKnown`].
    ActivityKnown {
        /// How far.
        through_ns: u64,
    },
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl From<AsrEvent> for WireEvent {
    fn from(event: AsrEvent) -> Self {
        match event {
            AsrEvent::Partial(partial) => Self::Partial {
                utterance: partial.utterance.0,
                text: partial.text,
            },
            AsrEvent::Segment(segment) => Self::Segment {
                utterance: segment.utterance.0,
                text: segment.text,
                start_ns: nanos(segment.start),
                end_ns: nanos(segment.end),
            },
            AsrEvent::SpeechStarted { at } => Self::SpeechStarted { at_ns: nanos(at) },
            AsrEvent::SpeechEnded { at, utterance } => Self::SpeechEnded {
                at_ns: nanos(at),
                utterance: utterance.0,
            },
            AsrEvent::ActivityKnown { through } => Self::ActivityKnown {
                through_ns: nanos(through),
            },
        }
    }
}

impl From<WireEvent> for AsrEvent {
    fn from(wire: WireEvent) -> Self {
        match wire {
            WireEvent::Partial { utterance, text } => Self::Partial(Partial {
                utterance: UtteranceId(utterance),
                text,
            }),
            WireEvent::Segment {
                utterance,
                text,
                start_ns,
                end_ns,
            } => Self::Segment(Segment {
                utterance: UtteranceId(utterance),
                text,
                start: Duration::from_nanos(start_ns),
                end: Duration::from_nanos(end_ns),
            }),
            WireEvent::SpeechStarted { at_ns } => Self::SpeechStarted {
                at: Duration::from_nanos(at_ns),
            },
            WireEvent::SpeechEnded { at_ns, utterance } => Self::SpeechEnded {
                at: Duration::from_nanos(at_ns),
                utterance: UtteranceId(utterance),
            },
            WireEvent::ActivityKnown { through_ns } => Self::ActivityKnown {
                through: Duration::from_nanos(through_ns),
            },
        }
    }
}

/// A [`SpeechError`] on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireError {
    /// `InvalidInput`.
    InvalidInput(String),
    /// `InvalidModel`.
    InvalidModel(String),
    /// `Unsupported`.
    Unsupported(String),
    /// `Backend`, or anything else.
    Backend {
        /// The backend's name.
        backend: String,
        /// Whether a retry might help.
        retryable: bool,
        /// What went wrong.
        message: String,
    },
}

impl From<&SpeechError> for WireError {
    fn from(error: &SpeechError) -> Self {
        match error {
            SpeechError::InvalidInput(m) => Self::InvalidInput(m.clone()),
            SpeechError::InvalidModel(m) => Self::InvalidModel(m.clone()),
            SpeechError::Unsupported(m) => Self::Unsupported(m.clone()),
            SpeechError::Backend {
                backend,
                retryable,
                source,
            } => Self::Backend {
                backend: backend.clone(),
                retryable: *retryable,
                message: source.to_string(),
            },
            other => Self::Backend {
                backend: "speechkit-worker".into(),
                retryable: other.retryable(),
                message: other.to_string(),
            },
        }
    }
}

impl From<WireError> for SpeechError {
    fn from(wire: WireError) -> Self {
        match wire {
            WireError::InvalidInput(m) => Self::InvalidInput(m),
            WireError::InvalidModel(m) => Self::InvalidModel(m),
            WireError::Unsupported(m) => Self::Unsupported(m),
            WireError::Backend {
                backend,
                retryable,
                message,
            } => Self::backend(backend, retryable, message),
        }
    }
}

/// Session options from an `Open` request.
pub(crate) fn options(language: Option<String>, hints: Vec<String>) -> AsrOptions {
    let mut options = AsrOptions::default().with_hints(hints);
    options.language = language;
    options
}

/// Writes one frame.
///
/// # Errors
///
/// Any write error, or an encoding failure as `InvalidData`.
pub(super) fn write_frame<T: Serialize>(out: &mut impl Write, message: &T) -> io::Result<()> {
    let mut payload = vec![VERSION];
    let encoded =
        postcard::to_stdvec(message).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    payload.extend_from_slice(&encoded);
    let length =
        u32::try_from(payload.len()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    out.write_all(&length.to_le_bytes())?;
    out.write_all(&payload)?;
    out.flush()
}

/// Reads one frame. `Ok(None)` means the stream ended cleanly.
///
/// # Errors
///
/// Any read error, or `InvalidData` for an oversized frame, a wrong
/// version, or a message that does not decode.
pub(super) fn read_frame<T: for<'de> Deserialize<'de>>(
    input: &mut impl Read,
) -> io::Result<Option<T>> {
    let mut length = [0_u8; 4];
    match input.read_exact(&mut length) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let length = u32::from_le_bytes(length);
    let invalid = |message: String| io::Error::new(io::ErrorKind::InvalidData, message);
    if length == 0 || length > MAX_FRAME {
        return Err(invalid(format!("bad frame length {length}")));
    }
    let mut payload = vec![0_u8; length as usize];
    input.read_exact(&mut payload)?;
    if payload[0] != VERSION {
        return Err(invalid(format!(
            "protocol version {} is not {VERSION}",
            payload[0]
        )));
    }
    postcard::from_bytes(&payload[1..])
        .map(Some)
        .map_err(|e| invalid(e.to_string()))
}

/// Skips input up to and including the handshake line, reading at most
/// 1 MiB.
///
/// # Errors
///
/// `UnexpectedEof` if the input ends first, or `InvalidData` for another
/// version's handshake or too much noise.
pub(super) fn skip_to_handshake(input: &mut impl BufRead) -> io::Result<()> {
    let mut skipped = 0_usize;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = input.read_until(b'\n', &mut line)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the worker exited before its handshake",
            ));
        }
        if line.ends_with(HANDSHAKE) {
            return Ok(());
        }
        if let Some(start) = line
            .windows(HANDSHAKE_PREFIX.len())
            .position(|window| window == HANDSHAKE_PREFIX)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "the worker speaks {}, not {}; build speechkit-worker from the same \
                     speechkit version",
                    String::from_utf8_lossy(&line[start..]).trim_end(),
                    String::from_utf8_lossy(HANDSHAKE).trim_end()
                ),
            ));
        }
        skipped += read;
        if skipped > 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "no worker handshake",
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip() {
        let mut buffer = Vec::new();
        let request = Request::Accept {
            samples: vec![0.5, -0.25],
        };
        write_frame(&mut buffer, &request).unwrap();
        write_frame(&mut buffer, &Response::Done).unwrap();
        let mut cursor = io::Cursor::new(buffer);
        assert_eq!(read_frame::<Request>(&mut cursor).unwrap(), Some(request));
        assert_eq!(
            read_frame::<Response>(&mut cursor).unwrap(),
            Some(Response::Done)
        );
        assert_eq!(read_frame::<Response>(&mut cursor).unwrap(), None);
    }

    #[test]
    fn bad_frames_are_rejected() {
        let mut wrong_version = Vec::new();
        wrong_version.extend_from_slice(&2_u32.to_le_bytes());
        wrong_version.extend_from_slice(&[9, 0]);
        assert!(read_frame::<Response>(&mut io::Cursor::new(wrong_version)).is_err());
        let huge = u32::MAX.to_le_bytes().to_vec();
        assert!(read_frame::<Response>(&mut io::Cursor::new(huge)).is_err());
        let truncated = vec![10, 0, 0, 0, VERSION];
        assert!(read_frame::<Response>(&mut io::Cursor::new(truncated)).is_err());
    }

    #[test]
    fn handshake_skips_noise() {
        let mut input = io::Cursor::new(
            b"running 1 test\nnoise"
                .iter()
                .chain(HANDSHAKE)
                .copied()
                .collect::<Vec<_>>(),
        );
        skip_to_handshake(&mut input).unwrap();
        assert!(skip_to_handshake(&mut io::Cursor::new(b"nothing\n".to_vec())).is_err());
        let old =
            skip_to_handshake(&mut io::Cursor::new(b"SPEECHKIT-WORKER-3\n".to_vec())).unwrap_err();
        assert!(old.to_string().contains("SPEECHKIT-WORKER-3"), "{old}");
    }

    #[test]
    fn events_errors_and_caps_convert() {
        let segment = AsrEvent::Segment(Segment {
            utterance: UtteranceId(2),
            text: "hi".into(),
            start: Duration::from_millis(5),
            end: Duration::from_millis(9),
        });
        assert_eq!(AsrEvent::from(WireEvent::from(segment.clone())), segment);
        let partial = AsrEvent::Partial(Partial {
            utterance: UtteranceId(1),
            text: "h".into(),
        });
        assert_eq!(AsrEvent::from(WireEvent::from(partial.clone())), partial);
        let activity = [
            AsrEvent::SpeechStarted {
                at: Duration::from_millis(3),
            },
            AsrEvent::SpeechEnded {
                at: Duration::from_millis(4),
                utterance: UtteranceId(7),
            },
            AsrEvent::ActivityKnown {
                through: Duration::from_millis(5),
            },
        ];
        for event in activity {
            assert_eq!(AsrEvent::from(WireEvent::from(event.clone())), event);
        }
        for error in [
            SpeechError::InvalidInput("a".into()),
            SpeechError::InvalidModel("b".into()),
            SpeechError::Unsupported("c".into()),
            SpeechError::backend("x", true, "d"),
            SpeechError::Capacity,
        ] {
            let back = SpeechError::from(WireError::from(&error));
            assert_eq!(back.retryable(), error.retryable(), "{error}");
        }
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.reports_partials = true;
        let back = AsrCapabilities::try_from(WireCaps::from(&caps)).unwrap();
        assert_eq!(back, caps);
        let opts = options(Some("zh".into()), vec!["a".into()]);
        assert_eq!(opts.hints, ["a"]);
    }
}
