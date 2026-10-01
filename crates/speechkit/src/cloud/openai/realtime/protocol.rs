//! Realtime events: pure encoding and decoding.

use std::time::Duration;

use crate::{SpeechError, audio::pcm16_bytes};
use base64::Engine as _;
use serde_json::{Value, json};

use crate::cloud::ws::server_error;

/// The backend name used in errors.
pub(crate) const BACKEND: &str = "openai-realtime";

/// The server error code for committing an empty buffer.
pub(crate) const COMMIT_EMPTY: &str = "input_audio_buffer_commit_empty";

/// `session.update` for a transcription session.
pub(crate) fn session_update(model: &str, language: Option<&str>, server_vad: bool) -> Value {
    let mut transcription = json!({ "model": model });
    if let Some(language) = language {
        transcription["language"] = json!(language);
    }
    json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": { "type": "audio/pcm", "rate": 24_000 },
                    "transcription": transcription,
                    "turn_detection": if server_vad { json!({ "type": "server_vad" }) } else { Value::Null },
                }
            }
        }
    })
}

/// `session.update` turning server VAD off, before the final commit.
pub(crate) fn disable_turn_detection() -> Value {
    json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": { "input": { "turn_detection": null } }
        }
    })
}

/// `input_audio_buffer.append` with 24 kHz samples.
pub(crate) fn append(samples: &[f32]) -> Value {
    let audio = base64::engine::general_purpose::STANDARD.encode(pcm16_bytes(samples));
    json!({ "type": "input_audio_buffer.append", "audio": audio })
}

/// `input_audio_buffer.commit`.
pub(crate) fn commit() -> Value {
    json!({ "type": "input_audio_buffer.commit" })
}

/// A server event the client acts on.
#[derive(Debug)]
pub(crate) enum Event {
    /// The session settings took effect.
    SessionUpdated,
    /// Server VAD found the start of speech.
    SpeechStarted {
        /// Where, from the start of the session.
        audio_start: Duration,
    },
    /// Server VAD found the end of an item's speech.
    SpeechStopped {
        /// The item.
        item_id: String,
        /// Where the item's audio ends, from the start of the session.
        audio_end: Duration,
    },
    /// A buffer was committed as a conversation item.
    Committed {
        /// The item.
        item_id: String,
    },
    /// Transcript text to append for an item.
    Delta {
        /// The item.
        item_id: String,
        /// The text.
        delta: String,
    },
    /// The final transcript of an item.
    Completed {
        /// The item.
        item_id: String,
        /// The text.
        transcript: String,
    },
    /// A server error with its code, if any.
    Error {
        /// The error code.
        code: Option<String>,
        /// The error, ready to report.
        error: SpeechError,
    },
    /// Anything else.
    Other,
}

// Only the tests compare events.
#[cfg(test)]
impl PartialEq for Event {
    /// Errors compare by code and message.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Error { code: a, error: x }, Self::Error { code: b, error: y }) => {
                a == b && x.to_string() == y.to_string()
            }
            (Self::SessionUpdated, Self::SessionUpdated) | (Self::Other, Self::Other) => true,
            (Self::SpeechStarted { audio_start: a }, Self::SpeechStarted { audio_start: b }) => {
                a == b
            }
            (
                Self::SpeechStopped {
                    item_id: a,
                    audio_end: x,
                },
                Self::SpeechStopped {
                    item_id: b,
                    audio_end: y,
                },
            ) => a == b && x == y,
            (Self::Committed { item_id: a }, Self::Committed { item_id: b }) => a == b,
            (
                Self::Delta {
                    item_id: a,
                    delta: x,
                },
                Self::Delta {
                    item_id: b,
                    delta: y,
                },
            )
            | (
                Self::Completed {
                    item_id: a,
                    transcript: x,
                },
                Self::Completed {
                    item_id: b,
                    transcript: y,
                },
            ) => a == b && x == y,
            _ => false,
        }
    }
}

fn missing(what: &str) -> SpeechError {
    SpeechError::backend(BACKEND, false, format!("an event is missing {what}"))
}

fn text(value: &Value, key: &str) -> Result<String, SpeechError> {
    value[key]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| missing(key))
}

/// Decodes one server event.
///
/// # Errors
///
/// A non-retryable backend error for a known event without its fields.
pub(crate) fn parse(value: &Value) -> Result<Event, SpeechError> {
    Ok(match value["type"].as_str() {
        Some("session.updated") => Event::SessionUpdated,
        Some("input_audio_buffer.speech_started") => Event::SpeechStarted {
            audio_start: Duration::from_millis(
                value["audio_start_ms"]
                    .as_u64()
                    .ok_or_else(|| missing("audio_start_ms"))?,
            ),
        },
        Some("input_audio_buffer.speech_stopped") => Event::SpeechStopped {
            item_id: text(value, "item_id")?,
            audio_end: Duration::from_millis(
                value["audio_end_ms"]
                    .as_u64()
                    .ok_or_else(|| missing("audio_end_ms"))?,
            ),
        },
        Some("input_audio_buffer.committed") => Event::Committed {
            item_id: text(value, "item_id")?,
        },
        Some("conversation.item.input_audio_transcription.delta") => Event::Delta {
            item_id: text(value, "item_id")?,
            delta: text(value, "delta")?,
        },
        Some("conversation.item.input_audio_transcription.completed") => Event::Completed {
            item_id: text(value, "item_id")?,
            transcript: text(value, "transcript")?,
        },
        Some("conversation.item.input_audio_transcription.failed") => {
            let error = &value["error"];
            Event::Error {
                code: error["code"].as_str().map(str::to_owned),
                error: server_error(
                    BACKEND,
                    "transcription failed",
                    &error["code"],
                    &error["message"],
                    false,
                ),
            }
        }
        Some("error") => {
            let error = &value["error"];
            let code = error["code"].as_str().map(str::to_owned);
            let retryable = code
                .as_deref()
                .is_some_and(|c| c.contains("rate_limit") || c.contains("server_error"));
            Event::Error {
                code,
                error: server_error(
                    BACKEND,
                    "the server reported an error",
                    &error["code"],
                    &error["message"],
                    retryable,
                ),
            }
        }
        _ => Event::Other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_requests() {
        let update = session_update("gpt-4o-transcribe", Some("zh"), true);
        assert_eq!(
            update["session"]["audio"]["input"]["format"]["rate"],
            24_000
        );
        assert_eq!(
            update["session"]["audio"]["input"]["transcription"]["language"],
            "zh"
        );
        assert_eq!(
            update["session"]["audio"]["input"]["turn_detection"]["type"],
            "server_vad"
        );
        let no_vad = session_update("m", None, false);
        assert!(no_vad["session"]["audio"]["input"]["turn_detection"].is_null());
        assert!(no_vad["session"]["audio"]["input"]["transcription"]["language"].is_null());
        assert!(disable_turn_detection()["session"]["audio"]["input"]["turn_detection"].is_null());
        assert_eq!(append(&[0.5])["audio"], "AEA=");
        assert_eq!(commit()["type"], "input_audio_buffer.commit");
    }

    #[test]
    fn decodes_events() {
        let parse_str = |s: &str| parse(&serde_json::from_str(s).unwrap());
        assert_eq!(
            parse_str(r#"{"type":"session.updated"}"#).unwrap(),
            Event::SessionUpdated
        );
        assert_eq!(
            parse_str(
                r#"{"type":"input_audio_buffer.committed","item_id":"a","previous_item_id":""}"#
            )
            .unwrap(),
            Event::Committed {
                item_id: "a".into()
            }
        );
        assert_eq!(
            parse_str(r#"{"type":"conversation.item.input_audio_transcription.delta","item_id":"a","delta":"hi"}"#).unwrap(),
            Event::Delta { item_id: "a".into(), delta: "hi".into() }
        );
        assert_eq!(
            parse_str(r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"a","transcript":"hi."}"#).unwrap(),
            Event::Completed { item_id: "a".into(), transcript: "hi.".into() }
        );
        assert_eq!(
            parse_str(
                r#"{"type":"input_audio_buffer.speech_stopped","item_id":"a","audio_end_ms":1250}"#
            )
            .unwrap(),
            Event::SpeechStopped {
                item_id: "a".into(),
                audio_end: Duration::from_millis(1_250)
            }
        );
        assert!(
            parse_str(r#"{"type":"input_audio_buffer.speech_stopped","item_id":"a"}"#).is_err()
        );
        assert_eq!(
            parse_str(
                r#"{"type":"input_audio_buffer.speech_started","item_id":"a","audio_start_ms":300}"#
            )
            .unwrap(),
            Event::SpeechStarted {
                audio_start: Duration::from_millis(300)
            }
        );
        assert!(parse_str(r#"{"type":"input_audio_buffer.committed"}"#).is_err());
        let Event::Error { code, error } = parse_str(
            r#"{"type":"error","error":{"code":"rate_limit_exceeded","message":"slow"}}"#,
        )
        .unwrap() else {
            panic!("error event");
        };
        assert_eq!(code.as_deref(), Some("rate_limit_exceeded"));
        assert!(error.retryable());
        assert!(matches!(
            parse_str(
                r#"{"type":"conversation.item.input_audio_transcription.failed","error":{}}"#
            )
            .unwrap(),
            Event::Error { code: None, .. }
        ));
        assert_eq!(
            parse_str(r#"{"type":"rate_limits.updated"}"#).unwrap(),
            Event::Other
        );
    }
}
