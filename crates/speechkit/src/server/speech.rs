//! `POST /v1/audio/speech`.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    RecvError, SampleRate, SpeechError,
    audio::pcm16_bytes,
    speech::{deadline::instant_after, opening::Opening},
    tts::{TtsEngine, TtsOptions, TtsOutput, TtsSession, TtsUpdate},
};
use axum::{
    Json,
    body::{Body, Bytes},
    extract::{State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::Value;

use crate::server::{
    error::{describe, respond, slot_busy, status_of},
    state::Shared,
};

/// The rate of every response, as OpenAI documents for `pcm`.
pub(crate) const RATE: SampleRate = SampleRate::HZ_24000;

/// Audio chunks in flight between the synthesis thread and the socket.
const IN_FLIGHT: usize = 4;

/// Cancels the session when dropped: a client that disconnects drops the
/// response body, and with it this guard.
struct CancelOnDrop(Arc<TtsSession>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Wav,
    Pcm,
}

struct Request {
    input: String,
    voice: Option<String>,
    speed: f32,
    format: Format,
}

#[expect(
    clippy::unnecessary_box_returns,
    reason = "a boxed response keeps `read`'s Result small"
)]
fn bad(message: impl std::fmt::Display) -> Box<Response> {
    Box::new(respond(StatusCode::BAD_REQUEST, message))
}

fn read(body: &Value, max_chars: usize) -> Result<Request, Box<Response>> {
    let input = body["input"]
        .as_str()
        .ok_or_else(|| bad("the `input` field is required"))?;
    if input.trim().is_empty() {
        return Err(bad("`input` is empty"));
    }
    let chars = input.chars().count();
    if chars > max_chars {
        return Err(bad(format!(
            "`input` has {chars} characters; the limit is {max_chars}"
        )));
    }
    let format = match &body["response_format"] {
        Value::Null => "wav",
        Value::String(format) => format.as_str(),
        _ => return Err(bad("`response_format` must be a string")),
    };
    let format = match format {
        "wav" => Format::Wav,
        "pcm" => Format::Pcm,
        other @ ("mp3" | "opus" | "aac" | "flac") => {
            return Err(bad(format!(
                "response_format {other:?} is not supported; use wav or pcm"
            )));
        }
        other => return Err(bad(format!("unknown response_format {other:?}"))),
    };
    let speed = match &body["speed"] {
        Value::Null => 1.0,
        value => {
            #[expect(clippy::cast_possible_truncation, reason = "speeds are small")]
            let speed = value
                .as_f64()
                .ok_or_else(|| bad("`speed` must be a number"))? as f32;
            speed
        }
    };
    let voice = match &body["voice"] {
        Value::Null => None,
        Value::String(voice) => Some(voice.clone()),
        _ => return Err(bad("`voice` must be a string")),
    };
    // `model` is accepted and ignored, as for transcription.
    Ok(Request {
        input: input.to_owned(),
        voice,
        speed,
        format,
    })
}

/// A WAV header for a stream of unknown length. The size fields hold
/// `0xFFFFFFFF`, which most players read as "until the end".
pub(crate) fn streaming_wav_header(rate: SampleRate) -> Vec<u8> {
    let unknown = u32::MAX.to_le_bytes();
    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&unknown);
    header.extend_from_slice(b"WAVEfmt ");
    header.extend_from_slice(&16_u32.to_le_bytes());
    header.extend_from_slice(&1_u16.to_le_bytes());
    header.extend_from_slice(&1_u16.to_le_bytes());
    header.extend_from_slice(&rate.hz().to_le_bytes());
    header.extend_from_slice(&(rate.hz() * 2).to_le_bytes());
    header.extend_from_slice(&2_u16.to_le_bytes());
    header.extend_from_slice(&16_u16.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&unknown);
    header
}

/// Moves audio from the output to the channel until the end, a failure,
/// or the receiver going away. Returning drops the output, which cancels a
/// synthesis that has not ended.
fn pump(
    mut output: TtsOutput,
    deadline: Instant,
    sender: &tokio::sync::mpsc::Sender<Result<Bytes, SpeechError>>,
) {
    loop {
        let item = match output.recv(deadline) {
            Ok(TtsUpdate::Audio(samples)) => Ok(Bytes::from(pcm16_bytes(&samples))),
            Ok(TtsUpdate::Closed(Ok(_))) | Err(RecvError::Closed) => return,
            Ok(TtsUpdate::Closed(Err(failure))) => Err(failure.error),
            Ok(_) => continue,
            Err(RecvError::Timeout | RecvError::Empty) => Err(SpeechError::DeadlineExceeded),
        };
        let failed = item.is_err();
        let sent = tokio::runtime::Handle::current()
            .block_on(async { tokio::time::timeout_at(deadline.into(), sender.send(item)).await });
        if !matches!(sent, Ok(Ok(()))) || failed {
            return;
        }
    }
}

/// Starts a synthesis, answering `Capacity` at once when every slot is
/// busy.
async fn open(
    engine: TtsEngine,
    options: TtsOptions,
    deadline: Instant,
) -> Result<(TtsSession, TtsOutput), SpeechError> {
    let task = tokio::task::spawn_blocking(move || {
        engine.start_with(options, deadline.into(), Opening::NoWait)
    });
    match tokio::time::timeout_at(deadline.into(), task).await {
        Ok(Ok(opened)) => opened,
        Ok(Err(error)) => Err(SpeechError::backend("speechkit-server", false, error)),
        Err(_) => Err(SpeechError::DeadlineExceeded),
    }
}

pub(crate) async fn handle(
    State(state): State<Shared>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Some((engine, _)) = &state.server.tts else {
        return respond(StatusCode::NOT_FOUND, "this server has no speech engine");
    };
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return respond(StatusCode::BAD_REQUEST, rejection.body_text()),
    };
    let request = match read(&body, state.server.max_speech_chars) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let mut options = TtsOptions::default()
        .with_speed(request.speed)
        .with_sample_rate(RATE);
    options.voice = request.voice;
    #[expect(clippy::cast_precision_loss, reason = "input is bounded")]
    let length = Duration::from_secs_f64(request.input.chars().count() as f64 * 0.5);
    let deadline = instant_after(Instant::now(), state.server.budget(length));
    let (session, output) = match open(engine.clone(), options, deadline).await {
        Ok((session, output)) => (Arc::new(session), output),
        Err(SpeechError::Capacity) => {
            tracing::warn!(
                active_sessions = engine.active_sessions(),
                "every synthesis slot is busy; answering 503"
            );
            return slot_busy();
        }
        Err(error) => return respond(status_of(&error), describe(&error)),
    };
    let guard = CancelOnDrop(session.clone());
    if let Err(error) = session.push_text(&request.input) {
        return respond(status_of(&error), describe(&error));
    }
    session.close_text();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(IN_FLIGHT);
    tokio::task::spawn_blocking(move || pump(output, deadline, &sender));
    // Wait for the first audio, so an early failure gets a proper status.
    let Ok(first) = tokio::time::timeout_at(deadline.into(), receiver.recv()).await else {
        return respond(StatusCode::GATEWAY_TIMEOUT, "synthesis timed out");
    };
    let first = match first {
        Some(Ok(bytes)) => Some(bytes),
        Some(Err(error)) => return respond(status_of(&error), describe(&error)),
        None => None,
    };
    let mut head = Vec::new();
    if request.format == Format::Wav {
        head.push(Bytes::from(streaming_wav_header(RATE)));
    }
    head.extend(first);
    let rest = futures::stream::unfold(
        (receiver, guard, false),
        move |(mut receiver, guard, ended)| async move {
            if ended {
                return None;
            }
            let item = if Instant::now() >= deadline {
                Err(SpeechError::DeadlineExceeded)
            } else {
                match tokio::time::timeout_at(deadline.into(), receiver.recv()).await {
                    Ok(item) => item?,
                    Err(_) => Err(SpeechError::DeadlineExceeded),
                }
            };
            let ended = item.is_err();
            if ended {
                guard.0.cancel();
            }
            // A failure mid-stream aborts the body rather than looking complete.
            let item = item.map_err(|error| std::io::Error::other(describe(&error)));
            Some((item, (receiver, guard, ended)))
        },
    );
    let stream = futures::StreamExt::chain(
        futures::stream::iter(head.into_iter().map(Ok::<_, std::io::Error>)),
        rest,
    );
    let content_type = match request.format {
        Format::Wav => "audio/wav",
        Format::Pcm => "audio/pcm",
    };
    (
        [
            (header::CONTENT_TYPE, content_type.to_owned()),
            (
                header::HeaderName::from_static("x-sample-rate"),
                RATE.hz().to_string(),
            ),
        ],
        Body::from_stream(stream),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn error_text(body: &Value) -> String {
        match read(body, 10) {
            Ok(_) => "ok".into(),
            Err(response) => response.status().to_string(),
        }
    }

    #[test]
    fn requests_are_checked() {
        assert_eq!(error_text(&json!({"input": "hello"})), "ok");
        assert_eq!(error_text(&json!({})), "400 Bad Request");
        assert_eq!(error_text(&json!({"input": "  "})), "400 Bad Request");
        assert_eq!(
            error_text(&json!({"input": "01234567890"})),
            "400 Bad Request"
        );
        assert_eq!(
            error_text(&json!({"input": "hi", "response_format": "mp3"})),
            "400 Bad Request"
        );
        assert_eq!(
            error_text(&json!({"input": "hi", "speed": "x"})),
            "400 Bad Request"
        );
        assert_eq!(
            error_text(&json!({"input": "hi", "voice": 3})),
            "400 Bad Request"
        );
        let request = read(
            &json!({"input": "hi", "voice": "v", "speed": 1.5, "response_format": "pcm", "model": "m"}),
            10,
        )
        .ok()
        .unwrap();
        assert_eq!(request.format, Format::Pcm);
        assert_eq!(request.voice.as_deref(), Some("v"));
        assert!((request.speed - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn wav_header_is_streaming() {
        let header = streaming_wav_header(RATE);
        assert_eq!(header.len(), 44);
        assert_eq!(&header[..4], b"RIFF");
        assert_eq!(&header[4..8], &[0xFF; 4]);
        assert_eq!(&header[24..28], &24_000_u32.to_le_bytes());
        assert_eq!(&header[40..44], &[0xFF; 4]);
    }
}
