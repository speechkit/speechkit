//! `POST /v1/audio/transcriptions`.

use std::{
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    AudioBuffer, RecvError, SpeechError,
    asr::{AsrOptions, AsrResult, AsrSession, AsrUpdate, AsrUpdates, LiveTranscript},
    audio::{self, DecodeLimits},
    speech::{audio::find_invalid, deadline::instant_after, opening::Opening},
};
use axum::{
    Json,
    body::Bytes,
    extract::{
        Multipart, State,
        multipart::{MultipartError, MultipartRejection},
    },
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde_json::json;

use crate::server::{
    delta::TextDeltaBuilder,
    error::{describe, respond, slot_busy, status_of},
    state::Shared,
};

/// Samples per push: 100 ms at 16 kHz, less at lower rates.
const PUSH: Duration = Duration::from_millis(100);

/// Cancels the session when dropped: a client that disconnects drops the
/// handler future or the response stream, and with it this guard.
struct CancelOnDrop {
    session: Arc<AsrSession>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.session.cancel();
    }
}

struct Request {
    file: Bytes,
    language: Option<String>,
    text: bool,
    stream: bool,
}

async fn read(mut multipart: Multipart) -> Result<Request, Box<Response>> {
    let failed = |error: MultipartError| {
        let status = error.status();
        Box::new(if status == StatusCode::PAYLOAD_TOO_LARGE {
            respond(status, "the request body exceeds the server's size limit")
        } else {
            respond(status, format!("invalid multipart request: {error}"))
        })
    };
    let mut file = None;
    let mut language = None;
    let mut format = "json".to_owned();
    let mut stream = false;
    while let Some(field) = multipart.next_field().await.map_err(failed)? {
        match field.name() {
            Some("file") => file = Some(field.bytes().await.map_err(failed)?),
            Some("language") => {
                let text = field.text().await.map_err(failed)?;
                language = Some(text.trim().to_owned()).filter(|t| !t.is_empty());
            }
            Some("response_format") => format = field.text().await.map_err(failed)?,
            Some("stream") => {
                stream = matches!(field.text().await.map_err(failed)?.trim(), "true" | "1");
            }
            // `model` is accepted and ignored: the server runs the engine it
            // was started with. Other fields have no local meaning.
            _ => {}
        }
    }
    let file =
        file.ok_or_else(|| respond(StatusCode::BAD_REQUEST, "the `file` field is required"))?;
    let text = match format.as_str() {
        "json" => false,
        "text" => true,
        other => {
            return Err(Box::new(respond(
                StatusCode::BAD_REQUEST,
                format!("unsupported response_format {other:?}; use json or text"),
            )));
        }
    };
    if stream && text {
        return Err(Box::new(respond(
            StatusCode::BAD_REQUEST,
            "stream=true needs response_format json",
        )));
    }
    Ok(Request {
        file,
        language,
        text,
        stream,
    })
}

/// WebM carries Opus, which cannot be decoded here: the EBML magic plus a
/// `webm` `DocType` element.
fn is_webm(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3])
        && bytes[..bytes.len().min(128)]
            .windows(7)
            .any(|w| w == [0x42, 0x82, 0x84, b'w', b'e', b'b', b'm'])
}

async fn decode(
    file: Bytes,
    permit: tokio::sync::OwnedSemaphorePermit,
    deadline: Instant,
) -> Result<(AudioBuffer, tokio::sync::OwnedSemaphorePermit), Box<Response>> {
    if is_webm(&file) {
        return Err(Box::new(respond(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "WebM audio is Opus, which is not supported; transcode to WAV, FLAC, or MP3 first",
        )));
    }
    // The blocking decoder retains admission even if its caller times out
    // or disconnects: detached work must still count against the limit.
    let task = tokio::task::spawn_blocking(move || {
        (audio::decode(&file, DecodeLimits::default()), permit)
    });
    let (decoded, permit) = tokio::time::timeout_at(deadline.into(), task)
        .await
        .map_err(|_| Box::new(respond(StatusCode::GATEWAY_TIMEOUT, "decoding timed out")))?
        .map_err(|e| Box::new(respond(StatusCode::INTERNAL_SERVER_ERROR, e)))?;
    let failure = match decoded {
        Ok(buffer) if !buffer.samples.is_empty() => return Ok((buffer, permit)),
        Ok(_) => respond(StatusCode::BAD_REQUEST, "the file contains no audio"),
        Err(error) if error.to_string().contains("Opus") => {
            respond(StatusCode::UNSUPPORTED_MEDIA_TYPE, describe(&error))
        }
        Err(error) => respond(StatusCode::BAD_REQUEST, describe(&error)),
    };
    Err(Box::new(failure))
}

/// Pushes the audio and waits for the result.
///
/// `handle` has checked the samples and `max_chunk`, the input queue,
/// bounds the pieces, so
/// a push fails only when the deadline passes or the session ends, and
/// `finish` reports either.
fn feed(
    session: &AsrSession,
    audio: &AudioBuffer,
    max_chunk: Duration,
    deadline: Instant,
) -> AsrResult {
    let rate = audio.sample_rate;
    let chunk = usize::try_from(rate.frames_in(PUSH.min(max_chunk)))
        .unwrap_or(1_600)
        .max(1);
    for piece in audio.samples.chunks(chunk) {
        if session.push(piece, deadline).is_err() {
            break;
        }
    }
    session.finish(deadline)
}

pub(crate) async fn handle(
    State(state): State<Shared>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let started = Instant::now();
    let Some((engine, _)) = &state.server.asr else {
        return respond(
            StatusCode::NOT_FOUND,
            "this server has no transcription engine",
        );
    };
    let initial_deadline = instant_after(started, state.server.budget(Duration::ZERO));
    if engine.active_sessions() >= engine.max_sessions() {
        return slot_busy();
    }
    let Ok(permit) = state.admission.clone().try_acquire_owned() else {
        return respond(
            StatusCode::SERVICE_UNAVAILABLE,
            "every transcription request slot is busy; retry later",
        );
    };
    let multipart = match multipart {
        Ok(multipart) => multipart,
        Err(rejection) => return respond(StatusCode::BAD_REQUEST, rejection.body_text()),
    };
    let Ok(request) = tokio::time::timeout_at(initial_deadline.into(), read(multipart)).await
    else {
        return respond(StatusCode::GATEWAY_TIMEOUT, "upload timed out");
    };
    let request = match request {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let (audio, permit) = match decode(request.file, permit, initial_deadline).await {
        Ok(decoded) => decoded,
        Err(response) => return *response,
    };
    // Checked here, so a bad sample is a 400 rather than a transcript cut
    // short where it was refused.
    if let Some(invalid) = find_invalid(&audio.samples) {
        return respond(StatusCode::BAD_REQUEST, invalid);
    }
    if request.stream && !engine.capabilities().reports_partials {
        return respond(
            StatusCode::BAD_REQUEST,
            "this engine reports no partial results, so stream=true is not supported",
        );
    }
    let options = AsrOptions {
        language: request.language.or_else(|| state.server.language.clone()),
        ..AsrOptions::default()
    };
    let deadline = instant_after(started, state.server.budget(audio.duration()));
    let opening = engine.clone();
    let rate = audio.sample_rate;
    let task = tokio::task::spawn_blocking(move || {
        let opened = opening.start_with(
            rate,
            options,
            deadline.into(),
            Opening::NoWait,
            Duration::ZERO,
        );
        (opened, permit)
    });
    let (opened, permit) = match tokio::time::timeout_at(deadline.into(), task).await {
        Ok(Ok(opened)) => opened,
        Ok(Err(error)) => return respond(StatusCode::INTERNAL_SERVER_ERROR, error),
        Err(_) => return respond(StatusCode::GATEWAY_TIMEOUT, "initialization timed out"),
    };
    let session = match opened {
        Ok(session) => Arc::new(session),
        Err(SpeechError::Capacity) => {
            tracing::warn!(
                active_sessions = engine.active_sessions(),
                max_sessions = engine.max_sessions(),
                "every session slot is busy; answering 503"
            );
            return slot_busy();
        }
        Err(error) => return respond(status_of(&error), describe(&error)),
    };
    let max_chunk = engine.limits().input_queue;
    if request.stream {
        return stream(session, audio, max_chunk, deadline, permit);
    }
    let guard = CancelOnDrop {
        session: session.clone(),
        _permit: permit,
    };
    let result =
        tokio::task::spawn_blocking(move || feed(&session, &audio, max_chunk, deadline)).await;
    drop(guard);
    match result {
        Ok(result) => match result.as_ref() {
            Ok(transcript) if request.text => transcript.text().into_response(),
            Ok(transcript) => Json(json!({ "text": transcript.text() })).into_response(),
            Err(failure) => respond(status_of(&failure.error), describe(&failure.error)),
        },
        Err(join) => respond(StatusCode::INTERNAL_SERVER_ERROR, join),
    }
}

fn event(kind: &str, mut payload: serde_json::Value) -> Event {
    if let Some(object) = payload.as_object_mut() {
        object.insert("type".into(), json!(kind));
    }
    Event::default().event(kind).data(payload.to_string())
}

/// Reads the updates on a blocking thread and forwards SSE events. A
/// closed channel means the client left: the session is cancelled.
fn forward(
    mut updates: AsrUpdates,
    session: &AsrSession,
    events: &tokio::sync::mpsc::Sender<Event>,
) {
    let mut view = LiveTranscript::new();
    let mut deltas = TextDeltaBuilder::new();
    loop {
        let update = match updates.recv(Duration::from_millis(250)) {
            Ok(update) => update,
            Err(RecvError::Closed) => return,
            Err(RecvError::Timeout | RecvError::Empty) => {
                if events.is_closed() {
                    session.cancel();
                }
                continue;
            }
        };
        view.apply(&update);
        let next = match &update {
            AsrUpdate::Closed(result) => Some(match result.as_ref() {
                Ok(transcript) => {
                    event("transcript.text.done", json!({ "text": transcript.text() }))
                }
                Err(failure) => event("error", json!({ "message": describe(&failure.error) })),
            }),
            _ => deltas
                .update(&view.text())
                .map(|delta| event("transcript.text.delta", json!({ "delta": delta }))),
        };
        if let Some(next) = next
            && events.blocking_send(next).is_err()
        {
            session.cancel();
        }
        if matches!(update, AsrUpdate::Closed(_)) {
            return;
        }
    }
}

fn stream(
    session: Arc<AsrSession>,
    audio: AudioBuffer,
    max_chunk: Duration,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let updates = session.updates();
    let (sender, receiver) = tokio::sync::mpsc::channel(32);
    let feeder = session.clone();
    tokio::task::spawn_blocking(move || drop(feed(&feeder, &audio, max_chunk, deadline)));
    let reader = session.clone();
    tokio::task::spawn_blocking(move || forward(updates, &reader, &sender));
    let guard = CancelOnDrop {
        session,
        _permit: permit,
    };
    let events = futures::stream::unfold((receiver, guard), |(mut receiver, guard)| async move {
        let event = receiver.recv().await?;
        Some((Ok::<_, Infallible>(event), (receiver, guard)))
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webm_detection() {
        let mut webm = vec![0x1A, 0x45, 0xDF, 0xA3, 0x9F];
        webm.extend_from_slice(&[0x42, 0x82, 0x84, b'w', b'e', b'b', b'm']);
        assert!(is_webm(&webm));
        let mut matroska = vec![0x1A, 0x45, 0xDF, 0xA3, 0x9F];
        matroska.extend_from_slice(&[0x42, 0x82, 0x88]);
        matroska.extend_from_slice(b"matroska");
        assert!(!is_webm(&matroska));
        assert!(!is_webm(b"RIFF"));
    }
}
