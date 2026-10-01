//! The wire format of OpenAI-compatible transcription: pure functions,
//! with no I/O.

use crate::SpeechError;
use reqwest::{
    Url,
    multipart::{Form, Part},
};
use serde_json::Value;

/// The backend name used in errors.
pub(crate) const BACKEND: &str = "openai-http";

fn protocol(message: impl Into<String>) -> SpeechError {
    SpeechError::backend(BACKEND, false, message.into())
}

/// The transcription URL for an API root such as `https://api.openai.com/v1`
/// or `https://proxy.example.com/openai/v1/`. `audio/transcriptions` is
/// appended to the path.
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] if the root is not an HTTP(S) URL, has
/// credentials, a query, or a fragment, or already ends in the endpoint.
pub(crate) fn endpoint(base_url: &str) -> Result<Url, SpeechError> {
    endpoint_for(base_url, "audio/transcriptions")
}

/// `base_url` with `path` appended, under the same rules as [`endpoint`].
pub(crate) fn endpoint_for(base_url: &str, path: &str) -> Result<Url, SpeechError> {
    let invalid = |why: &str| SpeechError::InvalidInput(format!("invalid API base URL: {why}"));
    let mut url = Url::parse(base_url).map_err(|_| invalid("not a URL"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(invalid("use an http:// or https:// URL with a host"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(invalid("credentials belong in the API key, not the URL"));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(invalid("remove the query and fragment"));
    }
    let root = url.path().trim_end_matches('/').to_owned();
    if root.ends_with(&format!("/{path}")) {
        return Err(invalid("give the API root, without the endpoint path"));
    }
    url.set_path(&format!("{root}/{path}"));
    Ok(url)
}

/// The multipart form for one upload. The answer is JSON, or with
/// `stream` server-sent events: `transcript.text.delta` events, then
/// `transcript.text.done`.
pub(crate) fn build_multipart(
    audio_wav: Vec<u8>,
    model: &str,
    language: Option<&str>,
    prompt: Option<&str>,
    stream: bool,
) -> Form {
    let part = Part::bytes(audio_wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .expect("audio/wav is a valid MIME type");
    let mut form = Form::new()
        .part("file", part)
        .text("model", model.to_owned())
        .text("response_format", "json");
    if stream {
        form = form.text("stream", "true");
    }
    if let Some(language) = language {
        form = form.text("language", language.to_owned());
    }
    if let Some(prompt) = prompt {
        form = form.text("prompt", prompt.to_owned());
    }
    form
}

/// The text of a JSON response.
///
/// # Errors
///
/// A non-retryable backend error if the body is not JSON with a string
/// `text` field.
pub(crate) fn parse_json(body: &[u8]) -> Result<String, SpeechError> {
    let value: Value =
        serde_json::from_slice(body).map_err(|_| protocol("the response is not valid JSON"))?;
    value
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| protocol("the JSON response has no text field"))
}

/// The error for a non-success status. The body is not echoed, since a
/// proxy might reflect request headers; only the error type and code of an
/// OpenAI-style error object are kept.
///
/// 408, 429, and 5xx are retryable; other statuses are not.
pub(crate) fn parse_error(status: u16, body: &[u8]) -> SpeechError {
    parse_error_for(BACKEND, status, body)
}

/// [`parse_error`] for another OpenAI endpoint, named `backend`.
pub(crate) fn parse_error_for(backend: &str, status: u16, body: &[u8]) -> SpeechError {
    let retryable = matches!(status, 408 | 429 | 500..=599);
    let mut message = format!("HTTP {status}");
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        let error = &value["error"];
        for key in ["type", "code"] {
            if let Some(text) = error.get(key).and_then(Value::as_str) {
                let text: String = text
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
                    .take(64)
                    .collect();
                message = format!("{message} {key}={text}");
            }
        }
    }
    SpeechError::backend(backend, retryable, message)
}

/// One server-sent event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct SseEvent {
    /// The `event:` field, if any.
    pub(crate) event: Option<String>,
    /// The `data:` lines, joined with `\n`.
    pub(crate) data: String,
}

/// Splits a byte stream into server-sent events, in any chunking.
///
/// It handles `\n`, `\r\n`, and `\r` line endings (including a `\r\n`
/// split across chunks), comment lines, and multi-line `data:` fields.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
    event: SseEvent,
    has_data: bool,
    /// The last byte seen was `\r`; a following `\n` belongs to it.
    after_cr: bool,
}

impl SseDecoder {
    /// A decoder at the start of a stream.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feeds bytes and returns the events they complete.
    ///
    /// # Errors
    ///
    /// A non-retryable backend error for a line that is not UTF-8.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, SpeechError> {
        let mut events = Vec::new();
        for &byte in bytes {
            if self.after_cr {
                self.after_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' => {
                    self.after_cr = true;
                    self.line(&mut events)?;
                }
                b'\n' => self.line(&mut events)?,
                _ => self.buffer.push(byte),
            }
        }
        Ok(events)
    }

    /// Ends the stream, returning an event left without a blank line.
    ///
    /// # Errors
    ///
    /// As [`push`](Self::push).
    pub(crate) fn finish(&mut self) -> Result<Vec<SseEvent>, SpeechError> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            self.line(&mut events)?;
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn line(&mut self, events: &mut Vec<SseEvent>) -> Result<(), SpeechError> {
        let line = std::mem::take(&mut self.buffer);
        if line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }
        let line = String::from_utf8(line).map_err(|_| protocol("the SSE stream is not UTF-8"))?;
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = line.split_once(':').unwrap_or((&line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "data" => {
                if self.has_data {
                    self.event.data.push('\n');
                }
                self.event.data.push_str(value);
                self.has_data = true;
            }
            "event" => self.event.event = Some(value.to_owned()),
            _ => {}
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if self.has_data {
            events.push(std::mem::take(&mut self.event));
        } else {
            self.event = SseEvent::default();
        }
        self.has_data = false;
    }
}

/// A transcription event carried by SSE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TranscriptEvent {
    /// Text to append.
    Delta(String),
    /// The complete text.
    Done(String),
}

/// Interprets one SSE event. Unknown event types and `[DONE]` give `None`.
///
/// # Errors
///
/// A non-retryable backend error for invalid JSON, a missing text field,
/// or an `error` event.
pub(crate) fn parse_event(event: &SseEvent) -> Result<Option<TranscriptEvent>, SpeechError> {
    if event.data.is_empty() || event.data == "[DONE]" {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(&event.data)
        .map_err(|_| protocol("an SSE event is not valid JSON"))?;
    let text = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| protocol(format!("an SSE event has no {key} field")))
    };
    match value.get("type").and_then(Value::as_str) {
        Some("transcript.text.delta") => Ok(Some(TranscriptEvent::Delta(text("delta")?))),
        Some("transcript.text.done") => Ok(Some(TranscriptEvent::Done(text("text")?))),
        Some("error") => Err(protocol("the server reported an error in the SSE stream")),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn endpoints() {
        let cases = [
            (
                "https://api.openai.com/v1",
                "https://api.openai.com/v1/audio/transcriptions",
            ),
            (
                "https://api.openai.com/v1/",
                "https://api.openai.com/v1/audio/transcriptions",
            ),
            (
                "http://localhost:8080",
                "http://localhost:8080/audio/transcriptions",
            ),
            (
                "https://proxy.example.com/openai/v1//",
                "https://proxy.example.com/openai/v1/audio/transcriptions",
            ),
        ];
        for (base, want) in cases {
            assert_eq!(endpoint(base).unwrap().as_str(), want, "{base}");
        }
        for bad in [
            "not a url",
            "ftp://example.com/v1",
            "https://user:pass@example.com/v1",
            "https://example.com/v1?key=1",
            "https://example.com/v1#x",
            "https://example.com/v1/audio/transcriptions",
        ] {
            assert!(
                matches!(endpoint(bad), Err(SpeechError::InvalidInput(_))),
                "{bad}"
            );
        }
    }

    #[test]
    fn forms() {
        let form = build_multipart(vec![1, 2], "whisper-1", Some("zh"), Some("hint"), true);
        assert!(form.boundary().len() > 10);
    }

    #[test]
    fn json_bodies() {
        assert_eq!(parse_json(br#"{"text":"hello"}"#).unwrap(), "hello");
        assert!(parse_json(br#"{"txt":"hello"}"#).is_err());
        assert!(parse_json(b"nope").is_err());
    }

    #[test]
    fn error_statuses() {
        for (status, retryable) in [
            (400, false),
            (401, false),
            (404, false),
            (408, true),
            (429, true),
            (500, true),
            (503, true),
        ] {
            assert_eq!(parse_error(status, b"").retryable(), retryable, "{status}");
        }
        let body = br#"{"error":{"message":"secret sk-123 leaked","type":"invalid_request_error","code":"bad key!"}}"#;
        let error = parse_error(401, body);
        let source = std::error::Error::source(&error).unwrap().to_string();
        assert_eq!(source, "HTTP 401 type=invalid_request_error code=badkey");
    }

    const STREAM: &str = ": keep-alive\r\n\r\nevent: transcript.text.delta\r\ndata: {\"type\":\"transcript.text.delta\",\"delta\":\"Hel\"}\r\n\r\ndata: {\"type\":\"transcript.text.delta\",\r\ndata: \"delta\":\"lo\"}\n\ndata: {\"type\":\"transcript.text.done\",\"text\":\"Hello\"}\r\rdata: [DONE]\n\n";

    fn decode(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.push(chunk).unwrap());
        }
        events.extend(decoder.finish().unwrap());
        events
    }

    #[test]
    fn sse_events() {
        let events = decode(&[STREAM.as_bytes()]);
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].event.as_deref(), Some("transcript.text.delta"));
        let parsed: Vec<_> = events.iter().map(|e| parse_event(e).unwrap()).collect();
        assert_eq!(
            parsed,
            [
                Some(TranscriptEvent::Delta("Hel".into())),
                Some(TranscriptEvent::Delta("lo".into())),
                Some(TranscriptEvent::Done("Hello".into())),
                None,
            ]
        );
    }

    #[test]
    fn sse_edge_cases() {
        assert_eq!(
            decode(&[b"data: x"]),
            [SseEvent {
                event: None,
                data: "x".into()
            }]
        );
        assert!(decode(&[b"event: only\n\n"]).is_empty());
        assert!(SseDecoder::new().push(b"data: \xFF\n").is_err());
        let error = parse_event(&SseEvent {
            event: None,
            data: r#"{"type":"error"}"#.into(),
        });
        assert!(error.is_err());
        assert!(
            parse_event(&SseEvent {
                event: None,
                data: "{".into()
            })
            .is_err()
        );
        assert!(
            parse_event(&SseEvent {
                event: None,
                data: r#"{"type":"transcript.text.done"}"#.into()
            })
            .is_err()
        );
        assert_eq!(
            parse_event(&SseEvent {
                event: None,
                data: r#"{"type":"other"}"#.into()
            })
            .unwrap(),
            None
        );
    }

    proptest! {
        #[test]
        fn any_split_gives_the_same_events(split in 0..STREAM.len()) {
            let bytes = STREAM.as_bytes();
            let whole = decode(&[bytes]);
            let (a, b) = bytes.split_at(split);
            prop_assert_eq!(decode(&[a, b]), whole.clone());
            let singles: Vec<&[u8]> = bytes.chunks(1).collect();
            prop_assert_eq!(decode(&singles), whole);
        }
    }
}
