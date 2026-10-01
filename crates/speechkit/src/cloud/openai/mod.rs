//! OpenAI and OpenAI-compatible services.

pub(crate) mod http;
pub(crate) mod realtime;
pub(crate) mod speech;

pub use http::{OpenAiTranscription, OpenAiTranscriptionConfig};
pub use realtime::{OpenAiRealtime, OpenAiRealtimeConfig};
pub use speech::{OpenAiSpeech, OpenAiSpeechConfig};

/// The OpenAI API root, the default endpoint of the HTTP backends.
const OPENAI_API: &str = "https://api.openai.com/v1";

/// Maps a reqwest failure for `backend`. Network trouble is retryable.
fn transport(backend: &str, error: &reqwest::Error) -> crate::SpeechError {
    let message = if error.is_timeout() {
        "the request timed out"
    } else if error.is_connect() {
        "could not connect"
    } else if error.is_body() || error.is_decode() {
        "the response body could not be read"
    } else if error.is_builder() {
        return crate::SpeechError::backend(backend, false, "the request could not be built");
    } else {
        "the request failed"
    };
    crate::SpeechError::backend(backend, true, message)
}
