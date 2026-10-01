//! Pure functions for `POST /v1/audio/speech`: the endpoint and the
//! request body.

use crate::SpeechError;
use reqwest::Url;
use serde_json::json;

use crate::cloud::openai::http::protocol::endpoint_for;

/// The backend name in errors and logs.
pub(crate) const BACKEND: &str = "openai-speech";

/// `{base_url}/audio/speech`, under the same rules as
/// [`endpoint`](crate::cloud::openai::http::protocol::endpoint).
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] for a URL that is not a plain http(s) API
/// root.
pub(crate) fn speech_endpoint(base_url: &str) -> Result<Url, SpeechError> {
    endpoint_for(base_url, "audio/speech")
}

/// The JSON body of one request. Audio is always asked for as raw PCM:
/// 16-bit signed little-endian mono.
pub(crate) fn request_body(model: &str, voice: &str, input: &str, speed: f32) -> Vec<u8> {
    let body = json!({
        "model": model,
        "voice": voice,
        "input": input,
        "speed": speed,
        "response_format": "pcm",
    });
    body.to_string().into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_names_every_field() {
        let body: serde_json::Value =
            serde_json::from_slice(&request_body("tts-1", "alloy", "Hi \"there\"", 1.5)).unwrap();
        assert_eq!(body["model"], "tts-1");
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["input"], "Hi \"there\"");
        assert_eq!(body["speed"], 1.5);
        assert_eq!(body["response_format"], "pcm");
    }

    #[test]
    fn endpoint_appends_the_path() {
        assert_eq!(
            speech_endpoint("https://api.openai.com/v1/")
                .unwrap()
                .as_str(),
            "https://api.openai.com/v1/audio/speech"
        );
        assert!(speech_endpoint("https://api.openai.com/v1/audio/speech").is_err());
    }
}
