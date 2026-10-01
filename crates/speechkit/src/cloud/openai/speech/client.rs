//! The HTTP transport for OpenAI-compatible speech synthesis.

use std::{sync::Arc, time::Duration};

use crate::{
    Flow, SampleRate, Secret, SpeechError,
    tts::{TtsBackend, TtsCapabilities, TtsOptions, TtsStream, Voice},
};
use reqwest::Url;

use super::protocol::{BACKEND, request_body, speech_endpoint};
use crate::cloud::{
    CloudRuntime,
    openai::{OPENAI_API, http::protocol::parse_error_for, transport},
    pcm::Pcm16Decoder,
};

/// The voices OpenAI documents for its speech models.
const OPENAI_VOICES: &[&str] = &[
    "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
];

/// The largest error body read, in bytes.
const MAX_ERROR_BODY: usize = 64 * 1024;

/// Settings for [`OpenAiSpeech`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OpenAiSpeechConfig {
    /// The API root. Default: `https://api.openai.com/v1`. It may carry a
    /// proxy prefix; `audio/speech` is appended.
    pub endpoint: String,
    /// The model name sent with each request, for example `gpt-4o-mini-tts`.
    pub model: String,
    /// Sent as a bearer token. `None` sends no `Authorization` header.
    pub api_key: Option<Arc<Secret>>,
    /// The voices offered; the first is the default. Default: the voices
    /// OpenAI documents, `alloy` first.
    pub voices: Vec<Voice>,
    /// The rate of the PCM the server returns. Default: 24 kHz, as OpenAI
    /// documents.
    pub sample_rate: SampleRate,
    /// The longest text one request carries, in characters: 4096, OpenAI's
    /// documented limit.
    pub(crate) max_input_chars: usize,
    /// The limit for connecting and for each wait on the response body.
    /// Time spent waiting for the session to take audio is not counted.
    /// Default: 30 s.
    pub timeout: Duration,
}

impl OpenAiSpeechConfig {
    /// Settings for `model` on OpenAI, sending no key. Add one with
    /// [`with_api_key`](Self::with_api_key), or point
    /// [`with_endpoint`](Self::with_endpoint) at a compatible server.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            endpoint: OPENAI_API.into(),
            model: model.into(),
            api_key: None,
            voices: OPENAI_VOICES.iter().map(|id| Voice::new(*id)).collect(),
            sample_rate: SampleRate::HZ_24000,
            max_input_chars: 4096,
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets the API root, such as `http://127.0.0.1:8080/v1`.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sends `api_key` as a bearer token: a [`Secret`], or an
    /// `Arc<Secret>` shared with other clients.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<Arc<Secret>>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the voices; the first is the default.
    #[must_use]
    pub fn with_voices(mut self, voices: Vec<Voice>) -> Self {
        self.voices = voices;
        self
    }

    /// Sets the rate of the returned PCM.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: SampleRate) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Sets the timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

struct Inner {
    config: OpenAiSpeechConfig,
    url: Url,
    client: reqwest::Client,
    runtime: CloudRuntime,
    caps: TtsCapabilities,
}

/// Speech synthesis through `POST {endpoint}/audio/speech`, on OpenAI or
/// any compatible server, including `speechkit serve`.
///
/// Each chunk of text is one request for raw PCM, and the body is fed to
/// the session as it arrives. When the session's output queue is full,
/// reading stops, so the server sees TCP backpressure.
#[derive(Clone)]
pub struct OpenAiSpeech {
    inner: Arc<Inner>,
}

impl OpenAiSpeech {
    /// A backend with `config`, running requests on `runtime`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad endpoint, an empty model,
    /// no voices, or a zero timeout.
    pub fn new(config: OpenAiSpeechConfig, runtime: CloudRuntime) -> Result<Self, SpeechError> {
        let url = speech_endpoint(&config.endpoint)?;
        let invalid = |message: &str| Err(SpeechError::InvalidInput(message.into()));
        if config.model.trim().is_empty() {
            return invalid("the model name is empty");
        }
        if config.voices.is_empty() {
            return invalid("the voice list is empty");
        }
        if config.timeout.is_zero() {
            return invalid("the timeout must be positive");
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(config.timeout)
            .build()
            .map_err(|e| SpeechError::backend(BACKEND, false, e))?;
        let mut caps = TtsCapabilities::new(config.sample_rate, config.max_input_chars);
        caps.streams_audio = true;
        caps.speed = Some(0.25..=4.0);
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                url,
                client,
                runtime,
                caps,
            }),
        })
    }
}

impl std::fmt::Debug for OpenAiSpeech {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiSpeech")
            .field("url", &self.inner.url.as_str())
            .field("model", &self.inner.config.model)
            .finish_non_exhaustive()
    }
}

impl TtsBackend for OpenAiSpeech {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn capabilities(&self) -> &TtsCapabilities {
        &self.inner.caps
    }

    fn voices(&self) -> &[Voice] {
        &self.inner.config.voices
    }

    fn open(&self, opts: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
        let voice = match &opts.voice {
            Some(voice) => voice.clone(),
            None => self.inner.config.voices[0].id.clone(),
        };
        Ok(Box::new(Stream {
            inner: self.inner.clone(),
            voice,
            speed: opts.speed,
        }))
    }
}

struct Stream {
    inner: Arc<Inner>,
    voice: String,
    speed: f32,
}

fn timed_out() -> SpeechError {
    SpeechError::backend(BACKEND, true, "the request timed out")
}

impl Stream {
    async fn exchange(
        &self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<(), SpeechError> {
        let inner = &*self.inner;
        let config = &inner.config;
        let mut request = inner
            .client
            .post(inner.url.clone())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(request_body(&config.model, &self.voice, chunk, self.speed));
        if let Some(key) = &config.api_key {
            request = request.bearer_auth(key.expose());
        }
        let mut response = tokio::time::timeout(config.timeout, request.send())
            .await
            .map_err(|_| timed_out())?
            .map_err(|e| transport(BACKEND, &e))?;
        let status = response.status();
        if !status.is_success() {
            let mut body = Vec::new();
            while body.len() < MAX_ERROR_BODY {
                match tokio::time::timeout(config.timeout, response.chunk()).await {
                    Ok(Ok(Some(piece))) => body.extend_from_slice(&piece),
                    _ => break,
                }
            }
            return Err(parse_error_for(BACKEND, status.as_u16(), &body));
        }
        let mut decoder = Pcm16Decoder::new(BACKEND);
        loop {
            let piece = tokio::time::timeout(config.timeout, response.chunk())
                .await
                .map_err(|_| timed_out())?
                .map_err(|e| transport(BACKEND, &e))?;
            let Some(piece) = piece else {
                return decoder.finish();
            };
            let samples = decoder.push(&piece);
            if !samples.is_empty() && sink(&samples) == Flow::Stop {
                return Ok(());
            }
        }
    }
}

impl TtsStream for Stream {
    fn synthesize(
        &mut self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<(), SpeechError> {
        if chunk.trim().is_empty() {
            return Ok(());
        }
        tracing::debug!(
            backend = BACKEND,
            chars = chunk.chars().count(),
            "requesting speech"
        );
        self.inner.runtime.block_on(self.exchange(chunk, sink))
    }
}
