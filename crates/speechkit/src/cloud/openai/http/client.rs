//! The HTTP transport for OpenAI-compatible transcription.

use std::{sync::Arc, time::Duration};

use crate::{
    Flow, SampleRate, Secret, SpeechError,
    asr::{
        AsrBackend, AsrCapabilities, AsrEvent, AsrEvents, AsrOptions, AsrStream, Partial, Segment,
        UtteranceId,
    },
    audio::encode_samples,
    vad::OfflineRecognizer,
};
use reqwest::Url;

use super::protocol::{self, BACKEND, SseDecoder, TranscriptEvent, build_multipart, endpoint};
use crate::cloud::{
    CloudRuntime,
    openai::{OPENAI_API, transport},
};

/// Bytes of a PCM16 WAV header.
const WAV_HEADER: usize = 44;

/// Settings for [`OpenAiTranscription`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OpenAiTranscriptionConfig {
    /// The API root. Default: `https://api.openai.com/v1`. It may carry a
    /// proxy prefix; `audio/transcriptions` is appended.
    pub endpoint: String,
    /// The model name sent with each request.
    pub model: String,
    /// Sent as a bearer token. `None` sends no `Authorization` header.
    pub api_key: Option<Arc<Secret>>,
    /// Asks the server to stream the transcript as server-sent events,
    /// reported as partial results once the upload finishes. Default:
    /// false, one JSON answer.
    pub streaming: bool,
    /// The default language, overridden by a session's language.
    pub language: Option<String>,
    /// A prompt sent with every request.
    pub prompt: Option<String>,
    /// The limit for one whole request, upload to last byte. Default: 60 s.
    pub timeout: Duration,
    /// The largest WAV upload: 24 MiB, OpenAI's 25 MB limit with room to
    /// spare, or about 13 minutes at 16 kHz. Longer audio is rejected
    /// before anything is sent.
    pub(crate) max_upload_bytes: usize,
    /// The largest response body: 1 MiB.
    pub(crate) max_response_bytes: usize,
}

impl OpenAiTranscriptionConfig {
    /// Settings for `model` on OpenAI, sending no key. Add one with
    /// [`with_api_key`](Self::with_api_key), or point
    /// [`with_endpoint`](Self::with_endpoint) at a compatible server; a
    /// local one such as `speechkit serve` needs no key.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            endpoint: OPENAI_API.into(),
            model: model.into(),
            api_key: None,
            streaming: false,
            language: None,
            prompt: None,
            timeout: Duration::from_secs(60),
            max_upload_bytes: 24 * 1024 * 1024,
            max_response_bytes: 1024 * 1024,
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

    /// Sets whether the transcript is streamed.
    #[must_use]
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Sets the default language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

struct Inner {
    config: OpenAiTranscriptionConfig,
    url: Url,
    client: reqwest::Client,
    runtime: CloudRuntime,
    caps: AsrCapabilities,
}

/// Transcription through `POST {endpoint}/audio/transcriptions`, on OpenAI
/// or any compatible server.
///
/// - As an [`AsrBackend`], it buffers a whole session and uploads it at
///   the end. With `streaming` set it reports the deltas as
///   partial results, but only once the upload finishes.
/// - As an [`OfflineRecognizer`], wrapped in a `VadBackend`, it uploads
///   each detected utterance on its own.
///
/// Audio is sent as 16 kHz PCM16 WAV.
#[derive(Clone)]
pub struct OpenAiTranscription {
    inner: Arc<Inner>,
}

impl OpenAiTranscription {
    /// A backend with `config`, running requests on `runtime`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad endpoint, an empty model, or
    /// a zero timeout.
    pub fn new(
        config: OpenAiTranscriptionConfig,
        runtime: CloudRuntime,
    ) -> Result<Self, SpeechError> {
        let url = endpoint(&config.endpoint)?;
        if config.model.trim().is_empty() {
            return Err(SpeechError::InvalidInput("the model name is empty".into()));
        }
        if config.timeout.is_zero() {
            return Err(SpeechError::InvalidInput(
                "the timeout must be positive".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| SpeechError::backend(BACKEND, false, e))?;
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.reports_partials = config.streaming;
        caps.accepts_language = true;
        caps.punctuated = true;
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

    fn check_size(&self, samples: usize) -> Result<(), SpeechError> {
        let bytes = WAV_HEADER.saturating_add(samples.saturating_mul(2));
        if bytes > self.inner.config.max_upload_bytes {
            return Err(SpeechError::InvalidInput(format!(
                "the audio needs a {bytes}-byte upload, over the {}-byte limit",
                self.inner.config.max_upload_bytes
            )));
        }
        Ok(())
    }

    /// Uploads `samples` (16 kHz), publishing partials without retaining them.
    fn upload(
        &self,
        samples: &[f32],
        language: Option<&str>,
        partial: &mut dyn FnMut(String) -> Result<(), SpeechError>,
    ) -> Result<String, SpeechError> {
        self.check_size(samples.len())?;
        let inner = &*self.inner;
        let wav = encode_samples(SampleRate::HZ_16000, samples)?;
        let config = &inner.config;
        let language = language.or(config.language.as_deref());
        let form = build_multipart(
            wav,
            &config.model,
            language,
            config.prompt.as_deref(),
            config.streaming,
        );
        let mut request = inner.client.post(inner.url.clone()).multipart(form);
        if let Some(key) = &config.api_key {
            request = request.bearer_auth(key.expose());
        }
        tracing::debug!(
            backend = BACKEND,
            samples = samples.len(),
            "uploading audio"
        );
        let timeout = config.timeout;
        inner.runtime.block_on(async move {
            match tokio::time::timeout(timeout, Self::exchange(request, config, partial)).await {
                Ok(result) => result,
                Err(_) => Err(SpeechError::backend(BACKEND, true, "the request timed out")),
            }
        })
    }

    async fn exchange(
        request: reqwest::RequestBuilder,
        config: &OpenAiTranscriptionConfig,
        partial: &mut dyn FnMut(String) -> Result<(), SpeechError>,
    ) -> Result<String, SpeechError> {
        let mut response = request.send().await.map_err(|e| transport(BACKEND, &e))?;
        let status = response.status();
        let mut body = Vec::new();
        let mut decoder = SseDecoder::new();
        let mut received = 0_usize;
        let mut text = String::new();
        let sse = status.is_success() && config.streaming;
        while let Some(chunk) = response.chunk().await.map_err(|e| transport(BACKEND, &e))? {
            received = received.saturating_add(chunk.len());
            if received > config.max_response_bytes {
                return Err(SpeechError::backend(
                    BACKEND,
                    false,
                    "the response exceeds max_response_bytes",
                ));
            }
            if sse {
                for event in decoder.push(&chunk)? {
                    match protocol::parse_event(&event)? {
                        Some(TranscriptEvent::Delta(delta)) => {
                            text.push_str(&delta);
                            partial(text.clone())?;
                        }
                        Some(TranscriptEvent::Done(done)) => return Ok(done),
                        None => {}
                    }
                }
            }
            if !sse {
                body.extend_from_slice(&chunk);
            }
        }
        if !status.is_success() {
            return Err(protocol::parse_error(status.as_u16(), &body));
        }
        if !config.streaming {
            return protocol::parse_json(&body);
        }
        for event in decoder.finish()? {
            if let Some(TranscriptEvent::Done(done)) = protocol::parse_event(&event)? {
                return Ok(done);
            }
        }
        Err(SpeechError::backend(
            BACKEND,
            true,
            "the SSE stream ended without transcript.text.done",
        ))
    }
}

impl OfflineRecognizer for OpenAiTranscription {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.inner.caps
    }

    fn recognize(&self, samples: &[f32], opts: &AsrOptions) -> Result<String, SpeechError> {
        self.upload(samples, opts.language.as_deref(), &mut |_| Ok(()))
    }
}

impl AsrBackend for OpenAiTranscription {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.inner.caps
    }

    fn open(&self, opts: &AsrOptions, sink: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        Ok(Box::new(WholeRecording {
            sink,
            backend: self.clone(),
            language: opts.language.clone(),
            samples: Vec::new(),
        }))
    }
}

impl std::fmt::Debug for OpenAiTranscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiTranscription")
            .field("url", &self.inner.url.as_str())
            .field("model", &self.inner.config.model)
            .finish_non_exhaustive()
    }
}

/// Buffers a whole session and uploads it on `finish`.
struct WholeRecording {
    sink: AsrEvents,
    backend: OpenAiTranscription,
    language: Option<String>,
    samples: Vec<f32>,
}

impl AsrStream for WholeRecording {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.backend
            .check_size(self.samples.len().saturating_add(samples.len()))?;
        self.samples.extend_from_slice(samples);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        let samples = std::mem::take(&mut self.samples);
        if samples.is_empty() {
            return Ok(());
        }
        let utterance = UtteranceId(0);
        let sink = &self.sink;
        let mut had_partial = false;
        let text = self
            .backend
            .upload(&samples, self.language.as_deref(), &mut |text| {
                had_partial = true;
                match sink.send(AsrEvent::Partial(Partial { utterance, text })) {
                    Flow::Continue => Ok(()),
                    Flow::Stop => Err(SpeechError::Cancelled),
                }
            })?;
        let text = text.trim().to_owned();
        if !text.is_empty() || had_partial {
            sink.send(AsrEvent::Segment(Segment {
                utterance,
                text,
                start: Duration::ZERO,
                end: SampleRate::HZ_16000.duration_of(samples.len() as u64),
            }));
        }
        Ok(())
    }
}
