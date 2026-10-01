//! What a server serves, and its settings.

use std::{sync::Arc, time::Duration};

use crate::{Secret, asr::AsrEngine, tts::TtsEngine};

/// An OpenAI-compatible server for a recognition engine, a synthesis
/// engine, or both.
///
/// Every setting has a default. Serve with [`run`](Self::run) or
/// [`start`](Self::start), or take the [`router`](Self::router) and serve
/// it yourself.
#[derive(Debug, Clone)]
pub struct Server {
    pub(crate) asr: Option<(AsrEngine, String)>,
    pub(crate) tts: Option<(TtsEngine, String)>,
    pub(crate) bind: String,
    pub(crate) max_body_bytes: usize,
    pub(crate) timeout: Option<Duration>,
    pub(crate) language: Option<String>,
    /// How long a client may take to send its request headers before the
    /// connection is closed: 30 s.
    pub(crate) header_timeout: Duration,
    /// How long a client may take to send its whole request body: 120 s,
    /// enough for a 25 MiB upload at about 2 Mbit/s.
    pub(crate) body_timeout: Duration,
    /// The longest `input` `/v1/audio/speech` accepts, in characters: 4096,
    /// OpenAI's documented limit.
    pub(crate) max_speech_chars: usize,
    pub(crate) auth: Option<Arc<Secret>>,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            asr: None,
            tts: None,
            bind: "127.0.0.1:8080".into(),
            max_body_bytes: 25 * 1024 * 1024,
            timeout: None,
            language: None,
            header_timeout: Duration::from_secs(30),
            body_timeout: Duration::from_secs(120),
            max_speech_chars: 4096,
            auth: None,
        }
    }
}

impl Server {
    /// A server with no engines yet, and the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serves `engine` on `POST /v1/audio/transcriptions`, reported as
    /// `model_id` by `/v1/models`. Without one, that route answers 404.
    #[must_use]
    pub fn with_asr(mut self, engine: AsrEngine, model_id: impl Into<String>) -> Self {
        self.asr = Some((engine, model_id.into()));
        self
    }

    /// Serves `engine` on `POST /v1/audio/speech`, reported as `model_id`
    /// by `/v1/models`. Without one, that route answers 404.
    #[must_use]
    pub fn with_tts(mut self, engine: TtsEngine, model_id: impl Into<String>) -> Self {
        self.tts = Some((engine, model_id.into()));
        self
    }

    /// Where [`start`](Self::start) and [`run`](Self::run) listen, such as
    /// `0.0.0.0:8080`. Default: `127.0.0.1:8080`.
    #[must_use]
    pub fn with_bind(mut self, address: impl Into<String>) -> Self {
        self.bind = address.into();
        self
    }

    /// The largest request body, enforced while it streams in. Default:
    /// 25 MiB, OpenAI's own limit.
    #[must_use]
    pub fn with_max_body_bytes(mut self, bytes: usize) -> Self {
        self.max_body_bytes = bytes;
        self
    }

    /// The time allowed for one transcription or synthesis.
    ///
    /// Default: `max(120 s, 3 × audio length)`; for synthesis the length is
    /// estimated at half a second per character. Transcription timing
    /// starts before the upload and decoding, with 120 s allowed until the
    /// audio's length is known. Opening the session and waiting for a slow
    /// reader of synthesized audio share the budget.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The language used when a transcription request names none.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Requires `Authorization: Bearer <token>` on the `/v1/` routes; `/health`
    /// stays open. The token is a [`Secret`], or an `Arc<Secret>` shared with
    /// other code.
    #[must_use]
    pub fn with_auth(mut self, token: impl Into<Arc<Secret>>) -> Self {
        self.auth = Some(token.into());
        self
    }

    /// The time allowed for audio of `length`.
    pub(crate) fn budget(&self, length: Duration) -> Duration {
        self.timeout
            .unwrap_or_else(|| (length * 3).max(Duration::from_secs(120)))
    }
}
