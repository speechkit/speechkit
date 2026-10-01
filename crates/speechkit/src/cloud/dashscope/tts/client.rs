//! The DashScope synthesis transport.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    Flow, SampleRate, Secret, SpeechError,
    speech::deadline::instant_after,
    tts::{TtsBackend, TtsCapabilities, TtsOptions, TtsStream, Voice},
};

use super::protocol::{self, BACKEND, Event};
use crate::cloud::{
    CloudRuntime,
    pcm::Pcm16Decoder,
    ws::{Incoming, Ws, check_session},
};

/// Settings for [`DashScopeTts`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DashScopeTtsConfig {
    /// The endpoint. Default: the mainland China endpoint.
    pub endpoint: String,
    /// The model, for example `cosyvoice-v2`.
    pub model: String,
    /// Sent as a bearer token.
    pub api_key: Arc<Secret>,
    /// The voices offered; the first is the default. A voice must suit the
    /// model, for example `longxiaochun_v2` for `cosyvoice-v2`.
    pub voices: Vec<Voice>,
    /// The rate requested from the server. Default: 22.05 kHz.
    pub sample_rate: SampleRate,
    /// The limit for connecting and for each wait on the server.
    /// Default: 30 s.
    pub timeout: Duration,
}

impl DashScopeTtsConfig {
    /// The mainland China endpoint. Singapore workspaces use
    /// `wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/`.
    pub const DEFAULT_ENDPOINT: &'static str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/";

    /// Settings for `model` speaking with `voices`.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<Arc<Secret>>,
        voices: Vec<Voice>,
    ) -> Self {
        Self {
            endpoint: Self::DEFAULT_ENDPOINT.into(),
            model: model.into(),
            api_key: api_key.into(),
            voices,
            sample_rate: SampleRate::HZ_22050,
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets the endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets the sample rate.
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

/// Speech synthesis on DashScope. Each session holds one WebSocket, and
/// each chunk of text is one task on it; audio streams as PCM16 frames.
#[derive(Clone)]
pub struct DashScopeTts {
    config: Arc<DashScopeTtsConfig>,
    runtime: CloudRuntime,
    caps: TtsCapabilities,
}

/// The longest text sent in one task, in characters.
const MAX_INPUT_CHARS: usize = 2_000;

impl DashScopeTts {
    /// A backend with `config`, running on `runtime`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad endpoint, an empty model or
    /// key, no voices, or a zero timeout.
    pub fn new(config: DashScopeTtsConfig, runtime: CloudRuntime) -> Result<Self, SpeechError> {
        check_session(
            &config.endpoint,
            &config.model,
            &config.api_key,
            config.timeout,
        )?;
        if config.voices.is_empty() {
            return Err(SpeechError::InvalidInput("the voice list is empty".into()));
        }
        let mut caps = TtsCapabilities::new(config.sample_rate, MAX_INPUT_CHARS);
        caps.streams_audio = true;
        caps.speed = Some(0.5..=2.0);
        Ok(Self {
            config: Arc::new(config),
            runtime,
            caps,
        })
    }
}

impl std::fmt::Debug for DashScopeTts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DashScopeTts")
            .field("endpoint", &self.config.endpoint)
            .field("model", &self.config.model)
            .finish_non_exhaustive()
    }
}

impl TtsBackend for DashScopeTts {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn capabilities(&self) -> &TtsCapabilities {
        &self.caps
    }

    fn voices(&self) -> &[Voice] {
        &self.config.voices
    }

    fn open(&self, opts: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
        let voice = match &opts.voice {
            Some(voice) => voice.clone(),
            None => self.config.voices[0].id.clone(),
        };
        let mut stream = TaskStream {
            ws: None,
            config: self.config.clone(),
            runtime: self.runtime.clone(),
            voice,
            speed: opts.speed,
        };
        stream.connection()?;
        Ok(Box::new(stream))
    }
}

struct TaskStream {
    /// `None` after a task was abandoned midway; the next chunk reconnects.
    ws: Option<Ws>,
    config: Arc<DashScopeTtsConfig>,
    runtime: CloudRuntime,
    voice: String,
    speed: f32,
}

impl TaskStream {
    fn connection(&mut self) -> Result<&mut Ws, SpeechError> {
        if self.ws.is_none() {
            let config = &self.config;
            self.ws = Some(Ws::connect(
                self.runtime.clone(),
                BACKEND,
                &config.endpoint,
                &config.api_key,
                &[],
                config.timeout,
            )?);
        }
        self.ws
            .as_mut()
            .ok_or_else(|| SpeechError::backend(BACKEND, true, "the connection is closed"))
    }

    /// Runs one task. Returns `Ok(false)` if the sink stopped it midway,
    /// leaving the connection in an unknown state.
    fn task(
        &mut self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<bool, SpeechError> {
        let timeout = self.config.timeout;
        let task = uuid::Uuid::new_v4().simple().to_string();
        let run = protocol::run_task(
            &task,
            &self.config.model,
            &self.voice,
            self.config.sample_rate.hz(),
            self.speed,
        );
        let ws = self.connection()?;
        ws.send_json(&run, timeout)?;
        let mut deadline = instant_after(Instant::now(), timeout);
        loop {
            if let Some(Incoming::Json(value)) = ws.recv_frame(ws.time_left(deadline)?)? {
                match protocol::parse(&value, &task) {
                    Event::Started => break,
                    Event::Failed(error) => return Err(error),
                    Event::Finished => {
                        return Err(SpeechError::backend(
                            BACKEND,
                            false,
                            "the task finished before it started",
                        ));
                    }
                    Event::Ignore => {}
                }
            }
        }
        ws.send_json(&protocol::continue_task(&task, chunk), timeout)?;
        ws.send_json(&protocol::finish_task(&task), timeout)?;
        let mut decoder = Pcm16Decoder::new(BACKEND);
        deadline = instant_after(Instant::now(), timeout);
        loop {
            match ws.recv_frame(ws.time_left(deadline)?)? {
                Some(Incoming::Binary(bytes)) => {
                    let samples = decoder.push(&bytes);
                    if !samples.is_empty() && sink(&samples) == Flow::Stop {
                        return Ok(false);
                    }
                    deadline = instant_after(Instant::now(), timeout);
                }
                Some(Incoming::Json(value)) => match protocol::parse(&value, &task) {
                    Event::Finished => {
                        decoder.finish()?;
                        return Ok(true);
                    }
                    Event::Failed(error) => return Err(error),
                    Event::Started | Event::Ignore => {}
                },
                None => {}
            }
        }
    }
}

impl TtsStream for TaskStream {
    fn synthesize(
        &mut self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<(), SpeechError> {
        if chunk.trim().is_empty() {
            return Ok(());
        }
        match self.task(chunk, sink) {
            Ok(true) => Ok(()),
            Ok(false) => {
                self.ws = None;
                Ok(())
            }
            Err(error) => {
                self.ws = None;
                Err(error)
            }
        }
    }

    fn cancel(&mut self) {
        if let Some(mut ws) = self.ws.take() {
            ws.close(Duration::from_secs(1));
        }
    }
}
