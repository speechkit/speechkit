//! The DashScope recognition transport.

use std::{sync::Arc, time::Duration};

use crate::{
    SampleRate, Secret, SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEvent, AsrEvents, AsrOptions, AsrStream},
    audio::pcm16_bytes,
};

use super::{
    protocol::{self, BACKEND},
    state::{DashScopeState, Phase},
};
use crate::cloud::{
    CloudRuntime,
    ws::{Handler, Incoming, Pushed, Ws, WsMessage, check_session},
};

/// Settings for [`DashScopeAsr`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DashScopeAsrConfig {
    /// The endpoint. Default: the mainland China endpoint.
    pub endpoint: String,
    /// The model, for example `paraformer-realtime-v2`.
    pub model: String,
    /// Sent as a bearer token.
    pub api_key: Arc<Secret>,
    /// How far behind the audio sent the service may report speech. While
    /// it reports nothing, speech activity is taken as known up to the
    /// audio sent minus this. Default: 1 s.
    pub activity_lag: Duration,
    /// The limit for connecting and for each wait on the server. Default: 30 s.
    pub timeout: Duration,
}

impl DashScopeAsrConfig {
    /// The mainland China endpoint. Singapore workspaces use
    /// `wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/`.
    pub const DEFAULT_ENDPOINT: &'static str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/";

    /// Settings for `model`.
    pub fn new(model: impl Into<String>, api_key: impl Into<Arc<Secret>>) -> Self {
        Self {
            endpoint: Self::DEFAULT_ENDPOINT.into(),
            model: model.into(),
            api_key: api_key.into(),
            activity_lag: Duration::from_secs(1),
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets the endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets how far behind the audio the service may report speech.
    #[must_use]
    pub fn with_activity_lag(mut self, lag: Duration) -> Self {
        self.activity_lag = lag;
        self
    }

    /// Sets the timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Real-time recognition on DashScope. Each session is one task on its
/// own WebSocket, whose messages a task on the runtime handles as they
/// arrive; audio streams as 16 kHz PCM16. The task sends heartbeats, so a
/// long silence keeps the connection.
///
/// Speech activity comes from the service's sentences: a sentence's first
/// result starts speech at its `begin_time`, and its final result ends it
/// at its `end_time`.
#[derive(Clone)]
pub struct DashScopeAsr {
    config: Arc<DashScopeAsrConfig>,
    runtime: CloudRuntime,
    caps: AsrCapabilities,
}

impl DashScopeAsr {
    /// A backend with `config`, running on `runtime`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad endpoint, an empty model or
    /// key, or a zero timeout.
    pub fn new(config: DashScopeAsrConfig, runtime: CloudRuntime) -> Result<Self, SpeechError> {
        check_session(
            &config.endpoint,
            &config.model,
            &config.api_key,
            config.timeout,
        )?;
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.reports_partials = true;
        caps.reports_activity = true;
        caps.accepts_language = true;
        caps.punctuated = true;
        Ok(Self {
            config: Arc::new(config),
            runtime,
            caps,
        })
    }
}

impl AsrBackend for DashScopeAsr {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(&self, opts: &AsrOptions, sink: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        let config = &self.config;
        let ws = Ws::connect(
            self.runtime.clone(),
            BACKEND,
            &config.endpoint,
            &config.api_key,
            &[],
            config.timeout,
        )?;
        let task = uuid::Uuid::new_v4().simple().to_string();
        let mut conn = ws.push(Task {
            state: DashScopeState::new(config.activity_lag),
            events: sink,
            id: task.clone(),
        })?;
        let run = protocol::run_task(&task, &config.model, opts.language.as_deref());
        conn.send_json(&run, config.timeout)?;
        conn.wait(config.timeout, |task| task.state.phase() == Phase::Running)?;
        Ok(Box::new(TaskStream {
            conn,
            config: config.clone(),
            task,
        }))
    }
}

impl std::fmt::Debug for DashScopeAsr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DashScopeAsr")
            .field("endpoint", &self.config.endpoint)
            .field("model", &self.config.model)
            .finish_non_exhaustive()
    }
}

/// What the reader task drives: the state, and where its events go.
struct Task {
    state: DashScopeState,
    events: AsrEvents,
    /// The task's ID.
    id: String,
}

impl Task {
    /// Applies `change` to the state and sends the events it makes.
    fn apply(&mut self, change: impl FnOnce(&mut DashScopeState) -> Vec<AsrEvent>) {
        let events = change(&mut self.state);
        self.events.send_all(events);
    }
}

impl Handler for Task {
    fn on_message(&mut self, message: Incoming) -> Result<(), SpeechError> {
        let Incoming::Json(value) = message else {
            return Ok(());
        };
        let events = self.state.on_event(protocol::parse(&value, &self.id))?;
        self.events.send_all(events);
        Ok(())
    }

    fn may_close(&self) -> bool {
        self.state.phase() == Phase::Finished
    }

    fn failed(&mut self, error: &SpeechError) {
        self.events.fail(error.clone());
    }
}

struct TaskStream {
    conn: Pushed<Task>,
    config: Arc<DashScopeAsrConfig>,
    task: String,
}

impl AsrStream for TaskStream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.conn.check()?;
        if !samples.is_empty() {
            let audio = WsMessage::Binary(pcm16_bytes(samples).into());
            self.conn.send(audio, self.config.timeout)?;
            self.conn
                .with(|task| task.apply(|state| state.sent(samples.len())));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        self.conn.check()?;
        // The phase changes first: the server may answer `task-finished` as
        // soon as the request is out, and the reader must find the state
        // already finishing. The lock is not held across the send, since the
        // reader would then block while both sides write.
        self.conn.with(|task| task.state.finishing());
        self.conn
            .send_json(&protocol::finish_task(&self.task), self.config.timeout)?;
        self.conn.wait(self.config.timeout, |task| {
            task.state.phase() == Phase::Finished
        })?;
        self.conn.with(|task| task.apply(DashScopeState::finished));
        self.conn.close(self.config.timeout);
        Ok(())
    }
}
