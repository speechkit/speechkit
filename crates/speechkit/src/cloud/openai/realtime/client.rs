//! The Realtime transport.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    SampleRate, Secret, SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEvent, AsrEvents, AsrOptions, AsrStream},
    speech::resample::Resampler,
    vad::{
        VadModel,
        tracker::{Detected, Tracker},
    },
};

use super::{
    protocol::{self, BACKEND},
    state::{Activity, Phase, RealtimeState},
};
use crate::cloud::{
    CloudRuntime,
    ws::{Handler, Incoming, Pushed, Ws, check_session},
};

/// 100 ms of silence at 24 kHz, appended before the final commit so the
/// last word is not cut. It is not counted as audio.
const TAIL: usize = 2_400;
/// How long OpenAI keeps one Realtime session.
const CONNECTION_LIMIT: Duration = Duration::from_secs(60 * 60);

/// Settings for [`OpenAiRealtime`].
#[derive(Clone)]
#[non_exhaustive]
pub struct OpenAiRealtimeConfig {
    /// The endpoint. Default: OpenAI's transcription endpoint.
    pub endpoint: String,
    /// The transcription model.
    pub model: String,
    /// Sent as a bearer token.
    pub api_key: Arc<Secret>,
    /// Let the server split utterances and report speech. Default: true.
    pub server_vad: bool,
    /// A VAD the stream runs instead of the server's: it commits at each
    /// end of speech and reports activity. Default: none.
    pub vad: Option<Arc<dyn VadModel>>,
    /// With [`vad`](Self::vad), where an unfinished utterance is cut and
    /// committed. Default: 20 s.
    pub max_utterance: Duration,
    /// How far behind the audio sent the server may report speech. While
    /// the server reports nothing, speech activity is taken as known up to
    /// the audio sent minus this. Default: 1 s.
    pub activity_lag: Duration,
    /// The limit for connecting and for each wait on the server. Default: 30 s.
    pub timeout: Duration,
    /// How long the service keeps one connection. Default: 60 minutes.
    #[doc(hidden)]
    pub connection_limit: Duration,
}

impl OpenAiRealtimeConfig {
    /// The default OpenAI endpoint.
    pub const DEFAULT_ENDPOINT: &'static str =
        "wss://api.openai.com/v1/realtime?intent=transcription";

    /// Settings for `model` on OpenAI.
    pub fn new(model: impl Into<String>, api_key: impl Into<Arc<Secret>>) -> Self {
        Self {
            endpoint: Self::DEFAULT_ENDPOINT.into(),
            model: model.into(),
            api_key: api_key.into(),
            server_vad: true,
            vad: None,
            max_utterance: Duration::from_secs(20),
            activity_lag: Duration::from_secs(1),
            timeout: Duration::from_secs(30),
            connection_limit: CONNECTION_LIMIT,
        }
    }

    /// Sets the endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Turns server VAD on or off. Turning it on drops a
    /// [`with_vad`](Self::with_vad) VAD.
    #[must_use]
    pub fn with_server_vad(mut self, server_vad: bool) -> Self {
        self.server_vad = server_vad;
        if server_vad {
            self.vad = None;
        }
        self
    }

    /// Runs `vad` on the audio instead of server VAD, which this turns
    /// off. The stream commits at each end of speech and at
    /// [`with_max_utterance`](Self::with_max_utterance), and reports
    /// activity from the VAD. Some transcription models refuse server VAD.
    #[must_use]
    pub fn with_vad(mut self, vad: impl VadModel) -> Self {
        self.vad = Some(Arc::new(vad));
        self.server_vad = false;
        self
    }

    /// With [`with_vad`](Self::with_vad), sets where an unfinished
    /// utterance is cut and committed.
    #[must_use]
    pub fn with_max_utterance(mut self, max_utterance: Duration) -> Self {
        self.max_utterance = max_utterance;
        self
    }

    /// Sets how far behind the audio the server may report speech.
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

    /// Sets how long the service keeps one connection, for tests of the
    /// reconnect. Not covered by semver.
    #[doc(hidden)]
    #[must_use]
    pub fn with_connection_limit(mut self, limit: Duration) -> Self {
        self.connection_limit = limit;
        self
    }

    fn activity(&self) -> Activity {
        if self.vad.is_some() {
            Activity::Client
        } else if self.server_vad {
            Activity::Server {
                lag: self.activity_lag,
            }
        } else {
            Activity::None
        }
    }
}

impl std::fmt::Debug for OpenAiRealtimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiRealtimeConfig")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("api_key", &self.api_key)
            .field("server_vad", &self.server_vad)
            .field("vad", &self.vad.is_some())
            .field("max_utterance", &self.max_utterance)
            .field("activity_lag", &self.activity_lag)
            .field("timeout", &self.timeout)
            .field("connection_limit", &self.connection_limit)
            .finish()
    }
}

/// Transcription through an OpenAI Realtime transcription session.
///
/// Each speechkit session opens its own WebSocket, whose messages a task
/// on the runtime handles as they arrive. Audio streams as 24 kHz PCM16;
/// deltas arrive as partial results and transcripts as segments, in commit
/// order.
///
/// Speech activity comes from server VAD (`speech_started` and
/// `speech_stopped`), or from the application's VAD with
/// [`OpenAiRealtimeConfig::with_vad`]. With neither, the stream reports no
/// activity and commits once, at the end.
///
/// On finish, server VAD is turned off, 100 ms of silence is appended,
/// and the buffer is committed; the session ends when every committed item
/// has its transcript.
///
/// OpenAI keeps a Realtime session for at most 60 minutes, so a stream
/// reconnects at the first pause after about 55 minutes, winding the old
/// connection down as a finish does. With no pause by 59 minutes, or no
/// activity to find one, it reconnects anyway, and the utterance in
/// progress becomes two segments. Times and utterance IDs go on across
/// connections.
#[derive(Clone)]
pub struct OpenAiRealtime {
    config: Arc<OpenAiRealtimeConfig>,
    runtime: CloudRuntime,
    caps: AsrCapabilities,
}

impl OpenAiRealtime {
    /// A backend with `config`, running on `runtime`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad endpoint, an empty model or
    /// key, a zero timeout, server VAD together with a VAD, or a VAD that
    /// cannot run on 24 kHz audio.
    pub fn new(config: OpenAiRealtimeConfig, runtime: CloudRuntime) -> Result<Self, SpeechError> {
        check_session(
            &config.endpoint,
            &config.model,
            &config.api_key,
            config.timeout,
        )?;
        if config.server_vad && config.vad.is_some() {
            return Err(SpeechError::InvalidInput(
                "use server VAD or a VAD of your own, not both".into(),
            ));
        }
        if let Some(vad) = &config.vad {
            Resampler::new(SampleRate::HZ_24000, vad.sample_rate())?;
        }
        let mut caps = AsrCapabilities::new(SampleRate::HZ_24000);
        caps.reports_partials = true;
        caps.reports_activity = config.activity() != Activity::None;
        caps.accepts_language = true;
        caps.punctuated = true;
        Ok(Self {
            config: Arc::new(config),
            runtime,
            caps,
        })
    }
}

impl AsrBackend for OpenAiRealtime {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(&self, opts: &AsrOptions, sink: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        let config = &self.config;
        let vad = match &config.vad {
            Some(model) => Some(ClientVad {
                tracker: Tracker::new(
                    model.create(config.max_utterance)?,
                    model.sample_rate(),
                    model.start_delay(),
                ),
                resampler: Resampler::new(SampleRate::HZ_24000, model.sample_rate())?,
                buffer: Vec::new(),
            }),
            None => None,
        };
        let session = Session {
            state: RealtimeState::new(config.activity()),
            events: sink,
            updates: 0,
        };
        let conn = connect(&self.runtime, config, opts.language.as_deref(), session)?;
        Ok(Box::new(RealtimeStream {
            conn,
            connected: Instant::now(),
            runtime: self.runtime.clone(),
            language: opts.language.clone(),
            config: config.clone(),
            vad,
        }))
    }
}

/// Opens a connection that hands its messages to `session`, and configures
/// it.
fn connect(
    runtime: &CloudRuntime,
    config: &OpenAiRealtimeConfig,
    language: Option<&str>,
    session: Session,
) -> Result<Pushed<Session>, SpeechError> {
    let ws = Ws::connect(
        runtime.clone(),
        BACKEND,
        &config.endpoint,
        &config.api_key,
        &[],
        config.timeout,
    )?;
    let mut conn = ws.push(session)?;
    let update = protocol::session_update(&config.model, language, config.server_vad);
    conn.send_json(&update, config.timeout)?;
    conn.wait(config.timeout, |session| {
        session.state.phase() != Phase::Connecting
    })?;
    Ok(conn)
}

impl std::fmt::Debug for OpenAiRealtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiRealtime")
            .field("endpoint", &self.config.endpoint)
            .field("model", &self.config.model)
            .finish_non_exhaustive()
    }
}

/// What the reader task drives: the state, and where its events go.
struct Session {
    state: RealtimeState,
    events: AsrEvents,
    /// `session.updated` events seen.
    updates: u64,
}

impl Session {
    /// Applies `change` to the state and sends the events it makes.
    fn apply(&mut self, change: impl FnOnce(&mut RealtimeState) -> Vec<AsrEvent>) {
        let events = change(&mut self.state);
        self.events.send_all(events);
    }
}

impl Handler for Session {
    fn on_message(&mut self, message: Incoming) -> Result<(), SpeechError> {
        let Incoming::Json(value) = message else {
            return Ok(());
        };
        let event = protocol::parse(&value)?;
        if matches!(event, protocol::Event::SessionUpdated) {
            self.updates += 1;
        }
        let events = self.state.on_event(event)?;
        self.events.send_all(events);
        Ok(())
    }

    fn may_close(&self) -> bool {
        self.state.phase() == Phase::Closed
    }

    fn failed(&mut self, error: &SpeechError) {
        self.events.fail(error.clone());
    }
}

/// The application's VAD, on the audio the stream sends.
struct ClientVad {
    tracker: Tracker,
    /// From 24 kHz to the VAD's rate.
    resampler: Resampler,
    buffer: Vec<f32>,
}

struct RealtimeStream {
    conn: Pushed<Session>,
    /// When `conn` opened.
    connected: Instant,
    runtime: CloudRuntime,
    language: Option<String>,
    config: Arc<OpenAiRealtimeConfig>,
    vad: Option<ClientVad>,
}

impl RealtimeStream {
    /// Acts on what the client VAD found: reports speech, and commits each
    /// utterance, before the audio after it is sent.
    fn detected(&mut self, detected: Vec<Detected>) -> Result<(), SpeechError> {
        for found in detected {
            match found {
                Detected::Started(at) => {
                    self.conn
                        .with(|session| session.apply(|state| state.speech_started(at)));
                }
                Detected::Segment { end, ended, .. } => {
                    self.conn.with(|session| {
                        session.apply(|state| {
                            let id = state.reserve();
                            if ended {
                                state.speech_ended(end, id)
                            } else {
                                Vec::new()
                            }
                        });
                    });
                    self.conn
                        .send_json(&protocol::commit(), self.config.timeout)?;
                }
                Detected::Dropped { end, .. } => self.conn.with(|session| {
                    session.apply(|state| {
                        let id = state.fresh();
                        let mut events = state.speech_ended(end, id);
                        events.extend(state.dropped(id, end));
                        events
                    });
                }),
                Detected::Known(through) => {
                    self.conn
                        .with(|session| session.apply(|state| state.client_known(through)));
                }
            }
        }
        Ok(())
    }

    /// Sends audio and records it.
    fn append(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.conn
            .send_json(&protocol::append(samples), self.config.timeout)?;
        self.conn
            .with(|session| session.apply(|state| state.sent(samples.len())));
        Ok(())
    }

    /// Whether the connection is old enough to replace: at a pause after
    /// 11/12 of the limit (55 of 60 minutes), or regardless after 59/60.
    fn due(&self) -> bool {
        let limit = self.config.connection_limit;
        let age = self.connected.elapsed();
        if age >= limit / 60 * 59 {
            return true;
        }
        let pause = self.config.activity() != Activity::None
            && !self.conn.with(|session| session.state.speaking());
        age >= limit / 12 * 11 && pause
    }

    /// Winds the connection down and opens the next, which goes on where
    /// this one ends.
    fn reconnect(&mut self) -> Result<(), SpeechError> {
        tracing::debug!("reconnecting before the Realtime session limit");
        self.wind_down(false)?;
        let (state, events) = self
            .conn
            .with(|session| (session.state.continued(), session.events.clone()));
        let session = Session {
            state,
            events,
            updates: 0,
        };
        self.conn = connect(
            &self.runtime,
            &self.config,
            self.language.as_deref(),
            session,
        )?;
        self.connected = Instant::now();
        Ok(())
    }

    /// Ends the connection: stops server VAD, sends a short tail of
    /// silence, commits what needs committing, waits until every item has
    /// its transcript, and closes. At the end of the audio (`last`), the
    /// client VAD commits the speech left; before it, speech in progress
    /// is committed and goes on in the next connection, and a pause's
    /// silence is dropped.
    fn wind_down(&mut self, last: bool) -> Result<(), SpeechError> {
        let timeout = self.config.timeout;
        self.conn.check()?;
        let sent_audio = self.conn.with(|session| session.state.frames() > 0);
        if self.config.server_vad && sent_audio {
            // Stop the server splitting on its own, and wait until it
            // confirms, so the explicit commit below covers the tail.
            let seen = self.conn.with(|session| session.updates);
            self.conn
                .send_json(&protocol::disable_turn_detection(), timeout)?;
            self.conn
                .wait(timeout, |session| session.updates > seen)
                .map_err(|error| match error {
                    SpeechError::Backend { .. } if self.conn.check().is_ok() => {
                        SpeechError::backend(
                            BACKEND,
                            true,
                            "the server did not confirm the VAD change",
                        )
                    }
                    other => other,
                })?;
        }
        // Each branch decides what to commit and enters the committing phase
        // before the commit goes out: the server may acknowledge it at once,
        // and an acknowledgement that arrives before the phase changes is
        // lost, so the wait below would run out its whole timeout.
        if sent_audio {
            self.conn
                .send_json(&protocol::append(&[0.0; TAIL]), timeout)?;
            if self.vad.is_none() {
                self.conn
                    .with(|session| session.state.committing(true, last));
                self.conn.send_json(&protocol::commit(), timeout)?;
            } else if last {
                // The VAD commits what speech is left, inside `detected`.
                let detected = self
                    .vad
                    .take()
                    .map(|mut vad| vad.tracker.finish())
                    .unwrap_or_default();
                let committed = detected
                    .iter()
                    .any(|found| matches!(found, Detected::Segment { .. }));
                self.conn
                    .with(|session| session.state.committing(committed, last));
                self.detected(detected)?;
            } else if self.conn.with(|session| session.state.speaking()) {
                self.conn.with(|session| {
                    session.state.reserve();
                    session.state.committing(true, last);
                });
                self.conn.send_json(&protocol::commit(), timeout)?;
            } else {
                self.conn
                    .with(|session| session.state.committing(false, last));
            }
        } else {
            self.conn
                .with(|session| session.state.committing(false, last));
        }
        self.conn.wait(timeout, |session| session.state.settled())?;
        self.conn.with(|session| {
            session.state.close();
            if last {
                session.apply(RealtimeState::finished);
            }
        });
        self.conn.close(timeout);
        Ok(())
    }
}

impl AsrStream for RealtimeStream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.conn.check()?;
        if samples.is_empty() {
            return Ok(());
        }
        if self.due() {
            self.reconnect()?;
        }
        if let Some(vad) = &mut self.vad {
            vad.buffer.clear();
            vad.resampler.process(samples, &mut vad.buffer)?;
            let detected = vad.tracker.accept(&vad.buffer);
            self.detected(detected)?;
        }
        self.append(samples)
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        self.wind_down(true)
    }
}
