//! A listening: one request, fed from the capture's timeline by a thread
//! of its own.

use std::{
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use super::{
    Device, Hold, TICK, frames, lock,
    timeline::{Read, Timeline},
};
use crate::{
    AudioBuffer, Deadline, SampleRate, SpeechError,
    asr::{AsrEngine, AsrOptions, AsrResult, AsrSession, AsrUpdates, PushError, PushErrorKind},
    speech::{deadline::wait_until, opening::Opening},
};

/// The longest chunk the feeder pushes at once.
const FEED: Duration = Duration::from_millis(100);
/// The smallest `max_backlog`.
const MIN_BACKLOG: Duration = Duration::from_millis(100);

/// Where a [`Listening`] starts, and what it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ListenOptions {
    /// The capture time the listening starts at, up to
    /// [`CaptureOptions::history`](super::CaptureOptions::history) back;
    /// `None` starts at the capture's position when it is created.
    /// Default: `None`.
    pub start: Option<Duration>,
    /// How much audio to record from the start, for
    /// [`Listening::recording`]; zero records nothing. A minute takes about
    /// 11 MB at 48 kHz. Default: zero.
    pub recording: Duration,
    /// The most audio the listening holds while its session opens or
    /// falls behind. Past it, the listening fails with
    /// [`SpeechError::Capacity`] (D-02). At least 100 ms. Default: 30 s.
    pub max_backlog: Duration,
}

impl Default for ListenOptions {
    fn default() -> Self {
        Self {
            start: None,
            recording: Duration::ZERO,
            max_backlog: Duration::from_secs(30),
        }
    }
}

impl ListenOptions {
    /// Starts at capture time `start`, which must still be held: an older
    /// one fails at once with [`SpeechError::Capacity`] (D-01).
    #[must_use]
    pub fn starting_at(start: Duration) -> Self {
        Self {
            start: Some(start),
            ..Self::default()
        }
    }

    /// Sets [`recording`](Self::recording).
    #[must_use]
    pub const fn with_recording(mut self, recording: Duration) -> Self {
        self.recording = recording;
        self
    }

    /// Sets [`max_backlog`](Self::max_backlog).
    #[must_use]
    pub const fn with_max_backlog(mut self, max_backlog: Duration) -> Self {
        self.max_backlog = max_backlog;
        self
    }

    pub(super) fn validate(&self) -> Result<(), SpeechError> {
        if self.max_backlog < MIN_BACKLOG {
            return Err(SpeechError::InvalidInput(
                "max_backlog must be at least 100 ms".into(),
            ));
        }
        Ok(())
    }
}

/// The audio a listening recorded, for a retry.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Recording {
    /// The audio, from the listening's start, at the capture's rate. Once
    /// the listening has ended, up to its [`end`](Listening::end).
    pub audio: AudioBuffer,
    /// Whether the recording reached its limit, or the listening lost
    /// audio: later audio is not in it.
    pub truncated: bool,
}

/// What the feeder records.
struct Recorder {
    samples: Vec<f32>,
    limit: usize,
    truncated: bool,
}

/// What the feeder reports to the handle.
struct Status {
    /// The capture time of the end, once the listening has ended.
    end: Option<Duration>,
    recording: Option<Recorder>,
}

/// State shared by the handle and the feeder.
struct State {
    device: Arc<Device>,
    reader: u64,
    origin: Duration,
    /// Keeps the microphone running until the listening stops or ends.
    hold: Mutex<Option<Arc<Hold>>>,
    status: Mutex<Status>,
    ended: Condvar,
}

impl State {
    fn timeline(&self) -> &Timeline {
        &self.device.timeline
    }

    fn release(&self) {
        let hold = lock(&self.hold).take();
        drop(hold);
    }

    fn record(&self, chunk: &[f32]) {
        let mut status = lock(&self.status);
        let Some(recorder) = status.recording.as_mut() else {
            return;
        };
        let room = recorder.limit.saturating_sub(recorder.samples.len());
        if chunk.len() > room && !recorder.truncated {
            tracing::warn!("the recording reached its limit; later audio is not recorded");
            recorder.truncated = true;
        }
        recorder
            .samples
            .extend_from_slice(&chunk[..chunk.len().min(room)]);
    }

    fn truncate(&self) {
        if let Some(recorder) = lock(&self.status).recording.as_mut() {
            recorder.truncated = true;
        }
    }

    /// Ends the listening at `at`, and the recording with it: a session
    /// that cut itself off never heard the audio after its cutoff, so a
    /// retry must not hear it either. Both change in one step.
    fn end(&self, at: Duration) {
        let mut status = lock(&self.status);
        status.end = Some(at);
        if let Some(recorder) = status.recording.as_mut() {
            let until = nearest_frames(self.device.rate, at.saturating_sub(self.origin));
            recorder
                .samples
                .truncate(usize::try_from(until).unwrap_or(usize::MAX));
        }
        drop(status);
        self.ended.notify_all();
    }
}

/// The frames in `duration` at `rate`, to the nearest. Unlike
/// [`SampleRate::frames_in`], which rounds down, a frame count that went
/// through a `Duration` (itself rounded down) comes back whole.
fn nearest_frames(rate: SampleRate, duration: Duration) -> u64 {
    const NANOS_PER_SECOND: u128 = 1_000_000_000;
    let frames =
        (duration.as_nanos() * u128::from(rate.hz()) + NANOS_PER_SECOND / 2) / NANOS_PER_SECOND;
    u64::try_from(frames).unwrap_or(u64::MAX)
}

/// One request on a capture, from [`Capture::listen`](super::Capture::listen),
/// [`Microphone::listen`](super::Microphone::listen), or
/// [`Wake::listen`](super::Wake::listen).
///
/// Its session opens in the background, receives the held audio first and
/// live audio after it, and reports capture times. A listening ends when:
///
/// - [`stop`](Self::stop), [`finish`](Self::finish), or
///   [`cancel`](Self::cancel) is called;
/// - its session ends by itself, at a pause, with no speech, or at its
///   maximum length, and [`end`](Self::end) is then that cutoff;
/// - the device is lost or the capture stops, at the last sample;
/// - it loses audio, by falling further behind the microphone than
///   [`max_backlog`](ListenOptions::max_backlog) or because the microphone
///   dropped samples it had yet to read: the session fails with
///   [`SpeechError::Capacity`], and [`end`](Self::end) is where the
///   listening had read to.
///
/// Its session then finishes with what it heard. A session that fails
/// does not end the listening: the recording goes on until it is stopped,
/// so a retry gets the whole utterance, up to the recording's limit.
///
/// The listening is `Send + Sync`, so one thread can stop it while another
/// reads its updates. Dropping it cancels it.
pub struct Listening {
    session: Arc<AsrSession>,
    state: Arc<State>,
}

impl Listening {
    /// Starts the session for `reader`, which starts at `start`, and the
    /// feeder. On error, the reader is removed.
    pub(super) fn start(
        device: &Arc<Device>,
        hold: Arc<Hold>,
        reader: u64,
        start: u64,
        engine: &AsrEngine,
        options: AsrOptions,
        listen: &ListenOptions,
    ) -> Result<Self, SpeechError> {
        let rate = device.rate;
        let origin = rate.duration_of(start);
        let started = engine.start_with(
            rate,
            options,
            Deadline::from(Duration::MAX),
            Opening::Background,
            origin,
        );
        let session = match started {
            Ok(session) => Arc::new(session),
            Err(error) => {
                device.timeline.remove_reader(reader);
                return Err(error);
            }
        };
        let recording = (!listen.recording.is_zero()).then(|| Recorder {
            samples: Vec::new(),
            limit: usize::try_from(frames(rate, listen.recording)).unwrap_or(usize::MAX),
            truncated: false,
        });
        let state = Arc::new(State {
            device: device.clone(),
            reader,
            origin,
            hold: Mutex::new(Some(hold)),
            status: Mutex::new(Status {
                end: None,
                recording,
            }),
            ended: Condvar::new(),
        });
        let feeder = Feeder {
            session: session.clone(),
            state: state.clone(),
        };
        if let Err(error) = std::thread::Builder::new()
            .name(format!("speechkit-listen-{}", session.id()))
            .spawn(move || feeder.run())
        {
            device.timeline.remove_reader(reader);
            session.cancel();
            return Err(SpeechError::backend("speechkit", true, error));
        }
        Ok(Self { session, state })
    }

    /// A new reader of the session's updates. There may be any number.
    pub fn updates(&self) -> AsrUpdates {
        self.session.updates()
    }

    /// Ends the input at the capture's position, and returns at once, so it
    /// can run on a UI thread when a key goes up (D-04). The listening's
    /// thread then pushes the audio up to there and closes the session's
    /// input, and the session goes on to process what it heard. If this
    /// was the capture's only user, the microphone stops. Calling it again
    /// does nothing.
    pub fn stop(&self) {
        self.state.timeline().stop_reader(self.state.reader);
        self.state.release();
    }

    /// [`stop`](Self::stop)s, then waits for the session's result. Pushing
    /// the audio still held counts against `deadline` too: if it cannot all
    /// be pushed in time, the session fails with
    /// [`SpeechError::DeadlineExceeded`]. Every call returns the same
    /// result.
    ///
    /// # Errors
    ///
    /// The session's failure, with what it confirmed.
    pub fn finish(&self, deadline: impl Into<Deadline>) -> AsrResult {
        let deadline = deadline.into();
        self.stop();
        let status = lock(&self.state.status);
        let (status, ended) = wait_until(&self.state.ended, status, deadline, |status| {
            status.end.is_some()
        });
        drop(status);
        if !ended {
            self.session.fail(SpeechError::DeadlineExceeded);
        }
        self.session.finish(deadline)
    }

    /// The result once the session has ended by itself, waiting until
    /// `deadline`, or `None` if it is still running. It changes nothing.
    /// A transcript comes only with its [`end`](Self::end): the listening's
    /// thread notices the session's end within a tick, and a deadline that
    /// passes first gives `None`, as for a session still running.
    pub fn wait(&self, deadline: impl Into<Deadline>) -> Option<AsrResult> {
        let deadline = deadline.into();
        let result = self.session.wait(deadline)?;
        if result.is_ok() {
            let status = lock(&self.state.status);
            let (_status, ended) = wait_until(&self.state.ended, status, deadline, |status| {
                status.end.is_some()
            });
            if !ended {
                return None;
            }
        }
        Some(result)
    }

    /// Cancels the session and stops the listening. The recording is kept.
    pub fn cancel(&self) {
        self.session.cancel();
        self.stop();
    }

    /// The session's result, once it has ended.
    pub fn result(&self) -> Option<&AsrResult> {
        self.session.result()
    }

    /// The capture time of the listening's first sample.
    pub fn origin(&self) -> Duration {
        self.state.origin
    }

    /// The capture time where the listening ended, once it has: where its
    /// session ended itself (its cutoff, which can be before the point
    /// where the listening was stopped, if the session had not yet worked
    /// through the audio queued for it), where it was stopped, or the last
    /// sample of a stopped capture or a lost device. After a
    /// [`stop`](Self::stop) it is set once the session has ended.
    pub fn end(&self) -> Option<Duration> {
        lock(&self.state.status).end
    }

    /// A copy of the audio recorded so far, or `None` if
    /// [`ListenOptions::recording`] is zero. It is complete once the
    /// listening has ended, and then runs from [`origin`](Self::origin) to
    /// [`end`](Self::end): audio after a cutoff, which the session never
    /// heard, is not in it.
    pub fn recording(&self) -> Option<Recording> {
        let status = lock(&self.state.status);
        let recorder = status.recording.as_ref()?;
        Some(Recording {
            audio: AudioBuffer::new(self.state.device.rate, recorder.samples.clone()),
            truncated: recorder.truncated,
        })
    }

    /// The level of the capture's latest audio, from 0.0 to 1.0, for a
    /// meter. See [`Capture::level`](super::Capture::level).
    pub fn level(&self) -> f32 {
        self.state.device.level()
    }

    /// Whether the device stopped delivering audio.
    pub fn device_lost(&self) -> bool {
        self.state.device.device_lost()
    }
}

impl Drop for Listening {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl std::fmt::Debug for Listening {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Listening")
            .field("session", &self.session)
            .field("origin", &self.state.origin)
            .field("end", &self.end())
            .finish_non_exhaustive()
    }
}

/// Where a listening whose input stopped at `stopped` ends, once its
/// session has ended. The session may still hold queued audio past the
/// point where it cuts itself off (A-07), so the stop is not yet the end:
/// the cutoff is, when there is one.
fn settled_end(session: &AsrSession, stopped: Duration) -> Duration {
    // `wait` only reads the session; `finish`'s deadline, or dropping the
    // listening, ends a session that never finishes.
    let _ = session.wait(Deadline::from(Duration::MAX));
    session.cutoff().unwrap_or(stopped)
}

/// The listening's thread: reads from its cursor and pushes into the
/// session.
struct Feeder {
    session: Arc<AsrSession>,
    state: Arc<State>,
}

impl Feeder {
    fn run(self) {
        let Self { session, state } = self;
        let rate = state.device.rate;
        let timeline = state.timeline();
        let reader = state.reader;
        let max = session
            .max_chunk_frames()
            .min(usize::try_from(frames(rate, FEED)).unwrap_or(usize::MAX))
            .max(1);
        let mut pending: Option<Vec<f32>> = None;
        // False once the session has failed: the listening then only
        // records, until it is stopped.
        let mut pushing = true;
        let end = loop {
            if let Some(chunk) = pending.take()
                && pushing
            {
                match session.push(chunk, Deadline::from(TICK)) {
                    Ok(()) => {}
                    Err(PushError {
                        kind: PushErrorKind::Full,
                        chunk,
                        ..
                    }) => {
                        // Room comes, or the timeline marks the reader
                        // behind and the read below says so.
                        if !timeline.is_lost(reader) {
                            pending = Some(chunk);
                            continue;
                        }
                    }
                    Err(PushError {
                        kind: PushErrorKind::Closed,
                        ..
                    }) => {
                        if let Some(cutoff) = session.cutoff() {
                            break cutoff;
                        }
                        pushing = false;
                    }
                    // Device samples are valid and chunks fit the queue, so
                    // this is not expected; fail rather than lose the audio.
                    Err(error) => {
                        session.fail(error.into());
                        pushing = false;
                    }
                }
            }
            match timeline.read(reader, max, Deadline::from(TICK)) {
                Read::Audio(chunk) => {
                    state.record(&chunk);
                    pending = Some(chunk);
                }
                Read::Idle => {
                    if pushing {
                        if let Some(cutoff) = session.cutoff() {
                            break cutoff;
                        }
                        pushing = session.result().is_none();
                    }
                }
                Read::End(at) => {
                    if pushing {
                        session.close_input();
                    }
                    // The input is closed, so nothing needs the reader or
                    // the microphone while the session works through the
                    // audio it was given.
                    timeline.remove_reader(reader);
                    state.release();
                    break settled_end(&session, rate.duration_of(at));
                }
                Read::Lost(at) => {
                    tracing::warn!(
                        "a listening lost audio, by falling further behind the microphone than max_backlog or because the microphone dropped samples; it fails"
                    );
                    session.fail(SpeechError::Capacity);
                    state.truncate();
                    break rate.duration_of(at);
                }
            }
        };
        timeline.remove_reader(reader);
        state.release();
        state.end(end);
    }
}
