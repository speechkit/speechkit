//! A watching: a wake-word detector reading the capture's timeline on a
//! thread of its own.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

use super::{
    Device, Hold, Listening, TICK, WeakHold, frames,
    listening::ListenOptions,
    lock,
    timeline::{Read, Timeline},
};
use crate::{
    Deadline, RecvError, SpeechError,
    asr::{AsrEngine, AsrOptions},
    speech::resample::Resampler,
    wake::{WakeEvent, WakeWordDetector, WakeWordModel},
};

/// The longest chunk the detector is fed at once.
const FEED: Duration = Duration::from_millis(100);
/// Audio before a watching's cursor that stays held, so a keyword the
/// detector reports late can still be reserved from its end.
const LOOKBACK: Duration = Duration::from_secs(2);
/// The smallest `max_backlog`.
const MIN_BACKLOG: Duration = Duration::from_millis(100);

/// How a [`Watching`] may fall behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct WatchOptions {
    /// How far the detector may fall behind the microphone. Past it, the
    /// watching closes with [`SpeechError::Capacity`] (D-02). At least
    /// 100 ms. Default: 30 s.
    pub max_backlog: Duration,
}

impl Default for WatchOptions {
    fn default() -> Self {
        Self {
            max_backlog: Duration::from_secs(30),
        }
    }
}

impl WatchOptions {
    /// Sets [`max_backlog`](Self::max_backlog).
    #[must_use]
    pub const fn with_max_backlog(mut self, max_backlog: Duration) -> Self {
        self.max_backlog = max_backlog;
        self
    }
}

/// What a [`Watching`] reports.
#[derive(Debug)]
#[non_exhaustive]
pub enum WakeUpdate {
    /// A keyword, with the audio after it reserved.
    Heard(Wake),
    /// The watching ended: `Ok` when the capture stopped or the watching
    /// was stopped, `Err` if the detector failed, fell behind, or lost audio. Always
    /// the last update.
    Closed(Result<(), SpeechError>),
}

/// A keyword a [`Watching`] heard, with the audio after it reserved.
///
/// The reservation holds that audio until the capture is 30 s past the
/// keyword's end, whether or not the application has read the event yet
/// (D-03). Dropping the `Wake` releases it.
pub struct Wake {
    event: WakeEvent,
    device: Arc<Device>,
    hold: WeakHold,
    reservation: Option<u64>,
}

impl Wake {
    /// The keyword, with its start and end in capture time.
    pub fn event(&self) -> &WakeEvent {
        &self.event
    }

    /// Starts a listening right after the keyword, from the reserved
    /// audio, as [`Capture::listen`](super::Capture::listen) does.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Capacity`] if the reservation lapsed, 30 s of capture
    /// after the keyword's end; [`SpeechError::Closed`] once the
    /// capture has stopped; or the errors
    /// [`AsrEngine::start`](crate::asr::AsrEngine::start) returns at once
    /// for the options.
    pub fn listen(
        mut self,
        engine: &AsrEngine,
        options: AsrOptions,
    ) -> Result<Listening, SpeechError> {
        let listen = ListenOptions::default();
        let reservation = self.reservation.take().ok_or(SpeechError::Capacity)?;
        let timeline = &self.device.timeline;
        let Some(hold) = self.hold.upgrade() else {
            timeline.release(reservation);
            return Err(SpeechError::Closed);
        };
        let limit = frames(self.device.rate, listen.max_backlog);
        let (reader, start) = timeline.claim(reservation, limit)?;
        Listening::start(&self.device, hold, reader, start, engine, options, &listen)
    }
}

impl Drop for Wake {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            self.device.timeline.release(reservation);
        }
    }
}

impl std::fmt::Debug for Wake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wake")
            .field("event", &self.event)
            .field("reserved", &self.reservation.is_some())
            .finish_non_exhaustive()
    }
}

/// State shared by the handle and the detector thread.
struct State {
    stop: AtomicBool,
    /// Keeps the microphone running until the watching stops or ends.
    hold: Mutex<Option<Arc<Hold>>>,
}

impl State {
    fn release(&self) {
        let hold = lock(&self.hold).take();
        drop(hold);
    }
}

/// A wake-word detector running on a capture, from
/// [`Capture::watch`](super::Capture::watch).
///
/// Read its [`WakeUpdate`]s with [`recv`](Self::recv),
/// [`try_recv`](Self::try_recv), or as an iterator. It keeps running while
/// the application listens and speaks, so a wake word can interrupt a
/// reply, until it is stopped or dropped, or the capture stops.
pub struct Watching {
    updates: mpsc::Receiver<WakeUpdate>,
    state: Arc<State>,
    closed: bool,
}

impl Watching {
    pub(super) fn start(
        device: &Arc<Device>,
        hold: &Arc<Hold>,
        model: &dyn WakeWordModel,
        options: WatchOptions,
    ) -> Result<Self, SpeechError> {
        if options.max_backlog < MIN_BACKLOG {
            return Err(SpeechError::InvalidInput(
                "max_backlog must be at least 100 ms".into(),
            ));
        }
        let detector = model.create()?;
        let resampler = Resampler::new(device.rate, model.sample_rate())?;
        let rate = device.rate;
        let (reader, start) = device.timeline.add_reader(
            None,
            frames(rate, LOOKBACK),
            frames(rate, options.max_backlog),
        )?;
        let state = Arc::new(State {
            stop: AtomicBool::new(false),
            hold: Mutex::new(Some(hold.clone())),
        });
        let (send, updates) = mpsc::channel();
        let detecting = Detecting {
            device: device.clone(),
            weak_hold: Arc::downgrade(hold),
            state: state.clone(),
            reader,
            origin: rate.duration_of(start),
            detector,
            resampler,
            send,
        };
        if let Err(error) = std::thread::Builder::new()
            .name("speechkit-wake".into())
            .spawn(move || detecting.run())
        {
            device.timeline.remove_reader(reader);
            return Err(SpeechError::backend("speechkit", true, error));
        }
        Ok(Self {
            updates,
            state,
            closed: false,
        })
    }

    /// The next update, waiting until `deadline`.
    ///
    /// # Errors
    ///
    /// [`RecvError::Timeout`] if the deadline passes first, or
    /// [`RecvError::Closed`] once `Closed` was delivered.
    pub fn recv(&mut self, deadline: impl Into<Deadline>) -> Result<WakeUpdate, RecvError> {
        if self.closed {
            return Err(RecvError::Closed);
        }
        let received = match deadline.into().remaining() {
            Some(left) => self
                .updates
                .recv_timeout(left)
                .map_err(|error| match error {
                    mpsc::RecvTimeoutError::Timeout => RecvError::Timeout,
                    mpsc::RecvTimeoutError::Disconnected => RecvError::Closed,
                }),
            None => self.updates.try_recv().map_err(|error| match error {
                mpsc::TryRecvError::Empty => RecvError::Timeout,
                mpsc::TryRecvError::Disconnected => RecvError::Closed,
            }),
        };
        self.seen(received)
    }

    /// The next update if one is ready.
    ///
    /// # Errors
    ///
    /// [`RecvError::Empty`] if none is ready, or [`RecvError::Closed`] once
    /// `Closed` was delivered.
    pub fn try_recv(&mut self) -> Result<WakeUpdate, RecvError> {
        if self.closed {
            return Err(RecvError::Closed);
        }
        let received = self.updates.try_recv().map_err(|error| match error {
            mpsc::TryRecvError::Empty => RecvError::Empty,
            mpsc::TryRecvError::Disconnected => RecvError::Closed,
        });
        self.seen(received)
    }

    fn seen(&mut self, received: Result<WakeUpdate, RecvError>) -> Result<WakeUpdate, RecvError> {
        match &received {
            Ok(WakeUpdate::Closed(_)) | Err(RecvError::Closed) => self.closed = true,
            _ => {}
        }
        received
    }

    /// Stops the watching, and returns at once (D-04). Its last update is
    /// `Closed(Ok(()))`. If it was the capture's only user, the
    /// microphone stops.
    pub fn stop(&self) {
        self.state.stop.store(true, Ordering::Release);
        self.state.release();
    }
}

impl Iterator for Watching {
    type Item = WakeUpdate;

    /// The next update, waiting as long as it takes; `None` after
    /// `Closed`.
    fn next(&mut self) -> Option<WakeUpdate> {
        if self.closed {
            return None;
        }
        let received = self.updates.recv().map_err(|_| RecvError::Closed);
        self.seen(received).ok()
    }
}

impl Drop for Watching {
    fn drop(&mut self) {
        self.stop();
    }
}

impl std::fmt::Debug for Watching {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Watching")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

/// The watching's thread.
struct Detecting {
    device: Arc<Device>,
    weak_hold: WeakHold,
    state: Arc<State>,
    reader: u64,
    /// The capture time of the first sample the detector hears.
    origin: Duration,
    detector: Box<dyn WakeWordDetector>,
    resampler: Resampler,
    send: mpsc::Sender<WakeUpdate>,
}

impl Detecting {
    fn run(mut self) {
        let result = self.detect();
        self.device.timeline.remove_reader(self.reader);
        self.state.release();
        let _ = self.send.send(WakeUpdate::Closed(result));
    }

    fn timeline(&self) -> &Timeline {
        &self.device.timeline
    }

    fn detect(&mut self) -> Result<(), SpeechError> {
        let max = usize::try_from(frames(self.device.rate, FEED)).unwrap_or(usize::MAX);
        let mut resampled = Vec::new();
        loop {
            if self.state.stop.load(Ordering::Acquire) {
                return Ok(());
            }
            match self.timeline().read(self.reader, max, Deadline::from(TICK)) {
                Read::Audio(chunk) => {
                    resampled.clear();
                    self.resampler.process(&chunk, &mut resampled)?;
                    let detector = &mut self.detector;
                    let events = catch_unwind(AssertUnwindSafe(|| detector.accept(&resampled)))
                        .map_err(|_| panicked())?;
                    if !self.heard(events) {
                        // The watching was dropped: nobody reads the rest.
                        return Ok(());
                    }
                }
                Read::Idle => {}
                Read::End(_) => {
                    resampled.clear();
                    self.resampler.flush(&mut resampled)?;
                    let detector = &mut self.detector;
                    let events = catch_unwind(AssertUnwindSafe(|| {
                        let mut events = detector.accept(&resampled);
                        events.extend(detector.flush());
                        events
                    }))
                    .map_err(|_| panicked())?;
                    self.heard(events);
                    return Ok(());
                }
                Read::Lost(_) => {
                    tracing::warn!(
                        "a wake-word detector lost audio, by falling further behind the microphone than max_backlog or because the microphone dropped samples; it stops"
                    );
                    return Err(SpeechError::Capacity);
                }
            }
        }
    }

    /// Reserves the audio after each keyword, then queues it (D-03).
    /// Returns false once nobody reads the updates.
    fn heard(&self, events: Vec<WakeEvent>) -> bool {
        let rate = self.device.rate;
        for event in events {
            let event = WakeEvent {
                start: self.origin + event.start,
                end: self.origin + event.end,
                keyword: event.keyword,
            };
            let reservation = self.timeline().reserve(frames(rate, event.end));
            tracing::debug!(reserved = reservation.is_some(), "wake word heard");
            let wake = Wake {
                event,
                device: self.device.clone(),
                hold: self.weak_hold.clone(),
                reservation,
            };
            if self.send.send(WakeUpdate::Heard(wake)).is_err() {
                return false;
            }
        }
        true
    }
}

fn panicked() -> SpeechError {
    SpeechError::backend("wake", false, "the wake-word detector panicked")
}
