//! A scripted fake speech recognition backend.

use std::{
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use speechkit::{
    Flow, SampleRate, SpeechError,
    asr::{
        AsrBackend, AsrCapabilities, AsrEvent, AsrEvents, AsrOptions, AsrStream, Partial, Segment,
        UtteranceId,
    },
};

use crate::Gate;

/// When a scripted step runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// Once the stream has accepted at least this many samples, at the
    /// backend's rate.
    AfterSamples(u64),
    /// When the stream is finished.
    OnFinish,
    /// When the stream is opened.
    OnOpen,
}

/// What a scripted step does.
#[derive(Debug)]
pub enum Step {
    /// Report a partial result for utterance `.0`.
    Partial(u64, &'static str),
    /// Commit utterance `.0`. Its end is the current sample position, and
    /// its start is the end of the previous segment.
    Segment(u64, &'static str),
    /// Send an event as it is.
    Send(AsrEvent),
    /// Commit utterance `.1` from a thread of the stream's own, `.0` after
    /// the step runs, while the session may be idle.
    SegmentLater(Duration, u64, &'static str),
    /// Report a failure from a thread of the stream's own, `.0` after the
    /// step runs, as a dropped connection would.
    FailLater(Duration, SpeechError),
    /// Sleep, like a slow model.
    Sleep(Duration),
    /// Block until the gate is released, like a native call that cannot
    /// be interrupted.
    BlockUntilReleased(Gate),
    /// Fail with (a copy of) this error.
    Fail(SpeechError),
    /// Panic.
    Panic,
}

impl Step {
    /// Sends `SpeechStarted` at `at`.
    pub fn started(at: Duration) -> Self {
        Self::Send(AsrEvent::SpeechStarted { at })
    }

    /// Sends `SpeechEnded` at `at`, naming `utterance` as the last one of
    /// the speech.
    pub fn ended(at: Duration, utterance: u64) -> Self {
        Self::Send(AsrEvent::SpeechEnded {
            at,
            utterance: UtteranceId(utterance),
        })
    }

    /// Sends `ActivityKnown` through `through`.
    pub fn known(through: Duration) -> Self {
        Self::Send(AsrEvent::ActivityKnown { through })
    }
}

/// A list of steps, each with its trigger. Steps run in list order.
#[derive(Debug, Default)]
pub struct Script {
    steps: Vec<(Trigger, Step)>,
}

impl Script {
    /// An empty script: the stream never reports anything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a step.
    #[must_use]
    pub fn then(mut self, trigger: Trigger, step: Step) -> Self {
        self.steps.push((trigger, step));
        self
    }
}

/// Counters shared by a fake backend and every stream it opens.
#[derive(Debug, Default)]
pub struct FakeStats {
    /// Streams opened.
    pub opened: AtomicUsize,
    /// Streams currently alive.
    pub alive: AtomicUsize,
    /// `finish` calls.
    pub finished: AtomicUsize,
    /// `cancel` calls.
    pub cancelled: AtomicUsize,
    /// Samples accepted, across all streams.
    pub samples: AtomicU64,
    /// `accept` calls, across all streams.
    pub accepts: AtomicUsize,
    /// The audio accepted, across all streams, in order.
    pub heard: Mutex<Vec<f32>>,
}

impl FakeStats {
    fn get(counter: &AtomicUsize) -> usize {
        counter.load(Ordering::SeqCst)
    }

    /// Streams opened so far.
    pub fn opened(&self) -> usize {
        Self::get(&self.opened)
    }

    /// Streams currently alive.
    pub fn alive(&self) -> usize {
        Self::get(&self.alive)
    }

    /// `cancel` calls so far.
    pub fn cancelled(&self) -> usize {
        Self::get(&self.cancelled)
    }

    /// `finish` calls so far.
    pub fn finished(&self) -> usize {
        Self::get(&self.finished)
    }

    /// `accept` calls so far.
    pub fn accepts(&self) -> usize {
        Self::get(&self.accepts)
    }

    /// A copy of the audio accepted so far, across all streams, at the
    /// backend's rate.
    pub fn heard(&self) -> Vec<f32> {
        self.heard
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// A fake backend that runs a [`Script`] in every stream it opens.
#[derive(Clone)]
pub struct FakeAsr {
    name: String,
    caps: AsrCapabilities,
    script: Arc<Script>,
    stats: Arc<FakeStats>,
}

impl FakeAsr {
    /// A fake at 16 kHz with partial results, running `script`.
    pub fn new(script: Script) -> Self {
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.reports_partials = true;
        Self {
            name: "fake".into(),
            caps,
            script: Arc::new(script),
            stats: Arc::default(),
        }
    }

    /// A fake that says "hello" after half a second and "world" at the end.
    pub fn hello_world() -> Self {
        Self::new(
            Script::new()
                .then(Trigger::AfterSamples(4_000), Step::Partial(0, "hel"))
                .then(Trigger::AfterSamples(8_000), Step::Segment(0, "hello"))
                .then(Trigger::OnFinish, Step::Segment(1, "world")),
        )
    }

    /// Says the fake reports speech activity, as a script with activity
    /// steps does.
    #[must_use]
    pub fn reporting_activity(mut self) -> Self {
        self.caps.reports_activity = true;
        self
    }

    /// Replaces the capabilities.
    #[must_use]
    pub fn with_capabilities(mut self, caps: AsrCapabilities) -> Self {
        self.caps = caps;
        self
    }

    /// The counters shared with every stream.
    pub fn stats(&self) -> Arc<FakeStats> {
        self.stats.clone()
    }
}

impl AsrBackend for FakeAsr {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(&self, _: &AsrOptions, events: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        self.stats.opened.fetch_add(1, Ordering::SeqCst);
        self.stats.alive.fetch_add(1, Ordering::SeqCst);
        let mut stream = FakeStream {
            events,
            script: self.script.clone(),
            stats: self.stats.clone(),
            rate: self.caps.sample_rate,
            done: vec![false; self.script.steps.len()],
            samples: 0,
            last_end: Duration::ZERO,
            threads: Vec::new(),
        };
        stream.run(|trigger, _| trigger == Trigger::OnOpen)?;
        Ok(Box::new(stream))
    }
}

impl std::fmt::Debug for FakeAsr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeAsr")
            .field("name", &self.name)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

#[expect(clippy::panic, reason = "the script asked for a panic")]
fn scripted_panic() -> ! {
    panic!("scripted panic")
}

struct FakeStream {
    events: AsrEvents,
    script: Arc<Script>,
    stats: Arc<FakeStats>,
    rate: SampleRate,
    done: Vec<bool>,
    samples: u64,
    last_end: Duration,
    /// Threads of its own, joined when the stream is dropped.
    threads: Vec<JoinHandle<()>>,
}

impl FakeStream {
    fn run(&mut self, due: impl Fn(Trigger, u64) -> bool) -> Result<(), SpeechError> {
        let script = self.script.clone();
        for (index, (trigger, step)) in script.steps.iter().enumerate() {
            if self.done[index] || !due(*trigger, self.samples) {
                continue;
            }
            self.done[index] = true;
            match step {
                Step::Partial(id, text) => {
                    self.events.send(AsrEvent::Partial(Partial {
                        utterance: UtteranceId(*id),
                        text: (*text).to_owned(),
                    }));
                }
                Step::Segment(id, text) => {
                    let segment = self.segment(*id, text);
                    self.events.send(segment);
                }
                Step::Send(event) => {
                    self.events.send(event.clone());
                }
                Step::SegmentLater(delay, id, text) => {
                    let (delay, events) = (*delay, self.events.clone());
                    let segment = self.segment(*id, text);
                    self.threads.push(std::thread::spawn(move || {
                        std::thread::sleep(delay);
                        events.send(segment);
                    }));
                }
                Step::FailLater(delay, error) => {
                    let (delay, events, error) = (*delay, self.events.clone(), error.clone());
                    self.threads.push(std::thread::spawn(move || {
                        std::thread::sleep(delay);
                        events.fail(error);
                    }));
                }
                Step::Sleep(duration) => std::thread::sleep(*duration),
                Step::BlockUntilReleased(gate) => gate.wait(),
                Step::Fail(error) => return Err(error.clone()),
                Step::Panic => scripted_panic(),
            }
        }
        Ok(())
    }

    /// A segment for utterance `id` that ends at the current position.
    fn segment(&mut self, id: u64, text: &str) -> AsrEvent {
        let end = self.rate.duration_of(self.samples);
        let segment = AsrEvent::Segment(Segment {
            utterance: UtteranceId(id),
            text: text.to_owned(),
            start: self.last_end.min(end),
            end,
        });
        self.last_end = end;
        segment
    }
}

impl AsrStream for FakeStream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.samples += samples.len() as u64;
        self.stats
            .samples
            .fetch_add(samples.len() as u64, Ordering::SeqCst);
        self.stats.accepts.fetch_add(1, Ordering::SeqCst);
        self.stats
            .heard
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(samples);
        self.run(|trigger, seen| matches!(trigger, Trigger::AfterSamples(n) if n <= seen))
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        self.stats.finished.fetch_add(1, Ordering::SeqCst);
        self.run(|trigger, seen| match trigger {
            Trigger::AfterSamples(n) => n <= seen,
            Trigger::OnFinish => true,
            Trigger::OnOpen => false,
        })
    }

    fn cancel(&mut self) {
        self.stats.cancelled.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        self.stats.alive.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Events a stream sent, collected for tests that drive a stream without
/// an engine.
#[derive(Debug, Clone, Default)]
pub struct Collected {
    events: Arc<Mutex<Vec<AsrEvent>>>,
    failure: Arc<Mutex<Option<SpeechError>>>,
}

impl Collected {
    /// An [`AsrEvents`] that collects into this.
    pub fn events(&self) -> AsrEvents {
        let (events, failure) = (self.events.clone(), self.failure.clone());
        AsrEvents::forward(
            move |event| {
                events
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(event);
                Flow::Continue
            },
            move |error| {
                *failure.lock().unwrap_or_else(PoisonError::into_inner) = Some(error);
            },
        )
    }

    /// The events sent since the last call.
    pub fn take(&self) -> Vec<AsrEvent> {
        std::mem::take(&mut *self.events.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// The failure reported, if any.
    pub fn failure(&self) -> Option<SpeechError> {
        self.failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use speechkit::asr::AsrOptions;

    use super::*;

    fn opts() -> AsrOptions {
        AsrOptions::default()
    }

    #[test]
    fn steps_fire_by_sample_count_and_finish() {
        let fake = FakeAsr::hello_world();
        let sent = Collected::default();
        let mut stream = fake.open(&opts(), sent.events()).unwrap();
        stream.accept(&[0.0; 3_999]).unwrap();
        assert!(sent.take().is_empty());
        stream.accept(&[0.0; 1]).unwrap();
        let events = sent.take();
        assert!(matches!(&events[..], [AsrEvent::Partial(p)] if p.text == "hel"));
        stream.accept(&[0.0; 4_000]).unwrap();
        let events = sent.take();
        let [AsrEvent::Segment(segment)] = &events[..] else {
            panic!("{events:?}");
        };
        assert_eq!(segment.end, Duration::from_millis(500));
        stream.finish().unwrap();
        let events = sent.take();
        assert!(matches!(&events[..], [AsrEvent::Segment(s)] if s.text == "world"));
        assert_eq!(fake.stats().finished(), 1);
        assert_eq!(fake.stats().alive(), 1);
        drop(stream);
        assert_eq!(fake.stats().alive(), 0);
        assert_eq!(fake.stats().opened(), 1);
    }

    #[test]
    fn on_open_steps_can_fail_or_send() {
        let failing =
            FakeAsr::new(Script::new().then(Trigger::OnOpen, Step::Fail(SpeechError::Capacity)));
        let sent = Collected::default();
        assert!(matches!(
            failing.open(&opts(), sent.events()),
            Err(SpeechError::Capacity)
        ));
        let early = FakeAsr::new(Script::new().then(Trigger::OnOpen, Step::Partial(0, "early")));
        let _stream = early.open(&opts(), sent.events()).unwrap();
        assert_eq!(sent.take().len(), 1);
    }

    #[test]
    fn later_steps_run_on_a_thread_joined_by_drop() {
        let fake = FakeAsr::new(
            Script::new()
                .then(
                    Trigger::OnOpen,
                    Step::SegmentLater(Duration::from_millis(20), 0, "late"),
                )
                .then(
                    Trigger::OnOpen,
                    Step::FailLater(Duration::from_millis(20), SpeechError::Closed),
                ),
        );
        let sent = Collected::default();
        let stream = fake.open(&opts(), sent.events()).unwrap();
        drop(stream);
        assert_eq!(sent.take().len(), 1);
        assert!(matches!(sent.failure(), Some(SpeechError::Closed)));
    }

    #[test]
    fn failures_and_panics() {
        let fake = FakeAsr::new(
            Script::new()
                .then(
                    Trigger::AfterSamples(1),
                    Step::Sleep(Duration::from_millis(1)),
                )
                .then(Trigger::AfterSamples(2), Step::Fail(SpeechError::Closed))
                .then(Trigger::OnFinish, Step::Panic),
        );
        let mut stream = fake.open(&opts(), Collected::default().events()).unwrap();
        stream.accept(&[0.0]).unwrap();
        assert!(matches!(stream.accept(&[0.0]), Err(SpeechError::Closed)));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stream.finish()));
        assert!(panicked.is_err());
        stream.cancel();
        assert_eq!(fake.stats().cancelled(), 1);
    }

    #[test]
    fn gate_blocks_until_released() {
        let gate = Gate::new();
        let fake = FakeAsr::new(Script::new().then(
            Trigger::AfterSamples(1),
            Step::BlockUntilReleased(gate.clone()),
        ));
        let mut stream = fake.open(&opts(), Collected::default().events()).unwrap();
        let worker = std::thread::spawn(move || stream.accept(&[0.0]));
        assert!(gate.wait_entered(1, Duration::from_secs(5)));
        assert!(!worker.is_finished());
        gate.release();
        assert!(worker.join().unwrap().is_ok());
    }
}
