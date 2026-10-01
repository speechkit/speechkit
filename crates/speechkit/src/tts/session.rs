//! A running synthesis: the text side, the audio side, and the thread
//! between them.

use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, OnceLock},
    time::Duration,
};

use super::{
    Chunker, Mark, TtsBackend, TtsFailure, TtsOptions, TtsResult, TtsStream, TtsSummary, TtsUpdate,
};
use crate::{
    Deadline, Flow, RecvError, SampleRate, SpeechError,
    speech::{
        deadline,
        opening::Stage,
        resample::Resampler,
        slots::{SlotGuard, SlotLimiter},
        sync::{Condvar, Mutex, MutexGuard, lock},
    },
};

/// One item waiting in the output queue.
enum Item {
    Audio(Vec<f32>),
    Mark(Mark),
}

pub(crate) struct Shared {
    pub(crate) id: u64,
    pub(crate) sample_rate: SampleRate,
    /// Output queue capacity, in samples. Marks don't count.
    pub(crate) capacity: usize,
    pub(crate) max_text_chars: usize,
    /// Woken when the session ends, so a thread waiting for a slot gives up.
    pub(crate) slots: SlotLimiter,
    pub(crate) core: Mutex<Core>,
    pub(crate) changed: Condvar,
    pub(crate) terminal: OnceLock<Arc<TtsResult>>,
}

pub(crate) struct Core {
    pub(crate) stage: Stage,
    /// Text pushed but not yet taken by the session thread.
    text: String,
    /// All the text pushed, for a speaker's `text_played`. It is bounded
    /// by `max_text_chars`.
    pushed: String,
    text_closed: bool,
    chars: usize,
    output: VecDeque<Item>,
    /// Samples in `output`.
    queued: usize,
    /// The most samples ever queued at once, for the T-01 checks.
    peak: usize,
    /// Samples queued so far, at the output rate.
    emitted: u64,
    /// Marks whose audio is not all queued yet, with the sample it ends at.
    pending_marks: VecDeque<(Mark, u64)>,
    /// Bytes of text fully synthesized.
    text_done: usize,
    /// The synthesis is done; the result is published once the reader has
    /// taken everything queued.
    done: Option<TtsSummary>,
    closed_delivered: bool,
}

impl Shared {
    pub(crate) fn new(
        id: u64,
        sample_rate: SampleRate,
        capacity: usize,
        max_text_chars: usize,
        slots: SlotLimiter,
    ) -> Self {
        Self {
            id,
            sample_rate,
            capacity: capacity.max(1),
            max_text_chars,
            slots,
            core: Mutex::new(Core {
                stage: Stage::Slot,
                text: String::new(),
                pushed: String::new(),
                text_closed: false,
                chars: 0,
                output: VecDeque::new(),
                queued: 0,
                peak: 0,
                emitted: 0,
                pending_marks: VecDeque::new(),
                text_done: 0,
                done: None,
                closed_delivered: false,
            }),
            changed: Condvar::new(),
            terminal: OnceLock::new(),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Core> {
        lock(&self.core)
    }

    /// Records `result` unless the session already ended. Taking the
    /// locked `Core` means callers settle under the lock, so a waiter that
    /// checked `terminal` under it cannot miss the notification.
    fn settle(&self, _locked: &Core, result: TtsResult) {
        if self.terminal.get().is_none() {
            if let Err(failure) = &result {
                tracing::debug!(session = self.id, error = %failure.error, "synthesis failed");
            } else {
                tracing::debug!(session = self.id, "synthesis finished");
            }
            let _ = self.terminal.set(Arc::new(result));
        }
        self.changed.notify_all();
        self.slots.wake();
    }

    /// Fails the session, keeping its progress (T-09).
    pub(crate) fn fail(&self, core: &Core, error: SpeechError) {
        let failure = TtsFailure {
            error,
            text_done: core.text_done,
            duration: self.sample_rate.duration_of(core.emitted),
        };
        self.settle(core, Err(failure));
    }

    /// Publishes the summary once the synthesis is done and the reader has
    /// taken everything queued.
    fn try_complete(&self, core: &mut Core) {
        if core.output.is_empty()
            && self.terminal.get().is_none()
            && let Some(summary) = core.done.take()
        {
            self.settle(core, Ok(summary));
        }
    }

    /// The next update for the reader, if there is one yet.
    fn take(&self, core: &mut Core) -> Result<TtsUpdate, RecvError> {
        if core.closed_delivered {
            return Err(RecvError::Closed);
        }
        if let Some(item) = core.output.pop_front() {
            let update = match item {
                Item::Audio(samples) => {
                    core.queued -= samples.len();
                    TtsUpdate::Audio(samples)
                }
                Item::Mark(mark) => TtsUpdate::Mark(mark),
            };
            self.try_complete(core);
            self.changed.notify_all();
            return Ok(update);
        }
        match self.terminal.get() {
            Some(result) => {
                core.closed_delivered = true;
                Ok(TtsUpdate::Closed(TtsResult::clone(result)))
            }
            None => Err(RecvError::Empty),
        }
    }
}

/// The text side of a synthesis: push text into it, and end it.
///
/// [`TtsEngine::start`](super::TtsEngine::start) returns it with its
/// [`TtsOutput`]. Dropping it before [`close_text`](Self::close_text)
/// cancels the synthesis; after, it changes nothing, and the text is
/// synthesized to the end (T-02).
pub struct TtsSession {
    shared: Arc<Shared>,
}

impl TtsSession {
    /// The session's ID.
    pub fn id(&self) -> u64 {
        self.shared.id
    }

    /// The rate of the audio, which is always mono: the rate set with
    /// `with_sample_rate`, or the backend's own (T-01).
    pub fn sample_rate(&self) -> SampleRate {
        self.shared.sample_rate
    }

    /// Adds text to synthesize, in pieces of any size, such as an LLM's
    /// tokens (T-08). Text is synthesized a sentence at a time, as each is
    /// completed. Text pushed while the backend opens waits for it.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] beyond `max_text_chars`, or
    /// [`SpeechError::Closed`] after `close_text` or once the synthesis
    /// ended.
    pub fn push_text(&self, text: &str) -> Result<(), SpeechError> {
        let shared = &*self.shared;
        let mut core = shared.lock();
        if core.text_closed || shared.terminal.get().is_some() {
            return Err(SpeechError::Closed);
        }
        let chars = text.chars().count();
        if core.chars + chars > shared.max_text_chars {
            return Err(SpeechError::InvalidInput(format!(
                "text exceeds the {}-character limit",
                shared.max_text_chars
            )));
        }
        core.chars += chars;
        core.text.push_str(text);
        core.pushed.push_str(text);
        drop(core);
        shared.changed.notify_all();
        Ok(())
    }

    /// Ends the text. What was pushed is synthesized to the end.
    pub fn close_text(&self) {
        self.shared.lock().text_closed = true;
        self.shared.changed.notify_all();
    }

    /// Closes the text and waits until the synthesis has ended and the
    /// reader of its [`TtsOutput`] has taken all its audio. Every call
    /// returns the same result (T-06).
    ///
    /// # Errors
    ///
    /// The synthesis's failure, with the progress made before it (T-09).
    /// If `deadline` passes first, the synthesis fails with
    /// [`SpeechError::DeadlineExceeded`].
    pub fn finish(&self, deadline: impl Into<Deadline>) -> TtsResult {
        let deadline = deadline.into();
        self.close_text();
        let shared = &*self.shared;
        let (core, done) = deadline::wait_until(&shared.changed, shared.lock(), deadline, |_| {
            shared.terminal.get().is_some()
        });
        if !done {
            shared.fail(&core, SpeechError::DeadlineExceeded);
        }
        drop(core);
        // A failure always stores a result, so the fallback is never used.
        self.result()
            .cloned()
            .unwrap_or_else(|| Err(TtsFailure::new(SpeechError::Closed)))
    }

    /// Cancels the synthesis. Its result becomes [`SpeechError::Cancelled`],
    /// unless it already ended. The slot is freed once the backend call in
    /// progress returns (T-07).
    pub fn cancel(&self) {
        let shared = &*self.shared;
        let core = shared.lock();
        shared.fail(&core, SpeechError::Cancelled);
    }

    /// The result, once the synthesis has ended. It is borrowed, so
    /// polling does not copy it.
    pub fn result(&self) -> Option<&TtsResult> {
        self.shared.terminal.get().map(AsRef::as_ref)
    }

    /// Audio waiting in the output queue.
    pub fn queued(&self) -> Duration {
        let queued = self.shared.lock().queued;
        self.shared.sample_rate.duration_of(queued as u64)
    }
}

impl Drop for TtsSession {
    fn drop(&mut self) {
        let shared = &*self.shared;
        let core = shared.lock();
        if !core.text_closed {
            shared.fail(&core, SpeechError::Cancelled);
        }
    }
}

impl std::fmt::Debug for TtsSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsSession")
            .field("id", &self.shared.id)
            .field("sample_rate", &self.shared.sample_rate)
            .finish_non_exhaustive()
    }
}

/// The audio side of a synthesis: its audio, its marks, and finally
/// [`TtsUpdate::Closed`] with the result.
///
/// It has one reader, and nothing is dropped: synthesis waits while the
/// output queue is full (T-01). A mark follows the last audio of its text,
/// and `Closed` is always last (T-03). Dropping the output cancels the
/// synthesis, since nobody can hear it any more (T-02).
///
/// As an [`Iterator`], it waits for each update without a deadline and
/// ends after `Closed`; [`recv`](Self::recv) bounds the wait.
pub struct TtsOutput {
    shared: Arc<Shared>,
}

impl TtsOutput {
    /// The rate of the audio.
    pub fn sample_rate(&self) -> SampleRate {
        self.shared.sample_rate
    }

    /// Waits for the next update until `deadline`. A timeout changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`RecvError::Timeout`] if the deadline passed first, or
    /// [`RecvError::Closed`] after `Closed` was delivered.
    pub fn recv(&mut self, deadline: impl Into<Deadline>) -> Result<TtsUpdate, RecvError> {
        let deadline = deadline.into();
        let shared = &*self.shared;
        let mut taken = Err(RecvError::Empty);
        let _core = deadline::wait_until(&shared.changed, shared.lock(), deadline, |core| {
            taken = shared.take(core);
            !matches!(taken, Err(RecvError::Empty))
        });
        taken.map_err(|error| match error {
            RecvError::Empty => RecvError::Timeout,
            other => other,
        })
    }

    /// The next update, without waiting.
    ///
    /// # Errors
    ///
    /// [`RecvError::Empty`] if none is ready, or [`RecvError::Closed`] after
    /// `Closed` was delivered.
    pub fn try_recv(&mut self) -> Result<TtsUpdate, RecvError> {
        let shared = &*self.shared;
        let mut core = shared.lock();
        shared.take(&mut core)
    }

    /// The text pushed into this synthesis, for a player that reports the
    /// text whose audio has played.
    #[cfg(feature = "devices")]
    pub(crate) fn pushed_text(&self) -> PushedText {
        PushedText(self.shared.clone())
    }

    /// The most samples the output queue ever held, for the T-01 checks.
    #[doc(hidden)]
    pub fn peak_queued_samples(&self) -> usize {
        self.shared.lock().peak
    }

    /// The output queue's capacity in samples, for the T-01 checks.
    #[doc(hidden)]
    pub fn queue_capacity(&self) -> usize {
        self.shared.capacity
    }
}

impl Iterator for TtsOutput {
    type Item = TtsUpdate;

    /// Waits for the next update. `None` after `Closed`.
    fn next(&mut self) -> Option<TtsUpdate> {
        let shared = &*self.shared;
        let mut taken = Err(RecvError::Empty);
        let _core = deadline::wait_forever(&shared.changed, shared.lock(), |core| {
            taken = shared.take(core);
            !matches!(taken, Err(RecvError::Empty))
        });
        taken.ok()
    }
}

impl Drop for TtsOutput {
    fn drop(&mut self) {
        let shared = &*self.shared;
        let mut core = shared.lock();
        shared.fail(&core, SpeechError::Cancelled);
        core.output.clear();
        core.queued = 0;
        drop(core);
        shared.changed.notify_all();
    }
}

impl std::fmt::Debug for TtsOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsOutput")
            .field("id", &self.shared.id)
            .field("sample_rate", &self.shared.sample_rate)
            .finish_non_exhaustive()
    }
}

/// The text pushed into a synthesis, from [`TtsOutput::pushed_text`].
#[cfg(feature = "devices")]
pub(crate) struct PushedText(Arc<Shared>);

#[cfg(feature = "devices")]
impl PushedText {
    /// The first `end` bytes of the text pushed, as a mark's range names
    /// them.
    pub(crate) fn prefix(&self, end: usize) -> String {
        let core = self.0.lock();
        let text = &core.pushed;
        let mut end = end.min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.get(..end).unwrap_or_default().to_owned()
    }
}

/// The two halves of a new session.
pub(crate) fn halves(shared: &Arc<Shared>) -> (TtsSession, TtsOutput) {
    (
        TtsSession {
            shared: shared.clone(),
        },
        TtsOutput {
            shared: shared.clone(),
        },
    )
}

/// Everything a session's thread owns.
pub(crate) struct Worker {
    pub(crate) shared: Arc<Shared>,
    pub(crate) backend: Arc<dyn TtsBackend>,
    pub(crate) options: TtsOptions,
    /// The backend's rate.
    pub(crate) from: SampleRate,
    /// The rate of the output. The thread builds the resampler between the
    /// two once it holds a slot, so `start` does no CPU-heavy work.
    pub(crate) to: SampleRate,
    pub(crate) chunker: Chunker,
    /// A slot the caller already took, or `None` to wait for one.
    pub(crate) slot: Option<SlotGuard>,
}

impl Worker {
    /// Waits for a slot, opens the stream, and synthesizes.
    pub(crate) fn run(self) {
        let Self {
            shared,
            backend,
            options,
            from,
            to,
            chunker,
            slot,
        } = self;
        let ended = || shared.terminal.get().is_some();
        let slot = match slot {
            Some(slot) => slot,
            None => match shared.slots.acquire(ended) {
                Some(slot) => slot,
                None => return,
            },
        };
        {
            let mut core = shared.lock();
            if ended() {
                return;
            }
            core.stage = Stage::Backend;
        }
        let name = backend.name().to_owned();
        // A failure to build the resampler ends the session like a failure to
        // open the stream.
        let opened = Resampler::new(from, to).and_then(|resampler| {
            if ended() {
                return Err(SpeechError::Closed);
            }
            let stream =
                catch_unwind(AssertUnwindSafe(|| backend.open(&options))).unwrap_or_else(|_| {
                    Err(SpeechError::backend(
                        name.clone(),
                        false,
                        "the backend panicked while opening",
                    ))
                })?;
            Ok((resampler, stream))
        });
        let (resampler, stream) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                // The slot goes before the result is published (T-11).
                drop(slot);
                let core = shared.lock();
                shared.fail(&core, error);
                return;
            }
        };
        {
            let mut core = shared.lock();
            if !ended() {
                core.stage = Stage::Open;
                tracing::debug!(session = shared.id, backend = %name, "synthesis started");
            }
        }
        shared.changed.notify_all();
        Running {
            shared,
            backend: name,
            stream,
            resampler,
            chunker,
            slot,
        }
        .run();
    }
}

/// A session with an open stream.
struct Running {
    shared: Arc<Shared>,
    backend: String,
    stream: Box<dyn TtsStream>,
    resampler: Resampler,
    chunker: Chunker,
    /// Released after the last backend call (T-07), but before the result
    /// is published (T-11).
    slot: SlotGuard,
}

impl Running {
    fn run(mut self) {
        let outcome = catch_unwind(AssertUnwindSafe(|| self.drive()));
        let Self {
            shared,
            backend,
            mut stream,
            slot,
            ..
        } = self;
        let result = match outcome {
            Ok(Ok(Some(summary))) => Some(Ok(summary)),
            Ok(Ok(None)) => None,
            Ok(Err(error)) => Some(Err(error)),
            Err(_) => Some(Err(SpeechError::backend(
                backend,
                false,
                "the backend panicked",
            ))),
        };
        // Cancelling a failed stream is the last backend call. After it,
        // the stream and the slot go before the result is published, so a
        // caller woken by `finish` can start the next session at once
        // (T-11). If the session already ended from outside, the slot was
        // held until now (T-07).
        let failed =
            matches!(result, Some(Err(_))) || shared.terminal.get().is_some_and(|r| r.is_err());
        if failed {
            let _ = catch_unwind(AssertUnwindSafe(|| stream.cancel()));
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(stream)));
        drop(slot);
        let mut core = shared.lock();
        match result {
            Some(Ok(summary)) => {
                core.done = Some(summary);
                shared.try_complete(&mut core);
                shared.changed.notify_all();
            }
            Some(Err(error)) => shared.fail(&core, error),
            None => {}
        }
    }

    /// Runs until the text ends. `Ok(None)` means the session was ended
    /// from outside.
    fn drive(&mut self) -> Result<Option<TtsSummary>, SpeechError> {
        let shared = self.shared.clone();
        let rate = shared.sample_rate;
        let mut marks = Vec::new();
        let mut spoke = false;
        loop {
            let (text, closed) = {
                let mut core = deadline::wait_forever(&shared.changed, shared.lock(), |core| {
                    shared.terminal.get().is_some() || !core.text.is_empty() || core.text_closed
                });
                if shared.terminal.get().is_some() {
                    return Ok(None);
                }
                (std::mem::take(&mut core.text), core.text_closed)
            };
            let mut ready = self.chunker.push(&text);
            if closed {
                ready.extend(self.chunker.flush());
            }
            for chunk in ready {
                if chunk.is_blank() {
                    shared.lock().text_done = chunk.range.end;
                    continue;
                }
                spoke = true;
                tracing::trace!(session = shared.id, text = %chunk.text, "synthesizing");
                let start = self.resampler.wanted();
                let backend = self.backend.clone();
                let resampler = &mut self.resampler;
                let stream = &mut self.stream;
                let mut failed = None;
                let synthesized = catch_unwind(AssertUnwindSafe(|| {
                    let mut sink = |audio: &[f32]| emit(&shared, resampler, &mut failed, audio);
                    stream.synthesize(&chunk.text, &mut sink)
                }))
                .unwrap_or_else(|_| {
                    Err(SpeechError::backend(backend, false, "the backend panicked"))
                });
                // A resampling error is why the sink said `Stop`, so it wins
                // over what the backend returned.
                if let Some(error) = failed {
                    return Err(error);
                }
                synthesized?;
                if shared.terminal.get().is_some() {
                    return Ok(None);
                }
                let end = self.resampler.wanted();
                let mark = Mark {
                    text: chunk.range.clone(),
                    audio: rate.duration_of(start)..rate.duration_of(end),
                };
                marks.push(mark.clone());
                let mut core = shared.lock();
                core.text_done = chunk.range.end;
                core.pending_marks.push_back((mark, end));
                release_marks(&mut core);
                drop(core);
                shared.changed.notify_all();
            }
            if closed {
                if !spoke {
                    return Err(SpeechError::InvalidInput(
                        "there is no text to synthesize".into(),
                    ));
                }
                let mut tail = Vec::new();
                self.resampler.flush(&mut tail)?;
                if queue(&shared, &tail) == Flow::Stop {
                    return Ok(None);
                }
                let mut core = shared.lock();
                while let Some((mark, _)) = core.pending_marks.pop_front() {
                    core.output.push_back(Item::Mark(mark));
                }
                return Ok(Some(TtsSummary {
                    duration: rate.duration_of(core.emitted),
                    marks,
                }));
            }
        }
    }
}

/// Queues the marks whose audio is all queued.
fn release_marks(core: &mut Core) {
    while core
        .pending_marks
        .front()
        .is_some_and(|(_, end)| *end <= core.emitted)
    {
        if let Some((mark, _)) = core.pending_marks.pop_front() {
            core.output.push_back(Item::Mark(mark));
        }
    }
}

/// The sink: resample and queue, blocking while the queue is full. If the
/// resampler fails, it keeps the error in `failed` and stops the backend;
/// `drive` then fails the session, once the backend call has returned, so
/// the slot is free before the result is published (T-11).
fn emit(
    shared: &Shared,
    resampler: &mut Resampler,
    failed: &mut Option<SpeechError>,
    audio: &[f32],
) -> Flow {
    if shared.terminal.get().is_some() {
        return Flow::Stop;
    }
    let mut out = Vec::with_capacity(audio.len());
    if let Err(error) = resampler.process(audio, &mut out) {
        *failed = Some(error);
        return Flow::Stop;
    }
    queue(shared, &out)
}

/// Queues audio in pieces no larger than the queue, waiting for space.
fn queue(shared: &Shared, audio: &[f32]) -> Flow {
    let capacity = shared.capacity;
    for piece in audio.chunks(capacity) {
        let needed = piece.len();
        let mut core = deadline::wait_forever(&shared.changed, shared.lock(), |core| {
            shared.terminal.get().is_some() || core.queued + needed <= capacity
        });
        if shared.terminal.get().is_some() {
            return Flow::Stop;
        }
        core.queued += needed;
        core.peak = core.peak.max(core.queued);
        core.emitted += needed as u64;
        core.output.push_back(Item::Audio(piece.to_vec()));
        release_marks(&mut core);
        drop(core);
        shared.changed.notify_all();
    }
    Flow::Continue
}

pub(crate) fn capacity(rate: SampleRate, queue: Duration) -> usize {
    usize::try_from(rate.frames_in(queue)).unwrap_or(usize::MAX)
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    /// A stream that makes one piece of audio for each chunk.
    struct Once;

    impl TtsStream for Once {
        fn synthesize(
            &mut self,
            _: &str,
            sink: &mut dyn FnMut(&[f32]) -> Flow,
        ) -> Result<(), SpeechError> {
            // A backend stops when the sink says so, and returns `Ok`.
            let _ = sink(&[0.1; 100]);
            Ok(())
        }
    }

    /// A resampler that fails on the first audio fails the synthesis with
    /// its error (T-01, T-09): the audio was not all produced, so the
    /// session must not finish `Ok`. The error is returned rather than
    /// published at once, so the worker frees the slot first (T-11).
    #[test]
    fn a_failed_resample_fails_the_synthesis() {
        let rate = SampleRate::HZ_16000;
        let slots = SlotLimiter::new(NonZeroUsize::MIN);
        let queue = capacity(rate, Duration::from_secs(2));
        let shared = Arc::new(Shared::new(1, rate, queue, 1_000, slots.clone()));
        let (session, _output) = halves(&shared);
        session.push_text("hello").unwrap();
        session.close_text();
        // A flushed resampler refuses more audio.
        let mut resampler = Resampler::new(rate, rate).unwrap();
        resampler.flush(&mut Vec::new()).unwrap();
        let mut running = Running {
            shared: shared.clone(),
            backend: "once".into(),
            stream: Box::new(Once),
            resampler,
            chunker: Chunker::new(100),
            slot: slots.try_acquire().unwrap(),
        };
        let error = running.drive().expect_err("the resampler failed");
        assert!(matches!(error, SpeechError::Closed), "{error:?}");
        assert!(shared.terminal.get().is_none());
    }
}
