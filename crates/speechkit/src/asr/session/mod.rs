//! A running recognition session.

mod endpoint;
mod events;
mod history;
mod input;
mod worker;

use std::{
    collections::VecDeque,
    sync::{Arc, OnceLock},
    time::Duration,
};

pub(crate) use endpoint::Endpoints;
pub use events::AsrEvents;
pub use history::AsrUpdates;
use history::History;
pub use input::{PushError, PushErrorKind};
pub(crate) use worker::Worker;

use super::{AsrFailure, AsrResult, Transcript, post::PostProcessing};
use crate::{
    Deadline, SampleRate, SpeechError,
    speech::{
        deadline,
        opening::Stage,
        slots::SlotLimiter,
        sync::{Condvar, Mutex, MutexGuard, lock},
    },
};

/// Settings fixed when a session starts.
pub(crate) struct Settings {
    pub(crate) id: u64,
    /// The rate of the audio pushed.
    pub(crate) sample_rate: SampleRate,
    /// Added to every time a backend reports.
    pub(crate) origin: Duration,
    /// Input queue capacity, in frames at `sample_rate`.
    pub(crate) capacity: usize,
    pub(crate) max_history_bytes: usize,
    pub(crate) post: Option<PostProcessing>,
    pub(crate) slots: SlotLimiter,
    /// Where turns end and where the session stops.
    pub(crate) endpoints: Endpoints,
}

/// State shared by the session handle, its readers, its events, and its
/// thread.
pub(crate) struct Shared {
    pub(crate) id: u64,
    /// The rate of the audio pushed.
    pub(crate) sample_rate: SampleRate,
    /// Added to every time a backend reports, so times count from the
    /// origin's point in a longer stream, such as a capture.
    pub(crate) origin: Duration,
    /// Input queue capacity, in frames at `sample_rate`.
    pub(crate) capacity: usize,
    pub(crate) max_history_bytes: usize,
    /// Serializes post-processing, so segments stay in order.
    pub(crate) post: Option<Mutex<PostProcessing>>,
    /// Woken when the session ends, so a thread waiting for a slot gives up.
    pub(crate) slots: SlotLimiter,
    pub(crate) core: Mutex<Core>,
    /// Signalled on every change: queue space, new audio, updates, the end.
    pub(crate) changed: Condvar,
    /// The result, set exactly once while holding `core`.
    pub(crate) terminal: OnceLock<Arc<AsrResult>>,
}

/// Mutable session state, guarded by `Shared::core`.
pub(crate) struct Core {
    pub(crate) stage: Stage,
    pub(crate) input_closed: bool,
    pub(crate) queue: VecDeque<Vec<f32>>,
    /// Frames in `queue`.
    pub(crate) queued: usize,
    /// Frames taken from the queue for the backend, at `sample_rate`.
    pub(crate) consumed: u64,
    pub(crate) history: History,
    /// A failure the stream or the history reported, which the session
    /// thread publishes once the stream is gone.
    pub(crate) failure: Option<SpeechError>,
    pub(crate) endpoints: Endpoints,
    /// The end of the audio fed to the backend so far, including the block
    /// being fed, from the origin.
    pub(crate) position: Duration,
    /// Where the session stopped taking audio, C, once it ended itself
    /// (A-07).
    pub(crate) cutoff: Option<Duration>,
}

impl Shared {
    pub(crate) fn new(settings: Settings) -> Self {
        Self {
            id: settings.id,
            sample_rate: settings.sample_rate,
            origin: settings.origin,
            capacity: settings.capacity.max(1),
            max_history_bytes: settings.max_history_bytes,
            post: settings.post.map(Mutex::new),
            slots: settings.slots,
            core: Mutex::new(Core {
                stage: Stage::Slot,
                input_closed: false,
                queue: VecDeque::new(),
                queued: 0,
                consumed: 0,
                history: History::default(),
                failure: None,
                endpoints: settings.endpoints,
                position: settings.origin,
                cutoff: None,
            }),
            changed: Condvar::new(),
            terminal: OnceLock::new(),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Core> {
        lock(&self.core)
    }

    /// Whether the session has ended, or is ending with a failure.
    pub(crate) fn stopping(&self, core: &Core) -> bool {
        self.terminal.get().is_some() || core.failure.is_some()
    }

    /// The transcript so far. After a cutoff, its duration ends there.
    pub(crate) fn transcript(&self, core: &Core) -> Transcript {
        let duration = match core.cutoff {
            Some(cutoff) => cutoff.saturating_sub(self.origin),
            None => self.sample_rate.duration_of(core.consumed),
        };
        Transcript {
            segments: core.history.segments(),
            duration,
        }
    }

    /// Stops taking audio at `at`, the cutoff: the input closes, and audio
    /// queued past it is dropped unheard. Only the first cutoff counts.
    pub(crate) fn cut(&self, core: &mut Core, at: Duration) {
        if core.cutoff.is_some() {
            return;
        }
        core.cutoff = Some(at);
        core.input_closed = true;
        core.queue.clear();
        core.queued = 0;
        self.changed.notify_all();
    }

    /// Fixes the result. Later calls do nothing, so the first result wins
    /// and never changes (A-03).
    pub(crate) fn settle(&self, core: &mut Core, result: AsrResult) {
        if self.terminal.get().is_some() {
            return;
        }
        // A session keeps no audio after it ends (A-06).
        core.queue.clear();
        core.queued = 0;
        core.history.abandon_partials();
        match &result {
            Ok(_) => tracing::debug!(session = self.id, "session completed"),
            Err(failure) => {
                tracing::debug!(session = self.id, error = %failure.error, "session failed");
            }
        }
        let _ = self.terminal.set(Arc::new(result));
        self.changed.notify_all();
        self.slots.wake();
    }

    /// Fails the session, keeping what it confirmed (A-04).
    pub(crate) fn settle_failure(&self, core: &mut Core, error: SpeechError) {
        let failure = AsrFailure {
            error,
            confirmed: self.transcript(core),
        };
        self.settle(core, Err(failure));
    }
}

/// One recognition session.
///
/// Push audio with [`push`](Self::push) or [`try_push`](Self::try_push),
/// read its progress with [`updates`](Self::updates), and get the result
/// with [`finish`](Self::finish). Dropping the session cancels it (A-13).
/// The session is `Send + Sync`: share it, for example in an `Arc`, to push
/// from one thread and finish from another.
pub struct AsrSession {
    pub(crate) shared: Arc<Shared>,
}

impl AsrSession {
    /// The session's ID, unique within the process.
    pub fn id(&self) -> u64 {
        self.shared.id
    }

    /// The rate of the audio this session accepts.
    pub fn sample_rate(&self) -> SampleRate {
        self.shared.sample_rate
    }

    /// A new reader of the session's updates. There may be any number.
    pub fn updates(&self) -> AsrUpdates {
        let cursor = self.shared.lock().history.cursor();
        AsrUpdates {
            shared: self.shared.clone(),
            cursor,
        }
    }

    /// Closes the input, then waits for the result: queued audio is still
    /// processed. Every call returns the same result (A-03).
    ///
    /// # Errors
    ///
    /// The session's failure, with what it confirmed (A-04). If `deadline`
    /// passes first, the session fails with
    /// [`SpeechError::DeadlineExceeded`] for everyone holding it, and this
    /// returns promptly. The slot stays held until the backend call in
    /// progress returns (A-09).
    pub fn finish(&self, deadline: impl Into<Deadline>) -> AsrResult {
        let deadline = deadline.into();
        self.close_input();
        let shared = &*self.shared;
        let (mut core, done) =
            deadline::wait_until(&shared.changed, shared.lock(), deadline, |_| {
                shared.terminal.get().is_some()
            });
        if !done {
            shared.settle_failure(&mut core, SpeechError::DeadlineExceeded);
        }
        drop(core);
        // A failure always stores a result, so the fallback is never used.
        self.result()
            .cloned()
            .unwrap_or_else(|| Err(AsrFailure::new(SpeechError::Closed)))
    }

    /// The result once the session has ended by itself, waiting until
    /// `deadline`, or `None` if it is still running. It changes nothing.
    pub fn wait(&self, deadline: impl Into<Deadline>) -> Option<AsrResult> {
        let shared = &*self.shared;
        let _core = deadline::wait_until(&shared.changed, shared.lock(), deadline.into(), |_| {
            shared.terminal.get().is_some()
        });
        self.result().cloned()
    }

    /// Cancels the session. Its result becomes [`SpeechError::Cancelled`],
    /// unless it already ended.
    pub fn cancel(&self) {
        self.fail(SpeechError::Cancelled);
    }

    /// The result, once the session has ended. It is borrowed, so polling
    /// does not copy the transcript.
    pub fn result(&self) -> Option<&AsrResult> {
        self.shared.terminal.get().map(AsRef::as_ref)
    }

    /// Audio waiting in the input queue.
    pub fn queued(&self) -> Duration {
        let queued = self.shared.lock().queued;
        self.shared.sample_rate.duration_of(queued as u64)
    }

    /// The longest chunk one push takes, in frames.
    #[cfg(feature = "devices")]
    pub(crate) fn max_chunk_frames(&self) -> usize {
        self.shared.capacity
    }

    /// Where the session stopped taking audio, C, once it ended itself
    /// (A-07), as a time from its origin's point in the longer stream.
    #[cfg(feature = "devices")]
    pub(crate) fn cutoff(&self) -> Option<Duration> {
        self.shared.lock().cutoff
    }

    /// Ends the session with `error`, keeping what it confirmed (A-04),
    /// unless it already ended.
    pub(crate) fn fail(&self, error: SpeechError) {
        let shared = &*self.shared;
        let mut core = shared.lock();
        shared.settle_failure(&mut core, error);
    }
}

impl Drop for AsrSession {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl std::fmt::Debug for AsrSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsrSession")
            .field("id", &self.shared.id)
            .field("sample_rate", &self.shared.sample_rate)
            .finish_non_exhaustive()
    }
}
