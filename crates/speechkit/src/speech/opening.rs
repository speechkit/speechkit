//! Opening a session, shared by recognition and synthesis: how it gets its
//! slot, how far it has got, and the wait until it is open.
//!
//! `start` does no CPU-heavy work before it has dealt with its deadline and
//! its slot, so a caller with a short deadline and a busy engine gets
//! `Capacity` promptly (A-14). What is costly, such as building a
//! resampler, happens on the session's thread once the slot is held.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{
    audio::SampleRate,
    deadline::{self, Deadline},
    error::SpeechError,
    resample::Resampler,
    slots::{SlotGuard, SlotLimiter},
    sync::{Condvar, MutexGuard},
};

/// A session's shared state, as opening it sees it.
pub(crate) trait Opens {
    /// The state behind the session's lock.
    type Core;

    /// Locks the session's state.
    fn lock(&self) -> MutexGuard<'_, Self::Core>;

    /// Signalled on every change to the state.
    fn changed(&self) -> &Condvar;

    /// The engine's session slots.
    fn slots(&self) -> &SlotLimiter;

    /// How far the session has got in opening.
    fn stage(core: &mut Self::Core) -> &mut Stage;

    /// Whether the session has a result.
    fn ended(&self) -> bool;

    /// The error a caller waiting for the open gets once the session has a
    /// result: its error, or `Closed` for a success.
    fn ended_error(&self) -> Option<SpeechError>;

    /// Ends the session with `error`, keeping what it made.
    fn fail(&self, core: &mut Self::Core, error: SpeechError);
}

/// Runs `run` on a new thread named `name`.
///
/// # Errors
///
/// A retryable [`SpeechError::Backend`] if the thread cannot be started.
pub(crate) fn spawn(name: String, run: impl FnOnce() + Send + 'static) -> Result<(), SpeechError> {
    std::thread::Builder::new()
        .name(name)
        .spawn(run)
        .map(drop)
        .map_err(|error| SpeechError::backend("speechkit", true, error))
}

/// Opens a session on its own thread. It waits for a slot unless `slot`
/// already is one, then builds the resampler from `from` to `to` and calls
/// `open`, catching a panic as a failure of `backend`.
///
/// Returns the slot, the resampler, and the stream once the session is
/// open. Returns `None` if the session ended first or failed to open; a
/// failure is published only after the slot is released (A-09, T-11).
pub(crate) fn open_stream<S: Opens, T>(
    shared: &S,
    slot: Option<SlotGuard>,
    (from, to): (SampleRate, SampleRate),
    backend: &str,
    open: impl FnOnce() -> Result<T, SpeechError>,
) -> Option<(SlotGuard, Resampler, T)> {
    let slot = match slot {
        Some(slot) => slot,
        None => shared.slots().acquire(|| shared.ended())?,
    };
    {
        let mut core = shared.lock();
        if shared.ended() {
            return None;
        }
        *S::stage(&mut core) = Stage::Backend;
    }
    // A failure to build the resampler ends the session like a failure to
    // open the stream.
    let opened = Resampler::new(from, to).and_then(|resampler| {
        if shared.ended() {
            return Err(SpeechError::Closed);
        }
        let stream = catch_unwind(AssertUnwindSafe(open)).unwrap_or_else(|_| {
            Err(SpeechError::backend(
                backend.to_owned(),
                false,
                "the backend panicked while opening",
            ))
        })?;
        Ok((resampler, stream))
    });
    match opened {
        Ok((resampler, stream)) => {
            {
                let mut core = shared.lock();
                if !shared.ended() {
                    *S::stage(&mut core) = Stage::Open;
                }
            }
            shared.changed().notify_all();
            Some((slot, resampler, stream))
        }
        Err(error) => {
            drop(slot);
            let mut core = shared.lock();
            shared.fail(&mut core, error);
            None
        }
    }
}

/// How a new session gets its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opening {
    /// Wait for a slot and the open within the deadline.
    Wait,
    /// Fail at once with `Capacity` when no slot is free, as a server
    /// answering 503 does; then wait for the open within the deadline.
    #[cfg_attr(
        not(any(feature = "server", test)),
        expect(dead_code, reason = "only the server, and the tests, fail at once")
    )]
    NoWait,
    /// Return at once; the session waits for its slot and opens in the
    /// background, and a failure to open becomes its result.
    Background,
}

/// How far a session has got in opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Waiting for a free slot.
    Slot,
    /// The session thread holds its slot and is preparing the backend
    /// (building the resampler) and opening its stream.
    Backend,
    /// Running.
    Open,
}

impl Opening {
    /// The slot a session starts with: one taken now for [`NoWait`], none
    /// for the others, whose thread waits for one.
    ///
    /// # Errors
    ///
    /// [`SpeechError::DeadlineExceeded`] if `deadline` has passed, except
    /// for [`Background`], which never waits (A-19); [`SpeechError::Capacity`]
    /// for [`NoWait`] when every slot is busy.
    ///
    /// [`NoWait`]: Self::NoWait
    /// [`Background`]: Self::Background
    pub(crate) fn take(
        self,
        slots: &SlotLimiter,
        deadline: Deadline,
    ) -> Result<Option<SlotGuard>, SpeechError> {
        if self != Self::Background && deadline.remaining().is_none() {
            return Err(SpeechError::DeadlineExceeded);
        }
        match self {
            Self::NoWait => Ok(Some(slots.try_acquire().ok_or(SpeechError::Capacity)?)),
            Self::Wait | Self::Background => Ok(None),
        }
    }
}

/// Waits until a session is open, or `deadline` passes.
///
/// When the deadline passes first, the session fails with the error that
/// fits its stage, `Capacity` while it waits for a slot (retryable) and
/// `DeadlineExceeded` once it holds one, and that error is returned too.
///
/// # Errors
///
/// The session's own error (`Closed` if it already succeeded), or the one
/// described above.
pub(crate) fn wait_open<S: Opens>(shared: &S, deadline: Deadline) -> Result<(), SpeechError> {
    let (mut core, open) =
        deadline::wait_until(shared.changed(), shared.lock(), deadline, |core| {
            *S::stage(core) == Stage::Open || shared.ended()
        });
    if let Some(error) = shared.ended_error() {
        return Err(error);
    }
    if open {
        return Ok(());
    }
    let error = match *S::stage(&mut core) {
        Stage::Slot => SpeechError::Capacity,
        Stage::Backend | Stage::Open => SpeechError::DeadlineExceeded,
    };
    shared.fail(&mut core, error.clone());
    Err(error)
}
