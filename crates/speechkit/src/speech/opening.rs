//! Opening a session, shared by recognition and synthesis: how it gets its
//! slot, how far it has got, and the wait until it is open.
//!
//! `start` does no CPU-heavy work before it has dealt with its deadline and
//! its slot, so a caller with a short deadline and a busy engine gets
//! `Capacity` promptly (A-14). What is costly, such as building a
//! resampler, happens on the session's thread once the slot is held.

use super::{
    deadline::{self, Deadline},
    error::SpeechError,
    slots::{SlotGuard, SlotLimiter},
    sync::{Condvar, MutexGuard},
};

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
/// `stage` reads the session's stage from its locked state `C`. `ended`
/// gives the error to return if the session already has a result, which is
/// `Closed` for a success. When the deadline passes first, `fail` settles
/// the session with the error that fits the stage, `Capacity` while it
/// waits for a slot (retryable) and `DeadlineExceeded` once it holds one,
/// and that error is returned too.
///
/// # Errors
///
/// The session's own error, or the one described above.
pub(crate) fn wait_open<C>(
    changed: &Condvar,
    core: MutexGuard<'_, C>,
    deadline: Deadline,
    stage: impl Fn(&C) -> Stage,
    ended: impl Fn() -> Option<SpeechError>,
    fail: impl FnOnce(MutexGuard<'_, C>, SpeechError),
) -> Result<(), SpeechError> {
    let (core, open) = deadline::wait_until(changed, core, deadline, |core| {
        stage(core) == Stage::Open || ended().is_some()
    });
    if let Some(error) = ended() {
        return Err(error);
    }
    if open {
        return Ok(());
    }
    let error = match stage(&core) {
        Stage::Slot => SpeechError::Capacity,
        Stage::Backend | Stage::Open => SpeechError::DeadlineExceeded,
    };
    fail(core, error.clone());
    Err(error)
}
