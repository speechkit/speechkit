//! The session's bounded input queue.

use super::AsrSession;
use crate::{
    Deadline, SpeechError,
    speech::{audio::find_invalid, deadline},
};

/// Why a push was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PushErrorKind {
    /// The input queue had no room before the deadline.
    #[error("input queue is full")]
    Full,
    /// The chunk is longer than the input queue.
    #[error("chunk is longer than the input queue")]
    TooLarge,
    /// The sample at `index` is NaN, infinite, or outside [-1.0, 1.0].
    #[error("sample {index} is not a finite value in [-1.0, 1.0]")]
    Invalid {
        /// Position of the first bad sample.
        index: usize,
    },
    /// The session takes no more input.
    #[error("session input is closed")]
    Closed,
}

/// A refused push, which always hands the chunk back unchanged: a chunk is
/// taken whole or not at all (A-01). Audio a session refused because it
/// had just closed can go to another session.
#[derive(PartialEq, thiserror::Error)]
#[error("{kind}")]
#[non_exhaustive]
pub struct PushError {
    /// Why the push was refused.
    pub kind: PushErrorKind,
    /// The refused chunk.
    pub chunk: Vec<f32>,
}

impl PushError {
    fn new(kind: PushErrorKind, chunk: Vec<f32>) -> Self {
        Self { kind, chunk }
    }

    /// The refused chunk.
    pub fn into_chunk(self) -> Vec<f32> {
        self.chunk
    }
}

impl std::fmt::Debug for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PushError")
            .field("kind", &self.kind)
            .field("chunk_len", &self.chunk.len())
            .finish()
    }
}

/// For `?`. The refused audio is dropped. `Full` becomes
/// [`SpeechError::Capacity`], which is retryable: the queue has room again
/// once the session catches up. `TooLarge` and `Invalid` become
/// [`SpeechError::InvalidInput`], and `Closed` stays
/// [`SpeechError::Closed`].
impl From<PushError> for SpeechError {
    fn from(error: PushError) -> Self {
        match error.kind {
            PushErrorKind::Full => Self::Capacity,
            PushErrorKind::TooLarge | PushErrorKind::Invalid { .. } => {
                Self::InvalidInput(error.kind.to_string())
            }
            PushErrorKind::Closed => Self::Closed,
        }
    }
}

impl AsrSession {
    /// Queues `chunk` if it fits, without waiting. The chunk is mono audio
    /// at the session's [`sample_rate`](Self::sample_rate), as a `Vec<f32>`
    /// or a slice.
    ///
    /// An empty chunk is accepted and does nothing (A-17).
    ///
    /// # Errors
    ///
    /// A [`PushError`] holding the chunk (A-01). A full queue refuses at
    /// once (A-02).
    pub fn try_push(&self, chunk: impl Into<Vec<f32>>) -> Result<(), PushError> {
        self.push_inner(chunk.into(), None)
    }

    /// Queues `chunk`, waiting until there is room, the deadline passes, or
    /// the session ends (A-02).
    ///
    /// # Errors
    ///
    /// As [`try_push`](Self::try_push). `Full` means the deadline passed.
    pub fn push(
        &self,
        chunk: impl Into<Vec<f32>>,
        deadline: impl Into<Deadline>,
    ) -> Result<(), PushError> {
        self.push_inner(chunk.into(), Some(deadline.into()))
    }

    fn push_inner(&self, chunk: Vec<f32>, deadline: Option<Deadline>) -> Result<(), PushError> {
        let shared = &*self.shared;
        if let Some(bad) = find_invalid(&chunk) {
            return Err(PushError::new(
                PushErrorKind::Invalid { index: bad.index },
                chunk,
            ));
        }
        if chunk.len() > shared.capacity {
            return Err(PushError::new(PushErrorKind::TooLarge, chunk));
        }
        let open = |core: &super::Core| !core.input_closed && !shared.stopping(core);
        let len = chunk.len();
        let fits = |core: &super::Core| core.queued + len <= shared.capacity;
        let mut core = shared.lock();
        if !open(&core) {
            return Err(PushError::new(PushErrorKind::Closed, chunk));
        }
        if chunk.is_empty() {
            return Ok(());
        }
        if !fits(&core) {
            let Some(deadline) = deadline else {
                return Err(PushError::new(PushErrorKind::Full, chunk));
            };
            let (waited, ready) = deadline::wait_until(&shared.changed, core, deadline, |core| {
                !open(core) || fits(core)
            });
            core = waited;
            if !open(&core) {
                return Err(PushError::new(PushErrorKind::Closed, chunk));
            }
            if !ready {
                return Err(PushError::new(PushErrorKind::Full, chunk));
            }
        }
        core.queued += chunk.len();
        core.queue.push_back(chunk);
        drop(core);
        shared.changed.notify_all();
        Ok(())
    }

    /// Closes the input: later pushes fail with [`PushErrorKind::Closed`].
    /// Queued audio is still processed; [`finish`](Self::finish) waits for
    /// the result.
    pub fn close_input(&self) {
        let shared = &*self.shared;
        shared.lock().input_closed = true;
        shared.changed.notify_all();
    }
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use super::*;

    #[test]
    fn push_errors_convert_for_the_question_mark() {
        let full = SpeechError::from(PushError::new(PushErrorKind::Full, vec![0.0]));
        assert!(matches!(full, SpeechError::Capacity) && full.retryable());
        for kind in [PushErrorKind::TooLarge, PushErrorKind::Invalid { index: 3 }] {
            assert!(matches!(
                SpeechError::from(PushError::new(kind, vec![0.0])),
                SpeechError::InvalidInput(_)
            ));
        }
        let closed = PushError::new(PushErrorKind::Closed, vec![0.5]);
        assert_eq!(closed.to_string(), "session input is closed");
        assert_eq!(closed.clone_chunk(), [0.5]);
        assert!(matches!(SpeechError::from(closed), SpeechError::Closed));
    }

    impl PushError {
        fn clone_chunk(&self) -> Vec<f32> {
            self.chunk.clone()
        }
    }
}
