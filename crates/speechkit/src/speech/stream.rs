//! Types shared by the streams of recognition and synthesis.

/// Why a read from a stream returned no item.
///
/// Deliberately exhaustive: callers match on it to decide whether to wait
/// again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RecvError {
    /// No item is available right now.
    #[error("no update available")]
    Empty,
    /// The deadline passed with no item.
    #[error("deadline exceeded")]
    Timeout,
    /// `Closed` was already delivered; no more items will come.
    #[error("session closed")]
    Closed,
}

/// What a backend's sink wants the backend to do next.
///
/// Deliberately exhaustive: backends match on it, so a new case must break
/// them rather than be silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Keep going.
    Continue,
    /// Stop as soon as possible and return `Ok`.
    Stop,
}
