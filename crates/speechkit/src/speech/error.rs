//! The error type shared by every speechkit API.

use std::sync::Arc;

/// Errors reported by speechkit.
///
/// speechkit never retries or falls back to another backend on its own:
/// a session cannot be replayed without the audio it does not keep, and
/// another backend changes the transcript, timing, and cost.
/// [`SpeechError::retryable`] tells the caller whether repeating the same
/// request might succeed.
///
/// Clones share a backend error's source.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum SpeechError {
    /// The caller passed something invalid: options, audio, or text.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A model is missing, has the wrong layout, or failed to load.
    #[error("invalid model: {0}")]
    InvalidModel(String),
    /// The backend or this build does not support the request.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A deadline passed before the operation finished.
    #[error("deadline exceeded")]
    DeadlineExceeded,
    /// The operation was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// Every execution slot is in use, or a session's input queue is full.
    #[error("capacity exhausted")]
    Capacity,
    /// The session or stream is already closed.
    #[error("closed")]
    Closed,
    /// A backend failed.
    #[error("backend `{backend}` failed")]
    Backend {
        /// The backend's name.
        backend: String,
        /// Whether repeating the request might succeed.
        retryable: bool,
        /// What went wrong. The `Box` keeps
        /// [`source`](std::error::Error::source) returning the backend's own
        /// error, so it can be downcast; an `Arc` alone would be returned
        /// itself.
        #[source]
        source: Arc<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl SpeechError {
    /// Whether repeating the same request might succeed: true for
    /// [`Capacity`](Self::Capacity) and for backend failures marked retryable.
    pub fn retryable(&self) -> bool {
        match self {
            Self::Backend { retryable, .. } => *retryable,
            Self::Capacity => true,
            _ => false,
        }
    }

    /// A [`Backend`](Self::Backend) error.
    pub fn backend(
        backend: impl Into<String>,
        retryable: bool,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Backend {
            backend: backend.into(),
            retryable,
            source: Arc::new(source.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_retryable() {
        let io = || std::io::Error::other("connection reset");
        let cases: Vec<(SpeechError, &str, bool)> = vec![
            (
                SpeechError::InvalidInput("x".into()),
                "invalid input: x",
                false,
            ),
            (
                SpeechError::InvalidModel("y".into()),
                "invalid model: y",
                false,
            ),
            (
                SpeechError::Unsupported("z".into()),
                "unsupported: z",
                false,
            ),
            (SpeechError::DeadlineExceeded, "deadline exceeded", false),
            (SpeechError::Cancelled, "cancelled", false),
            (SpeechError::Capacity, "capacity exhausted", true),
            (SpeechError::Closed, "closed", false),
            (
                SpeechError::backend("openai", true, io()),
                "backend `openai` failed",
                true,
            ),
            (
                SpeechError::backend("sherpa", false, io()),
                "backend `sherpa` failed",
                false,
            ),
        ];
        for (error, display, retryable) in cases {
            assert_eq!(error.to_string(), display);
            assert_eq!(error.retryable(), retryable, "{error}");
            let copy = error.clone();
            assert_eq!(copy.to_string(), display);
            assert_eq!(copy.retryable(), retryable);
        }
    }

    #[test]
    fn backend_error_keeps_its_source() {
        let error = SpeechError::backend("openai", true, "HTTP 503");
        let source = std::error::Error::source(&error).unwrap();
        assert_eq!(source.to_string(), "HTTP 503");
    }

    #[test]
    fn backend_source_downcasts_in_clones_too() {
        let error = SpeechError::backend("openai", true, std::io::Error::other("reset"));
        for error in [error.clone(), error] {
            let source = std::error::Error::source(&error).unwrap();
            assert!(source.downcast_ref::<std::io::Error>().is_some());
        }
    }
}
