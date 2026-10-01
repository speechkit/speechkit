//! The Tokio runtime cloud backends run on.

use std::{future::Future, sync::Arc};

use crate::SpeechError;
use tokio::runtime::{Handle, Runtime};

enum Inner {
    Borrowed(Handle),
    Owned(Option<Runtime>),
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Dropping a runtime from inside async code would panic; shut it
        // down in the background instead.
        if let Self::Owned(runtime) = self
            && let Some(runtime) = runtime.take()
        {
            runtime.shutdown_background();
        }
    }
}

/// The Tokio runtime a cloud backend uses. Clones share it.
///
/// Session workers are plain threads, so they block on the runtime from
/// outside it. Do not call a backend's blocking methods from inside a
/// Tokio task; use `spawn_blocking`.
#[derive(Clone)]
pub struct CloudRuntime {
    inner: Arc<Inner>,
}

impl CloudRuntime {
    /// Uses a runtime you already run.
    pub fn from_handle(handle: Handle) -> Self {
        Self {
            inner: Arc::new(Inner::Borrowed(handle)),
        }
    }

    /// Creates a multi-threaded runtime with `threads` workers, owned by
    /// this value and its clones.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for zero threads, or
    /// [`SpeechError::Backend`] if the runtime cannot start.
    pub fn owned(threads: usize) -> Result<Self, SpeechError> {
        if threads == 0 {
            return Err(SpeechError::InvalidInput(
                "a cloud runtime needs at least one thread".into(),
            ));
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .enable_all()
            .thread_name("speechkit-cloud")
            .build()
            .map_err(|e| SpeechError::backend("speechkit-cloud", false, e))?;
        Ok(Self {
            inner: Arc::new(Inner::Owned(Some(runtime))),
        })
    }

    /// A handle to the runtime.
    pub(crate) fn handle(&self) -> Handle {
        match &*self.inner {
            Inner::Borrowed(handle) => handle.clone(),
            Inner::Owned(Some(runtime)) => runtime.handle().clone(),
            Inner::Owned(None) => Handle::current(),
        }
    }

    /// Whether the runtime was created by [`owned`](Self::owned).
    pub(crate) fn is_owned(&self) -> bool {
        matches!(&*self.inner, Inner::Owned(_))
    }

    /// Runs `future` to completion from a thread outside the runtime.
    #[cfg_attr(
        not(any(feature = "openai", feature = "dashscope")),
        expect(dead_code, reason = "only the backends block on the runtime")
    )]
    pub(crate) fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.handle().block_on(future)
    }
}

impl std::fmt::Debug for CloudRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudRuntime")
            .field("owned", &self.is_owned())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_runtime_runs_futures() {
        let runtime = CloudRuntime::owned(1).unwrap();
        assert!(runtime.is_owned());
        assert_eq!(runtime.block_on(async { 1 + 1 }), 2);
        let clone = runtime.clone();
        drop(runtime);
        assert_eq!(clone.block_on(async { 3 }), 3);
        assert!(CloudRuntime::owned(0).is_err());
    }

    #[test]
    fn borrowed_handle_runs_futures() {
        let tokio = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let runtime = CloudRuntime::from_handle(tokio.handle().clone());
        assert!(!runtime.is_owned());
        let value = std::thread::spawn(move || runtime.block_on(async { 7 }))
            .join()
            .unwrap();
        assert_eq!(value, 7);
        assert!(format!("{:?}", CloudRuntime::owned(1).unwrap()).contains("owned: true"));
    }
}
