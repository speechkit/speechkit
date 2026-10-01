//! Synchronization primitives, swapped for loom's under `--cfg speechkit_loom`.

#[cfg(speechkit_loom)]
pub(crate) use loom::sync::{Arc, Condvar, Mutex, MutexGuard};
#[cfg(not(speechkit_loom))]
pub(crate) use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Locks `mutex`, recovering the guard if another thread panicked while
/// holding it. Every critical section in speechkit leaves its data
/// consistent before any call that could panic, so the data stays valid.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
