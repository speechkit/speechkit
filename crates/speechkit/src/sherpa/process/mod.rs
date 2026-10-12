//! Crash isolation: run a backend in a child process.
//!
//! sherpa-onnx aborts the process on a corrupt model, and a native crash
//! cannot be caught. [`IsolatedAsr`] runs the recognizer in a worker
//! process (`speechkit-worker`), so a crash fails only the sessions in
//! flight, with a retryable error, and the next session starts a new
//! worker.
//!
//! The worker speaks a small protocol over its stdin and stdout: after a
//! handshake line, length-prefixed frames, each holding a version byte and
//! a postcard-encoded message. Anything a native library prints before the
//! handshake is skipped. After it, stdout belongs to the protocol: text
//! written there breaks a frame, which the parent treats as a crash, so it
//! kills the worker and fails its session with a retryable error. sherpa-onnx
//! and onnxruntime log to stderr, which the worker shares with the parent.

mod client;
mod config;
mod protocol;
mod serve;

pub use client::{IsolatedAsr, WorkerCommand};
pub use serve::{serve, worker_main};

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `mutex`, recovering the guard if a thread panicked while holding
/// it. These are std locks even in loom builds, since no loom model runs
/// a worker.
fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
