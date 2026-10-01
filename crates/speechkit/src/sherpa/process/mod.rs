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
//! handshake is skipped.

mod client;
mod config;
mod protocol;
mod serve;

pub use client::{IsolatedAsr, WorkerCommand};
pub use serve::{serve, worker_main};
