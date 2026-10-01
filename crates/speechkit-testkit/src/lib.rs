#![doc = include_str!("../README.md")]
#![expect(
    clippy::panic,
    clippy::missing_panics_doc,
    reason = "contract checks are assertions: they panic to fail the calling test"
)]

pub mod asr;
pub mod cer;
pub mod contract;
mod gate;
pub mod gates;
pub mod tts;
pub mod vad;
pub mod wake;

pub use gate::Gate;

use std::time::{Duration, Instant};

/// Root-mean-square level of `samples`. Empty input is 0.0.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an RMS of f32 samples fits in f32"
    )]
    let level = (sum / samples.len() as f64).sqrt() as f32;
    level
}

/// A deadline `seconds` from now.
pub fn secs(seconds: u64) -> Instant {
    Instant::now() + Duration::from_secs(seconds)
}

/// Polls `condition` every millisecond until it holds or `timeout` passes.
/// Returns whether it held.
pub fn eventually(timeout: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    condition()
}
