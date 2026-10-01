//! Reusable contract suites. Every backend runs the same checks.
//!
//! Each rule has an ID: `A-xx` for recognition ([`asr`]), `T-xx` for
//! synthesis ([`tts`]), and `D-xx` for devices (`devices`, behind the
//! `devices` feature). A check is named after its rule, such as
//! `a07_endpointing`, and its doc comment states the rule. Code that
//! keeps a rule cites its ID. A change to a rule changes its check in the
//! same commit.

pub mod asr;
// Uses `speechkit::io`, which loom builds leave out.
#[cfg(all(feature = "devices", not(speechkit_loom)))]
pub mod devices;
pub mod tts;

use std::time::Duration;

/// How long a check waits for background work to wind down.
const SETTLE: Duration = Duration::from_secs(10);
