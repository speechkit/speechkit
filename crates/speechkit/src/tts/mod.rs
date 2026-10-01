//! Speech synthesis: the engine, sessions, and the backend traits. Every
//! session follows the TTS contract, rules `T-01` to `T-11`, which
//! `speechkit-testkit` checks for every backend.

pub(crate) mod backend;
pub(crate) mod chunk;
mod engine;
pub(crate) mod options;
mod session;
pub(crate) mod types;

pub use backend::{TtsBackend, TtsStream};
pub(crate) use chunk::Chunker;
pub use engine::TtsEngine;
pub(crate) use options::validate;
pub use options::{TtsCapabilities, TtsLimits, TtsOptions};
#[cfg(feature = "devices")]
pub(crate) use session::PushedText;
pub use session::{TtsOutput, TtsSession};
pub use types::{Mark, TtsFailure, TtsResult, TtsSummary, TtsUpdate, Voice};
