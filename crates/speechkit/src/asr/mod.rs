//! Speech recognition: the engine, sessions, and the backend traits.
//!
//! An [`AsrEngine`] wraps one [`AsrBackend`] and runs bounded, isolated
//! [`AsrSession`]s on it. Every session follows the ASR contract, rules
//! `A-01` to `A-19`, whichever backend runs it; `speechkit-testkit` checks
//! them for every backend.

pub(crate) mod backend;
mod engine;
pub(crate) mod options;
pub(crate) mod post;
mod session;
pub(crate) mod types;

pub use backend::{AsrBackend, AsrEvent, AsrStream};
pub use engine::AsrEngine;
pub(crate) use options::validate;
pub use options::{AsrCapabilities, AsrLimits, AsrOptions};
pub use post::PostProcessor;
pub use session::{AsrEvents, AsrSession, AsrUpdates, PushError, PushErrorKind};
pub use types::{
    AsrFailure, AsrResult, AsrUpdate, LiveTranscript, Partial, Segment, Transcript, Turn,
    UtteranceId,
};
