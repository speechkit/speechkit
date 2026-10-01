//! Direction-neutral building blocks shared by speech recognition and
//! synthesis: audio types, errors, secrets, execution slots, opening a
//! session, resampling, deadlines, and the types every stream shares. The
//! public ones are re-exported at the crate root.

pub(crate) mod audio;
pub(crate) mod deadline;
mod error;
pub(crate) mod opening;
pub(crate) mod resample;
mod secret;
pub(crate) mod slots;
mod stream;
pub(crate) mod sync;

pub use audio::{AudioBuffer, SampleRate};
pub use deadline::Deadline;
pub use error::SpeechError;
pub use secret::Secret;
pub use stream::{Flow, RecvError};
