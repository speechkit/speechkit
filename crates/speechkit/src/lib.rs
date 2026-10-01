#![doc = include_str!("../README.md")]
// The loom build leaves out `io` (see below), the only user of a few
// crate-private items, so they are dead there and nowhere else.
#![cfg_attr(
    speechkit_loom,
    expect(
        dead_code,
        unused_imports,
        reason = "the loom build leaves out io, the only user of some items"
    )
)]

pub mod asr;
pub mod audio;
mod speech;
pub mod tts;
pub mod vad;
pub mod wake;

pub use speech::{AudioBuffer, Deadline, Flow, RecvError, SampleRate, Secret, SpeechError};

/// Internals the benchmarks measure. Not part of the API, and not covered
/// by semver.
#[doc(hidden)]
pub mod bench {
    pub use crate::speech::resample::Resampler;
    pub use crate::tts::chunk::{Chunker, TextChunk};
}

#[cfg(feature = "sherpa")]
pub mod sherpa;

#[cfg(any(feature = "openai", feature = "dashscope"))]
pub mod cloud;

// Left out of loom builds: `speech::sync` swaps in loom's locks there, and
// `io` waits on std locks from cpal threads that run outside a loom model.
#[cfg(all(feature = "devices", not(speechkit_loom)))]
pub mod io;

#[cfg(feature = "server")]
pub mod server;
