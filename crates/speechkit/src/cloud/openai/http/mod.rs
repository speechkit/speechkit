//! `POST /v1/audio/transcriptions`, split into pure protocol functions and
//! the HTTP transport.

mod client;
pub(crate) mod protocol;

pub use client::{OpenAiTranscription, OpenAiTranscriptionConfig};
