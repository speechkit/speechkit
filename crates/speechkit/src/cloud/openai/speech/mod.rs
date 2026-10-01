//! `POST /v1/audio/speech`, split into pure protocol functions and the
//! HTTP transport.

mod client;
pub(crate) mod protocol;

pub use client::{OpenAiSpeech, OpenAiSpeechConfig};
