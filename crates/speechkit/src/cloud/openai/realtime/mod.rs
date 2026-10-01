//! OpenAI Realtime transcription sessions over WebSocket, at 24 kHz PCM16.

mod client;
pub(crate) mod protocol;
pub(crate) mod state;

pub use client::{OpenAiRealtime, OpenAiRealtimeConfig};
