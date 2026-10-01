//! DashScope speech synthesis (the `CosyVoice` WebSocket protocol).

mod client;
pub(crate) mod protocol;

pub use client::{DashScopeTts, DashScopeTtsConfig};
