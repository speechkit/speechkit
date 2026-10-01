//! DashScope real-time recognition (the Paraformer WebSocket protocol).

mod client;
pub(crate) mod protocol;
pub(crate) mod state;

pub use client::{DashScopeAsr, DashScopeAsrConfig};
