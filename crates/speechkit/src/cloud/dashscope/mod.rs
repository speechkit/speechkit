//! Alibaba Cloud DashScope (Model Studio) speech services.

mod asr;
mod task;
mod tts;

pub use asr::{DashScopeAsr, DashScopeAsrConfig};
pub use tts::{DashScopeTts, DashScopeTtsConfig};
