#![doc = include_str!("README.md")]

#[cfg(feature = "websocket")]
mod runtime;
#[cfg(feature = "websocket")]
pub use runtime::CloudRuntime;

#[cfg(feature = "dashscope")]
mod dashscope;
#[cfg(feature = "dashscope")]
pub use dashscope::{DashScopeAsr, DashScopeAsrConfig, DashScopeTts, DashScopeTtsConfig};
#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::{
    OpenAiRealtime, OpenAiRealtimeConfig, OpenAiSpeech, OpenAiSpeechConfig, OpenAiTranscription,
    OpenAiTranscriptionConfig,
};
// The internal `websocket` feature alone builds only the runtime.
#[cfg(any(feature = "openai", feature = "dashscope"))]
mod pcm;
#[cfg(any(feature = "openai", feature = "dashscope"))]
mod ws;
