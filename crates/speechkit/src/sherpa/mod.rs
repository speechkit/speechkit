#![doc = include_str!("README.md")]

pub(crate) mod asr;
pub(crate) mod bias;
pub(crate) mod config;
mod inspect;
pub(crate) mod kws;
pub(crate) mod layout;
pub mod process;
pub(crate) mod punct;
pub(crate) mod tts;
pub(crate) mod vad;

pub use asr::{Asr, AsrConfig};
pub use config::{AsrFamily, Inference, Provider};
pub use inspect::{ModelInfo, inspect};
pub use kws::{Keyword, KeywordSpotter, KeywordSpotterConfig};
pub use layout::{PunctuationFamily, TtsFamily};
pub use punct::{Punctuation, PunctuationConfig};
pub use tts::{Tts, TtsConfig};
pub use vad::{SileroVad, SileroVadConfig};

/// The error for a model in `dir` that sherpa-onnx refused to load.
fn load_failed(what: impl std::fmt::Display, dir: &std::path::Path) -> crate::SpeechError {
    crate::SpeechError::InvalidModel(format!(
        "sherpa-onnx could not load the {what} model in {}",
        dir.display()
    ))
}
