//! Speech recognition backends.

pub(crate) mod offline;
mod recognizer;
mod streaming;

pub use recognizer::{Asr, AsrConfig};
pub(crate) use streaming::{
    Hotwords, PreparedBias, SherpaStreaming, StreamingConfig, prepare_bias,
};

use crate::SpeechError;

/// The rate every sherpa-onnx recognizer here is fed.
pub(crate) const RATE: i32 = 16_000;

/// Right context fed after the last sample of a stream, so a streaming
/// model can finish the last word. It is not counted as input.
pub(crate) const TAIL_PADDING: usize = 12_800;

/// A failure inside sherpa-onnx.
pub(crate) fn native(what: &str) -> SpeechError {
    SpeechError::backend("sherpa-onnx", false, what.to_owned())
}
